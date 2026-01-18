import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
from functools import partial
import json

import gradio as gr
import numpy as np
import torch
from PIL import Image

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer, add_watermark_np
from inversion_utils import InversionModule

parser = ArgumentParser()
parser.add_argument('--share', action='store_true', default=False)
parser.add_argument('--cache-dir', type=str, default='./checkpoints')
parser.add_argument(
    '--listen',
    action='store_true',
    help='launch gradio with 0.0.0.0 as server name, allowing to respond to network requests',
)
args = parser.parse_args()

cache_dir = args.cache_dir

device = 'cuda'
inversion_module = None


def get_inversion_module():
    global inversion_module
    if inversion_module is None:
        inversion_module = InversionModule(device=device)
    return inversion_module


def invert_uploaded_image(image, method, global_state):
    if image is None:
        yield global_state, None
        return

    try:
        inv_module = get_inversion_module()
        renderer = global_state['renderer']

        if not hasattr(renderer, 'G'):
            print("Generator not initialized, initializing now...")
            init_images(global_state)

        G = renderer.G

        from PIL import ImageDraw, ImageFont

        def create_progress_image(step, total_steps, loss_info=""):
            img_size = 1024
            img = Image.new('RGB', (img_size, img_size), color=(30, 30, 30))
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype('arial.ttf', 48)
                small_font = ImageFont.truetype('arial.ttf', 32)
            except BaseException:
                font = ImageFont.load_default()
                small_font = font

            bar_width = 800
            bar_height = 60
            bar_x = (img_size - bar_width) // 2
            bar_y = 400

            draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                           fill=(60, 60, 60), outline=(100, 100, 100), width=2)

            progress = step / total_steps
            fill_width = int(bar_width * progress)
            draw.rectangle([bar_x, bar_y, bar_x + fill_width, bar_y + bar_height],
                           fill=(0, 150, 255))

            text = f"Optimizing: {step}/{total_steps} ({progress * 100:.1f}%)"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((img_size // 2 - text_width // 2, 300), text, fill=(255, 255, 255), font=font)

            if loss_info:
                bbox = draw.textbbox((0, 0), loss_info, font=small_font)
                text_width = bbox[2] - bbox[0]
                draw.text((img_size // 2 - text_width // 2, 500), loss_info, fill=(200, 200, 200), font=small_font)

            return img

        if method == 'PTI':
            num_steps = 450 + 350
            print(f"Starting PTI inversion with {num_steps} total steps...")
            yield global_state, create_progress_image(0, num_steps, "Initializing PTI...")

            from inversion_utils import InversionModule
            inv_module = InversionModule(device=device)

            progress_updates = []

            def progress_callback(step, loss):
                loss_info = f"Loss: {loss:.4f}"
                progress_updates.append((step, loss_info))

            try:
                w_latent, G_tuned = inv_module.pti_invert(
                    image, G,
                    num_pti_steps=350,
                    initial_inversion_steps=450,
                    pti_lr=5e-4,
                    initial_lr=8e-3,
                    progress_callback=progress_callback
                )

                for step, loss_info in progress_updates:
                    yield global_state, create_progress_image(step, num_steps, loss_info)

                G = G_tuned
                print(f'PTI inversion completed!')
            except Exception as e:
                print(f"PTI inversion failed: {e}, falling back to optimization")
                method = 'Optimization'
                num_steps = 3000

        if method == 'Optimization':
            num_steps = 3000
            print(f"Starting optimization inversion with {num_steps} steps...")
            yield global_state, create_progress_image(0, num_steps, "Initializing...")

            from inversion_utils import InversionModule
            inv_module = InversionModule(device=device)

            import copy
            from torchvision import transforms
            import torch.nn.functional as F

            target_h = G.img_resolution
            target_w = G.img_resolution
            transform = transforms.Compose([
                transforms.Resize((target_h, target_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)

            G_copy = copy.deepcopy(G).eval().requires_grad_(False).to(device)
            z_samples = torch.randn([10000, G_copy.z_dim], device=device)
            with torch.no_grad():
                w_samples = G_copy.mapping(z_samples, None)[:, :1, :]
                w_avg = w_samples.mean(dim=0, keepdim=True)

            w = w_avg.detach().clone().repeat(1, G_copy.num_ws, 1)
            w.requires_grad = True

            optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

            try:
                from lpips import LPIPS
                lpips_fn = LPIPS(net='alex').to(device).eval()
                for param in lpips_fn.parameters():
                    param.requires_grad = False
            except BaseException:
                lpips_fn = None

            best_loss = float('inf')
            best_w = w.detach().clone()

            for step in range(num_steps):
                synth_img = G_copy.synthesis(w, noise_mode='const')
                mse_loss = F.mse_loss(synth_img, img_tensor)

                if lpips_fn is not None:
                    lpips_loss = lpips_fn(synth_img, img_tensor).mean()
                    total_loss = mse_loss * 1.0 + lpips_loss * 1.0
                else:
                    total_loss = mse_loss

                if step < num_steps // 2:
                    w_reg = ((w - w_avg.repeat(1, G_copy.num_ws, 1)) ** 2).mean() * 0.01
                    total_loss = total_loss + w_reg

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_w = w.detach().clone()

                if step % 50 == 0 or step == num_steps - 1:
                    progress_img = create_progress_image(step + 1, num_steps, f"Loss: {total_loss.item():.4f}")
                    yield global_state, progress_img
                    print(f'Step {step}/{num_steps}: Loss={total_loss.item():.4f}')

            w_latent = best_w
            print(f'Optimization completed! Best loss: {best_loss:.4f}')

        global_state['inversion_mode'] = True
        global_state['uploaded_image'] = image
        global_state['renderer'].w_load = w_latent
        global_state['renderer'].init_network(
            global_state['generator_params'],
            valid_checkpoints_dict[global_state['pretrained_weight']],
            global_state['params']['seed'],
            w_latent,
            global_state['params']['latent_space'] == 'w+',
            'const',
            global_state['params']['trunc_psi'],
            global_state['params']['trunc_cutoff'],
            None,
            global_state['params']['lr']
        )

        global_state['renderer']._render_drag_impl(
            global_state['generator_params'],
            is_drag=False,
            to_pil=True
        )

        reconstructed_image = global_state['generator_params'].image
        global_state['images']['image_orig'] = reconstructed_image
        global_state['images']['image_raw'] = reconstructed_image
        global_state['images']['image_show'] = Image.fromarray(
            add_watermark_np(np.array(reconstructed_image))
        )

        clear_state(global_state)

        print("Successfully inverted image")
        yield global_state, global_state['images']['image_show']

    except Exception as e:
        print(f"Error during inversion: {e}")
        import traceback
        traceback.print_exc()
        error_img = Image.new('RGB', (512, 512), color=(50, 0, 0))
        draw = ImageDraw.Draw(error_img)
        draw.text((150, 250), f"Error: {str(e)[:50]}", fill=(255, 255, 255))
        yield global_state, error_img


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def init_images(global_state):
    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    state['renderer'].init_network(
        state['generator_params'],
        valid_checkpoints_dict[state['pretrained_weight']],
        state['params']['seed'],
        None,
        state['params']['latent_space'] == 'w+',
        'const',
        state['params']['trunc_psi'],
        state['params']['trunc_cutoff'],
        None,
        state['params']['lr'],
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        add_watermark_np(np.array(init_image)))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state


def update_image_draw(image, points, mask, show_mask, global_state=None):

    image_draw = draw_points_on_image(image, points)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw


def serialize_points(points):
    payload = []
    for idx in sorted(points.keys()):
        p = points[idx]
        start = p.get('start')
        target = p.get('target')
        if start is None or target is None:
            continue
        payload.append({'start': [int(start[0]), int(start[1])],
                        'target': [int(target[0]), int(target[1])]})
    return json.dumps(payload, ensure_ascii=False)


def deserialize_points(points_json):
    data = json.loads(points_json) if points_json else []
    points = {}
    for i, item in enumerate(data):
        start = item.get('start')
        target = item.get('target')
        if start is None or target is None:
            continue
        points[i] = {
            'start': [int(round(start[0])), int(round(start[1]))],
            'target': [int(round(target[0])), int(round(target[1]))],
        }
    return points


def preprocess_mask_info(global_state, image):
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    if (mask == 1).all():
        mask = last_mask
    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, '
              'do nothing to mask.')

    global_state['mask'] = updated_mask
    return global_state


valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print(f'File under cache_dir ({cache_dir}):')
print(os.listdir(cache_dir))
print('Valid checkpoint file:')
print(valid_checkpoints_dict)

init_pkl = 'stylegan2_lions_512_pytorch'

with gr.Blocks(css=".top-align-row{align-items:flex-start !important;} .top-align-col{align-self:flex-start !important;} .drag-image img{margin-top:0 !important;}") as app:

    global_state = gr.State({
        "images": {
        },
        "temporal_params": {
        },
        'mask':
        None,
        'last_mask': None,
        'show_mask': True,
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
            "tracker_type": "NN",
            "tracker_lambda": 0.5,
            "max_steps": 200,
            "stop_thresh_px": 2,
            "feature_blend": False,
            "blend_ratio": 0.5,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl,
        'inversion_mode': False,
        'uploaded_image': None,
    })
    global_state = init_images(global_state)

    with gr.Row():

        with gr.Row(elem_classes=["top-align-row"]):

            with gr.Column(scale=3):
                with gr.Row():

                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Pickle', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_pretrained_dropdown = gr.Dropdown(
                            choices=list(valid_checkpoints_dict.keys()),
                            label="Pretrained Model",
                            value=init_pkl,
                        )

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Latent', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_seed_number = gr.Number(
                            value=global_state.value['params']['seed'],
                            interactive=True,
                            label="Seed",
                        )
                        form_lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            interactive=True,
                            label="Step Size")

                        form_tracker_type = gr.Radio(
                            ['NN', 'RAFT', 'HYBRID', 'MULTISCALE'],
                            value=global_state.value['params']['tracker_type'],
                            interactive=True,
                            label='Tracking Method',
                        )

                        form_tracker_lambda = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=global_state.value['params']['tracker_lambda'],
                            label='Tracker Lambda (Hybrid distance weight)',
                            interactive=True,
                            visible=global_state.value['params']['tracker_type'] == 'HYBRID',
                        )

                        form_max_steps = gr.Number(
                            value=global_state.value['params']['max_steps'],
                            interactive=True,
                            label="Max Steps",
                        )

                        form_stop_thresh = gr.Number(
                            value=global_state.value['params']['stop_thresh_px'],
                            interactive=True,
                            label="Stop Threshold (px)",
                            info="Distance to target below this will stop early",
                        )

                        with gr.Row():
                            with gr.Column(scale=2, min_width=10):
                                form_reset_image = gr.Button("Reset Image")
                            with gr.Column(scale=3, min_width=10):
                                form_latent_space = gr.Radio(
                                    ['w', 'w+'],
                                    value=global_state.value['params']
                                    ['latent_space'],
                                    interactive=True,
                                    label='Latent space to optimize',
                                    show_label=False,
                                )

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Upload', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        upload_image = gr.Image(
                            type="pil",
                            label="Upload Real Image",
                            interactive=True
                        )
                        inversion_method = gr.Radio(
                            choices=['Optimization', 'PTI'],
                            value='Optimization',
                            label='Inversion Method',
                            info='Optimization: Fast (3000 steps). PTI: Higher quality (800 steps total)',
                            interactive=True
                        )
                        invert_button = gr.Button("Invert Image")

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Drag', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                enable_add_points = gr.Button('Add Points')
                            with gr.Column(scale=1, min_width=10):
                                undo_points = gr.Button('Reset Points')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_start_btn = gr.Button("Start")
                            with gr.Column(scale=1, min_width=10):
                                form_stop_btn = gr.Button("Stop")

                        form_steps_number = gr.Number(value=0,
                                                      label="Steps",
                                                      interactive=False)

                        form_distance_info = gr.Textbox(
                            label="Distances (px)",
                            interactive=False,
                            value="",
                        )

                        form_download_result_file = gr.File(
                            label="Download Result",
                            visible=False,
                        )

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Points IO', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            form_export_points = gr.Button('Export Points')
                            form_import_points = gr.Button('Load Points')
                        form_points_json = gr.Textbox(
                            label='Points JSON',
                            lines=3,
                            placeholder='[{"start": [x, y], "target": [x, y]}]',
                            interactive=True,
                        )

                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Mask', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        enable_add_mask = gr.Button('Edit Flexible Area')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_reset_mask_btn = gr.Button("Reset mask")
                            with gr.Column(scale=1, min_width=10):
                                show_mask = gr.Checkbox(
                                    label='Show Mask',
                                    value=global_state.value['show_mask'],
                                    show_label=False)

                        with gr.Row():
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"]
                                ["motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )

                        with gr.Row():
                            feature_blend_checkbox = gr.Checkbox(
                                label='Enable Feature Blend',
                                value=global_state.value["params"]["feature_blend"],
                                show_label=True)

                            blend_ratio_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=global_state.value["params"]["blend_ratio"],
                                step=0.05,
                                label="Blend Ratio",
                                interactive=True)

                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                    visible=False)

            with gr.Column(scale=8, elem_classes=["top-align-col"]):
                form_image = ImageMask(
                    value=global_state.value['images']['image_show'],
                    brush_radius=20,
                    width=768,
                    height=768,
                    show_label=False,
                    elem_classes=["drag-image"],
                )

    gr.Markdown("""
        ## Quick Start

        ### Option 1: Generate from Random Seed
        1. Select desired `Pretrained Model` and adjust `Seed` to generate an
           initial image.
        2. Click on image to add control points.
        3. Click `Start` and enjoy it!

        ### Option 2: Upload Real Image (NEW!)
        1. Upload your own image in the `Upload Real Image` section.
        2. Choose inversion method:
           * `Optimization`: Fast (3000 steps, ~5-10 minutes)
           * `PTI`: Higher quality (initial + fine-tuning, ~15-20 minutes)
        3. Click `Invert Image` and watch real-time progress.
        4. After completion, add control points and click `Start` to edit!

        ## Advance Usage

        1. Change `Step Size` to adjust learning rate in drag optimization.
        2. Select `w` or `w+` to change latent space to optimize:
        * Optimize on `w` space may cause greater influence to the image.
        * Optimize on `w+` space may work slower than `w`, but usually achieve
          better results.
        * Note that changing the latent space will reset the image, points and
          mask (this has the same effect as `Reset Image` button).
        3. Click `Edit Flexible Area` to create a mask and constrain the
           unmasked region to remain unchanged.
        """)
    gr.HTML("""
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <img src="https://avatars.githubusercontent.com/u/10245193?s=200&v=4" height="20" width="20" style="display:inline;">
        <a href="https://github.com/open-mmlab/mmagic">OpenMMLab MMagic</a>
        </div>
        """)

    def on_change_pretrained_dropdown(pretrained_value, global_state):

        global_state['pretrained_weight'] = pretrained_value
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]['image_show']

    form_pretrained_dropdown.change(
        on_change_pretrained_dropdown,
        inputs=[form_pretrained_dropdown, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_reset_image(global_state):

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_reset_image.click(
        on_click_reset_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    invert_button.click(
        invert_uploaded_image,
        inputs=[upload_image, inversion_method, global_state],
        outputs=[global_state, form_image],
        show_progress="hidden"
    )

    def on_change_update_image_seed(seed, global_state):

        global_state["params"]["seed"] = int(seed)
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_seed_number.change(
        on_change_update_image_seed,
        inputs=[form_seed_number, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_latent_space(latent_space, global_state):

        global_state['params']['latent_space'] = latent_space
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_latent_space.change(on_click_latent_space,
                             inputs=[form_latent_space, global_state],
                             outputs=[global_state, form_image])

    form_lambda_number.change(
        partial(on_change_single_global_state, ["params", "motion_lambda"]),
        inputs=[form_lambda_number, global_state],
        outputs=[global_state],
    )

    feature_blend_checkbox.change(
        partial(on_change_single_global_state, ["params", "feature_blend"]),
        inputs=[feature_blend_checkbox, global_state],
        outputs=[global_state],
    )

    blend_ratio_slider.change(
        partial(on_change_single_global_state, ["params", "blend_ratio"]),
        inputs=[blend_ratio_slider, global_state],
        outputs=[global_state],
    )

    def on_change_lr(lr, global_state):
        if lr == 0:
            print('lr is 0, do nothing.')
            return global_state
        else:
            global_state["params"]["lr"] = lr
            renderer = global_state['renderer']
            renderer.update_lr(lr)
            print('New optimizer: ')
            print(renderer.w_optim)
        return global_state

    def on_change_max_steps(v, global_state):
        global_state["params"]["max_steps"] = int(v)
        return global_state

    def on_change_stop_thresh(v, global_state):
        global_state["params"]["stop_thresh_px"] = float(v)
        return global_state

    form_lr_number.change(
        on_change_lr,
        inputs=[form_lr_number, global_state],
        outputs=[global_state],
    )

    def on_change_tracker_type(tracker_type, global_state):
        global_state['params']['tracker_type'] = tracker_type
        return global_state, gr.Slider.update(visible=(tracker_type == 'HYBRID'))

    form_tracker_type.change(
        on_change_tracker_type,
        inputs=[form_tracker_type, global_state],
        outputs=[global_state, form_tracker_lambda],
    )

    form_tracker_lambda.change(
        partial(on_change_single_global_state, ["params", "tracker_lambda"]),
        inputs=[form_tracker_lambda, global_state],
        outputs=[global_state],
    )

    form_max_steps.change(
        on_change_max_steps,
        inputs=[form_max_steps, global_state],
        outputs=[global_state],
    )

    form_stop_thresh.change(
        on_change_stop_thresh,
        inputs=[form_stop_thresh, global_state],
        outputs=[global_state],
    )

    def on_click_start(global_state, image, max_steps_val, stop_thresh_val):
        if max_steps_val is not None:
            global_state['params']['max_steps'] = int(max_steps_val)
        if stop_thresh_val is not None:
            global_state['params']['stop_thresh_px'] = float(stop_thresh_val)

        def sanitize_name(s):
            return ''.join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in s)

        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        global_state = preprocess_mask_info(global_state, image)
        if len(global_state["points"]) == 0:
            image_raw = global_state['images']['image_raw']
            update_image_draw(
                image_raw,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )

            yield (
                global_state,
                0,
                global_state['images']['image_show'],
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=False),
                gr.Dropdown.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Radio.update(interactive=True),
                gr.Slider.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Textbox.update(value=""),
                None,
            )
        else:

            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask = torch.tensor(global_state['mask']).float()
            drag_mask = 1 - mask

            renderer: Renderer = global_state["renderer"]
            global_state['temporal_params']['stop'] = False
            global_state['editing_state'] = 'running'

            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            print('Running with:')
            print(f'    Source: {p_in_pixels}')
            print(f'    Target: {t_in_pixels}')
            step_idx = 0
            while True:
                if global_state["temporal_params"]["stop"]:
                    break

                renderer._render_drag_impl(
                    global_state['generator_params'],
                    p_to_opt,
                    t_to_opt,
                    drag_mask,
                    global_state['params']['motion_lambda'],
                    reg=0,
                    feature_idx=5,
                    r1=global_state['params']['r1_in_pixels'],
                    r2=global_state['params']['r2_in_pixels'],
                    trunc_psi=global_state['params']['trunc_psi'],
                    is_drag=True,
                    to_pil=True,
                    tracker_type=global_state['params']['tracker_type'],
                    tracker_lambda=global_state['params']['tracker_lambda'],
                    stop_thresh_px=global_state['params']['stop_thresh_px'],
                    feature_blend=global_state['params'].get('feature_blend', False),
                    blend_ratio=global_state['params'].get('blend_ratio', 0.5))

                if step_idx % global_state['draw_interval'] == 0:
                    print('Current Source:')
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                                   t_to_opt):
                        global_state["points"][key_point]["start_temp"] = [
                            p_i[1],
                            p_i[0],
                        ]
                        global_state["points"][key_point]["target"] = [
                            t_i[1],
                            t_i[0],
                        ]
                        start_temp = global_state["points"][key_point][
                            "start_temp"]
                        print(f'    {start_temp}')

                    image_result = global_state['generator_params']['image']
                    image_draw = update_image_draw(
                        image_result,
                        global_state['points'],
                        global_state['mask'],
                        global_state['show_mask'],
                        global_state,
                    )
                    global_state['images']['image_raw'] = image_result

                dists = []
                for p_i, t_i in zip(p_to_opt, t_to_opt):
                    dy = p_i[0] - t_i[0]
                    dx = p_i[1] - t_i[1]
                    dists.append((dy * dy + dx * dx) ** 0.5)
                if len(dists) > 0:
                    mean_dist = sum(dists) / len(dists)
                    dist_text = f"mean={mean_dist:.2f}px; per-point={['{:.2f}'.format(d) for d in dists]}"
                else:
                    mean_dist = None
                    dist_text = "No points"

                yield (
                    global_state,
                    step_idx,
                    global_state['images']['image_show'],
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Radio.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=True),
                    gr.Dropdown.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Radio.update(interactive=False),
                    gr.Slider.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Checkbox.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Textbox.update(value=dist_text),
                    None,
                )
                step_idx += 1

                if step_idx >= global_state['params']['max_steps']:
                    print(f"Reached max_steps={global_state['params']['max_steps']}, stopping.")
                    break

            image_result = global_state['generator_params']['image']
            global_state['images']['image_raw'] = image_result
            image_draw = update_image_draw(image_result,
                                           global_state['points'],
                                           global_state['mask'],
                                           global_state['show_mask'],
                                           global_state)

            model_name = sanitize_name(global_state['pretrained_weight'])
            tracker_type = sanitize_name(global_state['params']['tracker_type'])
            lambda_part = ''
            if tracker_type == 'HYBRID':
                lambda_part = f"_{global_state['params']['tracker_lambda']:.2f}"
            if mean_dist is not None:
                dist_part = f"_{mean_dist:.2f}px"
            else:
                dist_part = "_NA"
            filename = f"{model_name}_{tracker_type}{lambda_part}{dist_part}.png"
            filename = sanitize_name(filename)
            download_path = osp.join(tempfile.gettempdir(), filename)
            try:
                image_draw.save(download_path)
                download_value = download_path
                download_visible = True
            except Exception as e:
                print(f"Failed to save download file: {e}")
                download_value = None
                download_visible = False

            global_state['editing_state'] = 'add_points'

            yield (
                global_state,
                0,
                global_state['images']['image_show'],
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=False),
                gr.Dropdown.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Radio.update(interactive=True),
                gr.Slider.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Textbox.update(),
                gr.File.update(value=download_value, visible=download_visible),
            )

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image, form_max_steps, form_stop_thresh],
        outputs=[
            global_state,
            form_steps_number,
            form_image,
            form_reset_image,
            enable_add_points,
            enable_add_mask,
            undo_points,
            form_reset_mask_btn,
            form_latent_space,
            form_start_btn,
            form_stop_btn,
            form_pretrained_dropdown,
            form_seed_number,
            form_lr_number,
            form_tracker_type,
            form_tracker_lambda,
            form_max_steps,
            form_stop_thresh,
            show_mask,
            form_lambda_number,
            form_distance_info,
            form_download_result_file,
        ],
    )

    def on_click_stop(global_state):
        global_state["temporal_params"]["stop"] = True

        return global_state, gr.Button.update(interactive=False)

    form_stop_btn.click(on_click_stop,
                        inputs=[global_state],
                        outputs=[global_state, form_stop_btn])

    form_draw_interval_number.change(
        partial(
            on_change_single_global_state,
            "draw_interval",
            map_transform=lambda x: int(x),
        ),
        inputs=[form_draw_interval_number, global_state],
        outputs=[global_state],
    )

    def on_click_remove_point(global_state):
        choice = global_state["curr_point"]
        del global_state["points"][choice]

        choices = list(global_state["points"].keys())

        if len(choices) > 0:
            global_state["curr_point"] = choices[0]

        return (
            gr.Dropdown.update(choices=choices, value=choices[0]),
            global_state,
        )

    def on_click_reset_mask(global_state):
        global_state['mask'] = np.ones(
            (
                global_state["images"]["image_raw"].size[1],
                global_state["images"]["image_raw"].size[0],
            ),
            dtype=np.uint8,
        )
        image_draw = update_image_draw(global_state['images']['image_raw'],
                                       global_state['points'],
                                       global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    form_reset_mask_btn.click(
        on_click_reset_mask,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_enable_draw(global_state, image):
        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    def on_click_remove_draw(global_state, image):
        global_state = preprocess_mask_info(global_state, image)
        global_state['edinting_state'] = 'remove_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    enable_add_mask.click(on_click_enable_draw,
                          inputs=[global_state, form_image],
                          outputs=[
                              global_state,
                              form_image,
                          ])

    def on_click_add_point(global_state, image: dict):
        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_points'
        mask = global_state['mask']
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'], mask,
                                       global_state['show_mask'], global_state)

        return (global_state,
                gr.Image.update(value=image_draw, interactive=False))

    enable_add_points.click(on_click_add_point,
                            inputs=[global_state, form_image],
                            outputs=[global_state, form_image])

    def on_click_export_points(global_state):
        return serialize_points(global_state['points'])

    form_export_points.click(
        on_click_export_points,
        inputs=[global_state],
        outputs=[form_points_json],
    )

    def on_click_import_points(global_state, points_json):
        try:
            points = deserialize_points(points_json)
        except Exception as e:
            print(f'Failed to load points: {e}')
            return global_state, global_state['images']['image_show'], form_points_json

        global_state['points'] = points
        global_state['editing_state'] = 'add_points'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    form_import_points.click(
        on_click_import_points,
        inputs=[global_state, form_points_json],
        outputs=[global_state, form_image],
    )

    def on_click_image(global_state, evt: gr.SelectData):
        xy = evt.index
        if global_state['editing_state'] != 'add_points':
            print(f'In {global_state["editing_state"]} state. '
                  'Do not add points.')

            return global_state, global_state['images']['image_show']

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')
        elif points[point_idx].get('target', None) is None:
            points[point_idx]['target'] = xy
            print(f'Click Image - Target - {xy}')
        else:
            points[point_idx + 1] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )

        return global_state, image_draw

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_clear_points(global_state):
        clear_state(global_state, target='point')

        renderer: Renderer = global_state["renderer"]
        renderer.feat_refs = None

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, {}, global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    undo_points.click(on_click_clear_points,
                      inputs=[global_state],
                      outputs=[global_state, form_image])

    def on_click_show_mask(global_state, show_mask):
        global_state['show_mask'] = show_mask

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[global_state, show_mask],
        outputs=[global_state, form_image],
    )

gr.close_all()
app.queue(concurrency_count=3, max_size=20)
app.launch(share=args.share, server_name="0.0.0.0" if args.listen else "127.0.0.1")
