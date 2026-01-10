# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error
from raft_tracker import RAFTTracker

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                print(data[key].init_args)
                print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'G_ema')
        self.G = G
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Generate random latents.
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        self.w0 = w.detach().clone()
        self.w_plus = w_plus
        if w_plus:
            self.w = w.detach()
        else:
            self.w = w[:, 0, :].detach()
        self.w.requires_grad = True
        self.w_optim = torch.optim.Adam([self.w], lr=lr)

        self.feat_refs = None
        self.points0_pt = None
        self.raft_tracker = None
        self.prev_img_for_raft = None

    def update_lr(self, lr):

        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

    def _render_drag_impl(self, res,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        stop_thresh_px  = 2,
        **kwargs
    ):
        G = self.G
        ws = self.w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        if reset:
            self.feat_refs = None
            self.points0_pt = None
            self.prev_img_for_raft = None
            self.raft_ref_img = None
            self.raft_ref_points = None
        self.points = points

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)

        h, w = G.img_resolution, G.img_resolution

        if is_drag:
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            xx, yy = torch.meshgrid(X, Y)
            tracker_type = kwargs.get('tracker_type', 'NN')
            tracker_lambda = kwargs.get('tracker_lambda', 0.5)

            # Base feature for motion supervision
            feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')

            # Build tracking feature (single-scale or multi-scale fusion)
            track_feat_resize = feat_resize
            if tracker_type in ['MULTISCALE', 'HYBRID', 'HYBRID_LAMBDA']:
                fuse_indices = [3, 5, 7, 9]
                fuse_indices = [idx for idx in fuse_indices if idx < len(feat)]
                if feature_idx not in fuse_indices:
                    fuse_indices.append(feature_idx)
                feats_to_fuse = [F.interpolate(feat[idx], [h, w], mode='bilinear') for idx in fuse_indices]
                track_feat_resize = torch.cat(feats_to_fuse, dim=1)

            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat[feature_idx].detach(), [h, w], mode='bilinear')
                self.feat_refs = []
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    self.feat_refs.append(track_feat_resize.detach()[:,:,py,px])
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2

            # Point tracking
            if tracker_type == 'RAFT' or tracker_type == 'HYBRID':
                if self.raft_tracker is None:
                    self.raft_tracker = RAFTTracker(device=self._device)

                # Use a fixed reference image/points from drag start to accumulate full displacement.
                if self.raft_ref_img is None:
                    self.raft_ref_img = img.detach()
                    self.raft_ref_points = torch.tensor(points, device=self._device, dtype=torch.float32)

                with torch.no_grad():
                    raft_pred = self.raft_tracker.update_points(self.raft_ref_img, img, self.raft_ref_points)

                if tracker_type == 'RAFT':
                    for j in range(len(points)):
                        points[j] = raft_pred[j].tolist()
                else:  # HYBRID: use RAFT prediction to guide NN search window
                    guided_points = raft_pred.detach().cpu().tolist()
                    with torch.no_grad():
                        for j, point in enumerate(points):
                            guide = guided_points[j]
                            r = round(r2 / 512 * h)
                            up = max(int(round(guide[0])) - r, 0)
                            down = min(int(round(guide[0])) + r + 1, h)
                            left = max(int(round(guide[1])) - r, 0)
                            right = min(int(round(guide[1])) + r + 1, w)
                            feat_patch = track_feat_resize[:,:,up:down,left:right]
                            L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                            # distance regularizer towards RAFT guide
                            yy_local, xx_local = torch.meshgrid(
                                torch.arange(up, down, device=self._device),
                                torch.arange(left, right, device=self._device),
                                indexing='ij'
                            )
                            dist = ((yy_local - guide[0])**2 + (xx_local - guide[1])**2).sqrt()
                            dist = dist.unsqueeze(0)  # broadcast over channel dimension after norm
                            r_norm = r + 1e-6
                            cost = L2 + tracker_lambda * (dist / r_norm)
                            _, idx = torch.min(cost.view(1,-1), -1)
                            width = right - left
                            point = [idx.item() // width + up, idx.item() % width + left]
                            points[j] = point

            elif tracker_type == 'HYBRID_LAMBDA' or tracker_type == 'MULTISCALE':
                # HYBRID_LAMBDA here means: no RAFT; but distance prior is zero (falls back to NN). For clarity, treat as NN on multiscale.
                with torch.no_grad():
                    for j, point in enumerate(points):
                        r = round(r2 / 512 * h)
                        up = max(point[0] - r, 0)
                        down = min(point[0] + r + 1, h)
                        left = max(point[1] - r, 0)
                        right = min(point[1] + r + 1, w)
                        feat_patch = track_feat_resize[:,:,up:down,left:right]
                        L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                        _, idx = torch.min(L2.view(1,-1), -1)
                        width = right - left
                        point = [idx.item() // width + up, idx.item() % width + left]
                        points[j] = point

            else:
                # Point tracking with feature matching
                with torch.no_grad():
                    for j, point in enumerate(points):
                        r = round(r2 / 512 * h)
                        up = max(point[0] - r, 0)
                        down = min(point[0] + r + 1, h)
                        left = max(point[1] - r, 0)
                        right = min(point[1] + r + 1, w)
                        feat_patch = track_feat_resize[:,:,up:down,left:right]
                        L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                        _, idx = torch.min(L2.view(1,-1), -1)
                        width = right - left
                        point = [idx.item() // width + up, idx.item() % width + left]
                        points[j] = point

            res.points = [[point[0], point[1]] for point in points]

            # Motion supervision
            loss_motion = 0
            res.stop = True
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                thr = max(stop_thresh_px / 512 * h, stop_thresh_px)
                if torch.linalg.norm(direction) > thr:
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

            loss = loss_motion
            if mask is not None:
                if mask.min() == 0 and mask.max() == 1:
                    mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
                    loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
                    loss += lambda_mask * loss_fix

            loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
            if not res.stop:
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img
        res.w = ws.detach().cpu().numpy()

#----------------------------------------------------------------------------
