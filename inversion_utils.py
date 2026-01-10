# Copyright (c) 2024. GAN Inversion utilities for DragGAN.
# This module provides functionality to invert real images into GAN latent space.

import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

try:
    from lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available, will use L2 loss only")


class InversionModule:
    """Module for performing GAN inversion on real images using optimization."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32 if self.device.type == 'mps' else torch.float64
        
    def preprocess_image(self, image, target_size=(512, 512), model_type='general'):
        """
        Preprocess image for inversion.
        
        Args:
            image: PIL Image or numpy array
            target_size: tuple (height, width)
            model_type: 'general', 'ffhq', 'stylegan_human'
            
        Returns:
            torch.Tensor: preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image
        if model_type == 'stylegan_human':
            # StyleGAN-Human uses 1024x512 (height x width)
            target_size = (1024, 512)
        elif model_type == 'ffhq':
            # FFHQ uses square 1024x1024
            target_size = (1024, 1024)
        else:
            # Most other models use square images
            # Detect size from model if possible
            pass
        
        # Resize and convert to tensor
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(image)
        return img_tensor
    
    def optimize_invert(self, image, G, num_steps=500, initial_lr=0.01, 
                       w_avg_samples=10000, progress_callback=None):
        """
        Perform GAN inversion using optimization.
        
        Args:
            image: PIL Image or torch.Tensor
            G: Generator network
            num_steps: number of optimization steps
            initial_lr: initial learning rate
            w_avg_samples: number of samples for computing W average
            progress_callback: optional callback function(step, loss)
            
        Returns:
            torch.Tensor: inverted latent code in W+ space
        """
        print(f'Starting optimization inversion with {num_steps} steps...')
        
        # Preprocess image
        if isinstance(image, Image.Image):
            target_h = G.img_resolution
            target_w = G.img_resolution
            
            transform = transforms.Compose([
                transforms.Resize((target_h, target_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            img_tensor = transform(image).unsqueeze(0).to(self.device)
        else:
            img_tensor = image
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.min() >= 0:
                img_tensor = img_tensor * 2 - 1
            img_tensor = img_tensor.to(self.device)
        
        # Make a copy of G for inversion
        G_copy = copy.deepcopy(G).eval().requires_grad_(False).to(self.device)
        
        # Compute W mean
        print(f'Computing W mean from {w_avg_samples} samples...')
        z_samples = torch.randn([w_avg_samples, G_copy.z_dim], device=self.device)
        with torch.no_grad():
            w_samples = G_copy.mapping(z_samples, None)[:, :1, :]
            w_avg = w_samples.mean(dim=0, keepdim=True)
            w_std = w_samples.std(dim=0, keepdim=True)
        
        # Initialize latent code close to mean
        w = w_avg.detach().clone()
        w = w.repeat(1, G_copy.num_ws, 1)
        w.requires_grad = True
        
        # Initialize optimizer with learning rate scheduling
        optimizer = torch.optim.Adam([w], lr=initial_lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        
        # Initialize LPIPS if available
        lpips_fn = None
        if LPIPS_AVAILABLE:
            try:
                lpips_fn = LPIPS(net='alex').to(self.device).eval()
                for param in lpips_fn.parameters():
                    param.requires_grad = False
            except:
                print("LPIPS initialization failed, using MSE only")
        
        # Optimization loop
        print(f'Optimizing latent code...')
        best_loss = float('inf')
        best_w = w.detach().clone()
        
        for step in range(num_steps):
            # Generate image
            synth_img = G_copy.synthesis(w, noise_mode='const')
            
            # Compute losses
            mse_loss = F.mse_loss(synth_img, img_tensor)
            
            # LPIPS perceptual loss
            if lpips_fn is not None:
                lpips_loss = lpips_fn(synth_img, img_tensor).mean()
                # Balanced combination
                total_loss = mse_loss * 1.0 + lpips_loss * 1.0
            else:
                total_loss = mse_loss
                lpips_loss = torch.tensor(0.0)
            
            # Optional: L2 regularization to stay close to W mean
            if step < num_steps // 2:  # Only in first half
                w_reg = ((w - w_avg.repeat(1, G_copy.num_ws, 1)) ** 2).mean() * 0.01
                total_loss = total_loss + w_reg
            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_w = w.detach().clone()
            
            # Progress update every 10 steps
            if step % 10 == 0:
                print(f'Step {step}/{num_steps}: MSE={mse_loss.item():.4f}, '
                      f'LPIPS={lpips_loss.item():.4f}, Total={total_loss.item():.4f}')
            
            # Callback for Gradio progress (less frequent to avoid timeout)
            if progress_callback is not None and step % 20 == 0:
                try:
                    progress_callback(step, total_loss.item())
                except:
                    pass  # Ignore progress callback errors
        
        print(f'Optimization completed! Best loss: {best_loss:.4f}')
        return best_w
    
    def pti_invert(self, image, G, num_pti_steps=350, initial_inversion_steps=450, 
                   pti_lr=5e-4, initial_lr=8e-3, progress_callback=None):
        """
        Perform PTI (Pivotal Tuning Inversion) - fine-tune generator while optimizing latent.
        
        Args:
            image: PIL Image or torch.Tensor
            G: Generator network (will be fine-tuned)
            num_pti_steps: number of PTI fine-tuning steps
            initial_inversion_steps: steps for initial w inversion
            pti_lr: learning rate for generator fine-tuning
            initial_lr: learning rate for initial inversion
            progress_callback: optional callback function(step, loss)
            
        Returns:
            tuple: (w_latent, fine_tuned_G)
        """
        print(f'Starting PTI inversion...')
        
        # Step 1: Initial W inversion (same as optimize_invert)
        print(f'Step 1: Initial W inversion with {initial_inversion_steps} steps...')
        w_pivot = self.optimize_invert(image, G, num_steps=initial_inversion_steps, 
                                       initial_lr=initial_lr, progress_callback=progress_callback)
        
        # Preprocess image for loss computation
        if isinstance(image, Image.Image):
            target_h = G.img_resolution
            target_w = G.img_resolution
            
            transform = transforms.Compose([
                transforms.Resize((target_h, target_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            target_img = transform(image).unsqueeze(0).to(self.device)
        else:
            target_img = image
            if target_img.dim() == 3:
                target_img = target_img.unsqueeze(0)
            if target_img.min() >= 0:
                target_img = target_img * 2 - 1
            target_img = target_img.to(self.device)
        
        # Step 2: PTI - fine-tune generator while keeping w fixed
        print(f'Step 2: PTI fine-tuning with {num_pti_steps} steps...')
        
        # Make a copy of G for PTI
        G_pti = copy.deepcopy(G).to(self.device)
        G_pti.requires_grad_(True)
        
        # Original G for locality regularization
        G_original = copy.deepcopy(G).eval().requires_grad_(False).to(self.device)
        
        # Optimizer for generator parameters
        optimizer = torch.optim.Adam(G_pti.parameters(), lr=pti_lr, betas=(0.9, 0.999))
        
        # Initialize LPIPS
        lpips_fn = None
        if LPIPS_AVAILABLE:
            try:
                lpips_fn = LPIPS(net='alex').to(self.device).eval()
                for param in lpips_fn.parameters():
                    param.requires_grad = False
            except:
                print("LPIPS initialization failed for PTI")
        
        best_loss = float('inf')
        best_G_state = copy.deepcopy(G_pti.state_dict())
        
        for step in range(num_pti_steps):
            # Forward pass with fine-tuned generator
            synth_img = G_pti.synthesis(w_pivot, noise_mode='const')
            
            # Reconstruction loss
            pt_l2_loss = F.mse_loss(synth_img, target_img)
            total_loss = pt_l2_loss
            
            # LPIPS loss
            if lpips_fn is not None:
                pt_lpips_loss = lpips_fn(synth_img, target_img).mean()
                total_loss = total_loss + pt_lpips_loss
            
            # Locality regularization: penalize large changes to generator
            # Sample random w codes and check that original G and tuned G still produce similar images
            if step % 1 == 0:  # Locality regularization every step
                z_samples = torch.randn([5, G_original.z_dim], device=self.device)
                with torch.no_grad():
                    w_samples = G_original.mapping(z_samples, None)
                
                for w_sample in w_samples:
                    w_sample_expanded = w_sample.unsqueeze(0)
                    
                    with torch.no_grad():
                        original_img = G_original.synthesis(w_sample_expanded, noise_mode='const')
                    
                    tuned_img = G_pti.synthesis(w_sample_expanded, noise_mode='const')
                    
                    # L2 regularization
                    reg_l2 = F.mse_loss(original_img, tuned_img)
                    total_loss = total_loss + reg_l2 * 0.1
                    
                    # LPIPS regularization
                    if lpips_fn is not None:
                        reg_lpips = lpips_fn(original_img, tuned_img).mean()
                        total_loss = total_loss + reg_lpips * 0.1
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_G_state = copy.deepcopy(G_pti.state_dict())
            
            if step % 10 == 0:
                print(f'PTI Step {step}/{num_pti_steps}: Loss={total_loss.item():.4f}')
            
            if progress_callback is not None and step % 50 == 0:
                try:
                    progress_callback(initial_inversion_steps + step, total_loss.item())
                except:
                    pass
        
        # Load best generator state
        G_pti.load_state_dict(best_G_state)
        G_pti.eval()
        
        print(f'PTI completed! Best loss: {best_loss:.4f}')
        return w_pivot, G_pti


def create_inversion_module(device='cuda'):
    """Factory function to create InversionModule."""
    return InversionModule(device=device)


# Utility functions for integration with Gradio
def invert_and_reconstruct(image, G, device='cuda', num_steps=500):
    """
    Invert image and reconstruct it with the generator.
    
    Args:
        image: PIL Image
        G: Generator network
        device: computation device
        num_steps: number of optimization steps
        
    Returns:
        tuple: (w_latent, reconstructed_image)
    """
    inv_module = InversionModule(device=device)
    
    # Perform optimization inversion
    w_latent = inv_module.optimize_invert(image, G, num_steps=num_steps)
    
    # Reconstruct image
    with torch.no_grad():
        label = torch.zeros([1, G.c_dim], device=device)
        reconstructed = G.synthesis(w_latent, noise_mode='const')
        
        # Convert to PIL Image
        reconstructed = (reconstructed.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        reconstructed = reconstructed[0].cpu().numpy()
        reconstructed_img = Image.fromarray(reconstructed, 'RGB')
    
    return w_latent, reconstructed_img
