import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

class RAFTTracker:
    """
    A standalone helper class for point tracking using RAFT (Optical Flow).
    This class is designed to assist DragGAN in tracking handle points between optimization steps.
    """
    def __init__(self, device='cuda'):
        """
        Initialize the RAFT model with pre-trained weights.
        
        Args:
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        
        # Load the pre-trained RAFT Large model
        # Raft_Large_Weights.DEFAULT provides the best available pre-trained weights
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model.eval()
        
        # Initialize transforms for RAFT (handles image normalization)
        self.transforms = self.weights.transforms()

    def preprocess_image(self, image_tensor):
        """
        Convert StyleGAN output tensor to the format expected by RAFT.
        
        Args:
            image_tensor (torch.Tensor): Tensor of shape [1, 3, H, W] in range [-1, 1].
            
        Returns:
            torch.Tensor: Tensor in range [0, 1] for RAFT transforms.
        """
        # Ensure image is on the correct device and detached
        image_tensor = image_tensor.to(self.device).detach()

        # RAFT expects images in range [0, 1] for float tensors, which it maps to [-1, 1] internally.
        # StyleGAN outputs are in range [-1, 1].
        # We must clamp to avoid artifacts.
        image_tensor = image_tensor.clamp(-1.0, 1.0)
        
        # Map [-1, 1] to [0, 1]
        processed_image = (image_tensor + 1.0) / 2.0
        return processed_image

    def update_points(self, image_prev, image_curr, points_prev):
        """
        Track points from previous image to current image using optical flow.
        
        Args:
            image_prev (torch.Tensor): Previous image tensor [1, 3, H, W] in [-1, 1].
            image_curr (torch.Tensor): Current image tensor [1, 3, H, W] in [-1, 1].
            points_prev (torch.Tensor or list): points to track, shape [N, 2] as [y, x].
            
        Returns:
            torch.Tensor: Updated points of shape [N, 2] as [y, x].
        """
        if not isinstance(points_prev, torch.Tensor):
            points_prev = torch.tensor(points_prev, device=self.device, dtype=torch.float32)
        
        if points_prev.shape[0] == 0:
            return points_prev

        # 1. Preprocess images
        img1 = self.preprocess_image(image_prev)
        img2 = self.preprocess_image(image_curr)

        # 2. Apply RAFT-specific transforms (normalization)
        # These transforms expect images in [0, 1] (float) or [0, 255] (byte)
        img1_proc, img2_proc = self.transforms(img1, img2)

        # 3. Compute Flow
        with torch.no_grad():
            # RAFT returns a list of flow predictions from its iterative refinement
            # The last element in the list is the most accurate
            predictions = self.model(img1_proc, img2_proc)
            flow = predictions[-1] # Shape: [1, 2, H, W]
            
            # DEBUG: Print flow statistics
            print(f"RAFT Flow Stats - Max: {flow.abs().max().item():.4f}, Mean: {flow.abs().mean().item():.4f}")

        # 4. Sample flow at point locations
        # flow[0, 0] is horizontal displacement (dx), flow[0, 1] is vertical (dy)
        _, _, H, W = flow.shape
        N = points_prev.shape[0]

        # Prepare grid for grid_sample (standardized coordinates in [-1, 1])
        # Note: grid_sample expects (x, y) order for coordinates.
        # DragGAN uses (y, x) order for points.
        points_y = points_prev[:, 0]
        points_x = points_prev[:, 1]

        # Scale coordinates to [-1, 1] for grid_sample
        grid_x = 2.0 * points_x / (W - 1) - 1.0
        grid_y = 2.0 * points_y / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, N, 1, 2)

        # Use bilinear interpolation to get precise flow values at point locations
        # sampled_flow shape: [1, 2, N, 1]
        sampled_flow = F.grid_sample(
            flow, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        # Reshape to [N, 2] where 2 components are (dx, dy)
        sampled_flow = sampled_flow.view(2, N).permute(1, 0)
        dx = sampled_flow[:, 0]
        dy = sampled_flow[:, 1]
        
        # DEBUG: Print sampled flow
        print(f"Sampled Flow (dx, dy): {sampled_flow.cpu().tolist()}")

        # 5. Update points
        points_curr = points_prev.clone()
        points_curr[:, 0] += dy # Update y
        points_curr[:, 1] += dx # Update x

        # Ensure points remain within image bounds
        points_curr[:, 0] = points_curr[:, 0].clamp(0, H - 1)
        points_curr[:, 1] = points_curr[:, 1].clamp(0, W - 1)

        return points_curr

if __name__ == "__main__":
    # Quick sanity check/test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker = RAFTTracker(device=device)
    print(f"RAFTTracker initialized on {device}")
    
    # Dummy tensors mimicking StyleGAN output [1, 3, 512, 512]
    img_prev = torch.zeros((1, 3, 512, 512), device=device)
    img_curr = torch.zeros((1, 3, 512, 512), device=device)
    
    # Dummy points [y, x]
    points = torch.tensor([[256.0, 256.0], [10.0, 10.0]], device=device)
    
    updated_points = tracker.update_points(img_prev, img_curr, points)
    print(f"Original points:\n{points}")
    print(f"Updated points:\n{updated_points}")
