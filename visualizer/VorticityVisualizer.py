import os
import numpy as np
from PIL import Image
import torch
import math
from visualizer.render_utils import render_simulation

class VorticityVisualizer:
    def __init__(self, sim, width=512, height=512, border_size=8, framerate=30):
        self.sim = sim
        self.border_size = border_size
        self.width = width + 2 * border_size
        self.height = height + 2 * border_size
        self.framerate = framerate

        # Track global min/max vorticity across all frames
        self.min_vort = float('inf')
        self.max_vort = -float('inf')

        self.current_frame = 0

        # Output directories
        self.output_dir = os.path.join('outputs', f'{self.sim.id}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Directory for raw vorticity values
        self.values_dir = os.path.join(self.output_dir, 'vorticity_values')
        os.makedirs(self.values_dir, exist_ok=True)

        # Directory for final images (to be created after all frames are captured)
        self.frames_dir = os.path.join(self.output_dir, 'vorticity_frames')
        os.makedirs(self.frames_dir, exist_ok=True)

        # Keep all vorticity frames in memory to avoid repeated np.load calls
        self.vorticity_frames = []

    def _draw_border(self, image):
        # Draw a solid white border around the edges of an image
        image[:self.border_size,:,:] = 255.0
        image[-self.border_size:,:,:] = 255.0
        image[:,:self.border_size,:] = 255.0
        image[:,-self.border_size:,:] = 255.0

    def update(self, dt):
        # No-op for this visualizer
        pass

    def visualize_frame(self):
        # Obtain the vorticity field from the simulation
        vorticity = self.sim.get_domain_vorticity_field()  # shape: [ny, nx]
        ny, nx = vorticity.shape

        # Update global min/max
        frame_min = vorticity.min().item()
        frame_max = vorticity.max().item()
        if frame_min < self.min_vort:
            self.min_vort = frame_min
        if frame_max > self.max_vort:
            self.max_vort = frame_max

        # Save the raw vorticity values for this frame and keep them in memory
        vorticity_np = vorticity.cpu().numpy()  # ensure CPU numpy
        np.save(os.path.join(self.values_dir, f'frame_{self.current_frame}.npy'), vorticity_np)
        self.vorticity_frames.append(vorticity_np)

        self.current_frame += 1

    def _save_vorticity_images(self):
        # After all frames have been captured, we know global min/max.
        # We now convert each vorticity array in memory into a consistent image scale.

        if not self.vorticity_frames:
            return  # No frames to process

        # Create coords only once
        vorticity_np = self.vorticity_frames[0]
        ny, nx = vorticity_np.shape
        x_coords = np.linspace(0, math.pi, nx)
        y_coords = np.linspace(0, math.pi, ny)

        for f_idx, vorticity_np in enumerate(self.vorticity_frames):
            print(f"writing frame={f_idx}/{len(self.vorticity_frames)}")
            # Normalize based on global min/max
            if self.max_vort == self.min_vort:
                vorticity_norm = np.zeros_like(vorticity_np, dtype=np.float32)
            else:
                vorticity_norm = (vorticity_np - self.min_vort) / (self.max_vort - self.min_vort)
            
            # Create an image buffer
            image = np.zeros((self.height, self.width, 3), dtype=np.float32)

            for j in range(ny):
                for i in range(nx):
                    x_img = int((x_coords[i] / math.pi) * (self.width - 2 * self.border_size) + self.border_size)
                    y_img = int((y_coords[j] / math.pi) * (self.height - 2 * self.border_size) + self.border_size)
                    val = vorticity_norm[j, i] * 255.0
                    if 0 <= x_img < self.width and 0 <= y_img < self.height:
                        image[y_img, x_img, :] = val

            self._draw_border(image)

            img = np.clip(image, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(self.frames_dir, f'frame_{f_idx}.png'))

    def render(self):
        # Create consistently scaled vorticity images
        self._save_vorticity_images()
        # Now render the simulation with these frames
        render_simulation(self.sim.id, self.framerate, 'vorticity_frames', 'vorticity')