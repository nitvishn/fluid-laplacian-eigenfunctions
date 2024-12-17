import os
import numpy as np
from PIL import Image
import torch
import math
from visualizer.render_utils import render_simulation

class ImageParticleVisualizer:
    def __init__(self, sim, num_particles, width=512, height=512, border_size=8, framerate=30, gamma=0.99, line_color=(255, 255, 255)):
        self.sim = sim
        self.num_particles = num_particles
        self.border_size = border_size
        self.width = width + 2 * border_size
        self.height = height + 2 * border_size

        self.framerate = framerate

        self.gamma = gamma
        self.line_color = np.array(line_color, dtype=np.float32)
        
        # Initialize particles randomly in domain [0, pi] x [0, pi]
        self.particles = torch.rand(num_particles, 2).to(sim.device) * math.pi
        # Divide by 2 in the y direction to remove particles in bottom half
        # self.particles[:, 1] = np.pi - self.particles[:, 1] / 2
        self.old_particles = self.particles.clone()

        # Initialize a blank image (black background)
        self.image = np.zeros((self.height, self.width, 3), dtype=np.float32)

        self.current_frame = 0
        
        # Ensure output directories exist
        self.output_dir = os.path.join('outputs', f'{self.sim.id}')
        os.makedirs(self.output_dir, exist_ok=True)
        frames_dir = os.path.join('outputs', f'{self.sim.id}', 'particle_frames')
        os.makedirs(frames_dir, exist_ok=True)

    def update(self, dt: float):
        velocity_field = self.sim.sample_velocity_field(self.particles)
        k1 = dt * velocity_field
        velocity_field = self.sim.sample_velocity_field(self.particles + 0.5 * k1)
        k2 = dt * velocity_field
        self.old_particles = self.particles.clone()
        self.particles += k2
        self.particles = torch.clamp(self.particles, 0, math.pi)

    def _world_to_image(self, coords):
        # coords: Nx2 tensor in [0, pi]
        # map [0, pi] to [0, width or height]
        x_img = (coords[:, 0] / math.pi) * (self.width - 2 * self.border_size) + self.border_size
        y_img = (coords[:, 1] / math.pi) * (self.height - 2 * self.border_size) + self.border_size
        return x_img, y_img

    def _draw_line(self, x0, y0, x1, y1):
        # Simple line drawing (Bresenham-like or float-based)
        # We'll just do a simple DDA for simplicity
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy))) + 1
        if steps == 0:
            return
        x_inc = dx / steps
        y_inc = dy / steps
        x = x0
        y = y0

        for i in range(steps):
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                # Blend line color onto existing image pixel
                self.image[iy, ix, :] = 0.5 * self.image[iy, ix, :] + 0.5 * self.line_color
            x += x_inc
            y += y_inc

    def _draw_border(self):
        # Draw a solid white border around the edges
        self.image[:self.border_size,:,:] = self.line_color
        self.image[-self.border_size:,:,:] = self.line_color
        self.image[:,:self.border_size,:] = self.line_color
        self.image[:,-self.border_size:,:] = self.line_color

    def visualize_frame(self):
        # Fade image
        self.image *= self.gamma

        self._draw_border()

        # Convert old and new positions to image coordinates
        old_x, old_y = self._world_to_image(self.old_particles)
        new_x, new_y = self._world_to_image(self.particles)

        # Draw lines for each particle
        for i in range(self.num_particles):
            self._draw_line(old_x[i].item(), old_y[i].item(), new_x[i].item(), new_y[i].item())

        # Save the image
        img = (np.clip(self.image, 0, 255)).astype(np.uint8)
        img = 255 - img # Invert colors

        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(self.output_dir, f'{self.sim.id}_current.png'))
        img_pil.save(os.path.join(f'outputs/{self.sim.id}/particle_frames', f'frame_{self.current_frame}.png'))

        self.current_frame += 1

    
    def render(self):
        render_simulation(self.sim.id, self.framerate, 'particle_frames', 'particles')