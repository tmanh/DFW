# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn as nn


# Displacement
# TODO: distance, slope, elevation diff
class InverseDistance(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, fmts):
        super().__init__()

    def forward(self, xs, x, valid, inputs, train, stage):
        weights = (1 / (torch.abs(xs[:, :, 1:2]) + torch.abs(xs[:, :, 2:3]))) * valid
        alpha = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * alpha, dim=1, keepdim=True)

    def freeze(self):
        pass

    def unfreeze(self):
        pass


class EllipticalInverseDistance(nn.Module):
    def __init__(self, in_dim, n_layers, n_dim, fmts, angle_deg=0.0, major_range=1.0, minor_range=1.0, power=2):
        """
        angle_deg: orientation of ellipse (0 degrees is north, clockwise)
        major_range: influence distance along major axis
        minor_range: influence distance along minor axis
        power: IDW exponent
        """
        super().__init__()
        
        self.angle_rad = torch.deg2rad(torch.tensor(angle_deg))
        self.major_range = major_range
        self.minor_range = minor_range
        self.power = power

    def elliptical_distance(self, dx, dy):
        """Compute elliptical distances accounting for anisotropy."""
        cos_a = torch.cos(self.angle_rad)
        sin_a = torch.sin(self.angle_rad)

        # Rotate coordinates
        dx_rot = dx * cos_a + dy * sin_a
        dy_rot = -dx * sin_a + dy * cos_a

        # Elliptical scaling
        dist = torch.sqrt((dx_rot / self.major_range)**2 + (dy_rot / self.minor_range)**2)

        return dist

    def forward(self, xs, x, inputs, train, stage):
        """
        xs: [batch_size, num_points, 3] where xs[..., 0]=distance, xs[..., 1]=dx, xs[..., 2]=dy
        x: [batch_size, num_points, 1] values at known locations
        """
        dx = xs[..., 1]
        dy = xs[..., 2]

        # Compute elliptical distances
        dist = self.elliptical_distance(dx, dy)  # shape: [batch_size, num_points]

        # Avoid division by zero
        dist = torch.clamp(dist, min=1e-10)

        weights = 1 / dist**self.power
        alpha = weights / torch.sum(weights, dim=1, keepdim=True)

        # Perform weighted interpolation
        interpolated = torch.sum(x * alpha.unsqueeze(-1), dim=1, keepdim=True)
        return interpolated.unsqueeze(-1)

    def freeze(self):
        pass

    def unfreeze(self):
        pass
