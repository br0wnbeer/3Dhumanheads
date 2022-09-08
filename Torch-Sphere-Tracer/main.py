import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse

import renderers as renderers
import signed_distance_functions as sdf
import constructive_solid_geometry as csg


def main(args):

    device = torch.device("cuda")

    num_iterations = 2000
    convergence_threshold = 1e-3

    # ---------------- camera matrix ---------------- #

    fx = fy = 1024
    cx = cy = 512
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)

    # ---------------- camera position ---------------- #

    distance = 5.0
    azimuth = np.pi / 4.0
    elevation = np.pi / 4.0

    camera_position = torch.tensor([
        +np.cos(elevation) * np.sin(azimuth), 
        -np.sin(elevation), 
        -np.cos(elevation) * np.cos(azimuth)
    ], device=device) * distance

    # ---------------- camera rotation ---------------- #

    target_position = torch.tensor([0.0, -1.0, 0.0], dtype = torch.float, device=device)
    up_direction = torch.tensor([0.0, 1.0, 0.0], dtype = torch.float, device=device)

    camera_z_axis = (target_position - camera_position).float()
    camera_x_axis = torch.cross(up_direction, camera_z_axis, dim=-1)
    camera_y_axis = torch.cross(camera_z_axis, camera_x_axis, dim=-1)
    camera_rotation = torch.stack((camera_x_axis, camera_y_axis, camera_z_axis), dim=-1)
    camera_rotation = nn.functional.normalize(camera_rotation, dim=-2)

    # ---------------- directional light ---------------- #

    light_directions = torch.tensor([1.0, -0.5, 0.0], device=device)

    # ---------------- ray marching ---------------- #
    
    y_positions = torch.arange(cy * 2, dtype=camera_matrix.dtype, device=device)
    x_positions = torch.arange(cx * 2, dtype=camera_matrix.dtype, device=device)
    y_positions, x_positions = torch.meshgrid(y_positions, x_positions)
    z_positions = torch.ones_like(y_positions)
    ray_positions = torch.stack((x_positions, y_positions, z_positions), dim=-1)
    ray_positions = torch.einsum("mn,...n->...m", torch.inverse(camera_matrix),  ray_positions)
    ray_positions = torch.einsum("mn,...n->...m", camera_rotation, ray_positions) + camera_position
    ray_directions = nn.functional.normalize(ray_positions - camera_position, dim=-1)

    # ---------------- rendering ---------------- #

    def compute_rotation_matrix(axes, angles):
        nx, ny, nz = torch.unbind(axes, dim=-1)
        c, s = torch.cos(angles), torch.sin(angles)
        rotation_matrices = torch.stack([
            torch.stack([nx * nx * (1.0 - c) + 1. * c, ny * nx * (1.0 - c) - nz * s, nz * nx * (1.0 - c) + ny * s], dim=-1),
            torch.stack([nx * ny * (1.0 - c) + nz * s, ny * ny * (1.0 - c) + 1. * c, nz * ny * (1.0 - c) - nx * s], dim=-1),
            torch.stack([nx * nz * (1.0 - c) - ny * s, ny * nz * (1.0 - c) + nx * s, nz * nz * (1.0 - c) + 1. * c], dim=-1),
        ], dim=-2)
        return rotation_matrices

    signed_distance_functions = [
        sdf.translation(sdf.sphere(1.0) , torch.tensor([0.0, -1.0, 0.0], device=device)),
    ]
    # Bachground Plane 
    ground = sdf.plane(torch.tensor([0.0, -1.0, 0.0], device=device), 0.0) 

    def generator():

        for signed_distance_function in signed_distance_functions:
            # Ground and SDF are combinded 
            signed_distance_function = csg.union(signed_distance_function, ground)

            surface_positions, converged ,color  = renderers.sphere_tracing(
                signed_distance_function=signed_distance_function, 
                ray_positions=ray_positions, 
                ray_directions=ray_directions, 
                num_iterations=num_iterations, 
                convergence_threshold=convergence_threshold,
            )
            print(surface_positions.shape)
            print(converged.shape)
            print(color.shape)
            
            surface_positions = torch.where(converged, surface_positions, torch.zeros_like(surface_positions))

            surface_normals = renderers.compute_normal(
                signed_distance_function=signed_distance_function, 
                surface_positions=surface_positions,
            )
            # Surface normals are Saved for the Object only
            surface_normals = torch.where(converged, surface_normals, torch.zeros_like(surface_normals)) 
            image = torch.where(converged, color, torch.zeros_like(color)*255)
            # Thong Shading not unsed so far for image calculation 
            """image = renderers.phong_shading(
                surface_normals=surface_normals, 
                view_directions=camera_position - surface_positions, 
                light_directions=light_directions, 
                light_ambient_color=torch.ones(1, 1, 3, device=device),
                light_diffuse_color=torch.ones(1, 1, 3, device=device), 
                light_specular_color=torch.ones(1, 1, 3, device=device), 
                material_ambient_color=torch.full((1, 1, 3), 0.2, device=device) + (torch.rand(1, 1, 3, device=device) * 2 - 1) * 0.1,
                material_diffuse_color=torch.full((1, 1, 3), 0.7, device=device) + (torch.rand(1, 1, 3, device=device) * 2 - 1) * 0.1,
                material_specular_color=torch.full((1, 1, 3), 0.1, device=device),
                material_emission_color=torch.zeros(1, 1, 3, device=device),
                material_shininess=64.0,
            )"""
           
            grounded = torch.abs(ground(surface_positions)) < convergence_threshold # Convereged rays at object 
            image = torch.where(grounded, torch.full_like(image, 0.9), image)
            # Shadow comutation depenedent on light source 
            shadowed = renderers.compute_shadows(
                signed_distance_function=signed_distance_function, 
                surface_positions=surface_positions, 
                surface_normals=surface_normals,
                light_directions=light_directions, 
                num_iterations=num_iterations, 
                convergence_threshold=convergence_threshold,
                foreground_masks=converged,
            )
            #image = torch.where(shadowed, image * 0.5, image)
            #print(image.shape)
            image = torch.where(converged, image, torch.zeros_like(image))
            #print(image.shape)
            yield image

    images = image
    torchvision.utils.save_image(images, "csg.png", nrow=3)


if __name__ == "__main__":
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description="Sphere Tracing")
    parser.add_argument("--width", type=str, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
