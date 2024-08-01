import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil

from util import *
from camera import get_world2view_mat, get_proj_mat_J


class Rasterizer:
    def __init__(self) -> None:
        pass

    def splat(
        self,
        guassian_num,  # int, num of guassians
        basis_degree,  # int, degree of spherical harmonics
        basis_num,  # int, num of sh base function
        background,
        width,
        height,
        means3D,
        SpheHarmonics,
        opacities,
        scales,
        scale_modifier,
        rotations,
        viewmatrix,
        projmatrix,
        cam_pos,
        tan_fovx,
        tan_fovy,
    ) -> None:

        focal_y = height / (2 * tan_fovy)  # focal of y axis
        focal_x = width / (2 * tan_fovx)

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SpheHarmonics to RGB
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(
            guassian_num,
            basis_degree,
            basis_num,
            means3D,
            scales,
            scale_modifier,
            rotations,
            opacities,
            SpheHarmonics,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        )

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth
        depths = preprocessed["depths"]
        point_list = np.argsort(depths)

        # render
        logger.info("Starting render...")
        out_color = self.render(
            point_list,
            width,
            height,
            preprocessed["points_xy_image"],
            preprocessed["rgbs"],
            preprocessed["conic_opacity"],
            background,
        )
        return out_color

    def preprocess(
        self,
        guassian_num,
        basis_degree,
        basis_num,
        orig_points,
        scales,
        scale_modifier,
        rotations,
        opacities,
        SpheHarmonics,
        viewmatrix,
        projmatrix,
        cam_pos,
        W,
        H,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
    ):

        rgbs = []
        cov3Ds = []
        depths = []
        radii = []
        conic_opacity = []
        points_xy_image = []
        for idx in range(guassian_num):
            p_orig = orig_points[idx]
            p_view = in_cutoff(p_orig, viewmatrix)
            if p_view is None:
                continue
            depths.append(p_view[2])

            # transform point, from world to ndc
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix)
            p_w = 1 / (p_hom[3] + 0.0000001)
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]

            # compute covarianve
            scale = scales[idx]
            rotation = rotations[idx]
            cov3D = computeCov3D(scale, scale_modifier, rotation)
            cov3Ds.append(cov3D)

            cov = computeCov2D(
                p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
            )

            # invert covarianve(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1]
            if det == 0:
                depths.pop()
                cov3Ds.pop()
                continue
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])

            # compute radius
            mid = 0.5 * (cov[0] + cov[1])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)]

            radii.append(my_radius)
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            sh = SpheHarmonics[idx]
            result = computeColorFromSH(basis_degree, p_orig, cam_pos, sh)
            rgbs.append(result)

        return dict(
            rgbs=rgbs,
            cov3Ds=cov3Ds,
            depths=depths,
            radii=radii,
            conic_opacity=conic_opacity,
            points_xy_image=points_xy_image,
        )

    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, back_color
    ):

        out_color = np.zeros((H, W, 3))
        process_bar = tqdm(range(H * W))

        for i in range(H):
            for j in range(W):
                process_bar.update(1)
                cur_pixel = [i, j]
                C = [0, 0, 0]

                for idx in point_list:

                    transmirrance = 1

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]
                    d = [
                        xy[0] - cur_pixel[0],
                        xy[1] - cur_pixel[1],
                    ]
                    con_o = conic_opacity[idx]
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    if power > 0:
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    alpha = min(0.99, con_o[3] * np.exp(power))
                    if alpha < 1 / 255:
                        continue
                    T_trial = transmirrance * (1 - alpha)
                    if T_trial < 0.0001:
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * transmirrance

                    transmirrance = T_trial

                for ch in range(3):
                    out_color[j, i, ch] = C[ch] + transmirrance * back_color[ch]

        return out_color