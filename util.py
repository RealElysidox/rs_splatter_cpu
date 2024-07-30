import numpy as np
from camera import getProjectionMatrix


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def computeColorFromSH(deg, pos, campos, sh):
    # The implementation is loosely based on code for
    # "Differentiable Point-Based Radiance Fields for
    # Efficient View Synthesis" by Zhang et al. (2022)

    dir = pos - campos
    dir = dir / np.linalg.norm(dir)

    result = SH_C0 * sh[0]

    if deg > 0:
        x, y, z = dir
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]

        if deg > 1:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (
                result
                + SH_C2[0] * xy * sh[4]
                + SH_C2[1] * yz * sh[5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                + SH_C2[3] * xz * sh[7]
                + SH_C2[4] * (xx - yy) * sh[8]
            )

            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh[9]
                    + SH_C3[1] * xy * z * sh[10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                    + SH_C3[5] * z * (xx - yy) * sh[14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
                )
    result += 0.5
    return np.clip(result, a_min=0, a_max=1)

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def in_frustum(p_orig, viewmatrix):
    # bring point to screen space
    p_view = transformPoint4x3(p_orig, viewmatrix)

    if p_view[2] <= 0.2:
        return None
    return p_view


def transformPoint4x4(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
            matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
        ]
    )
    return transformed


def transformPoint4x3(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        ]
    )
    return transformed


# covariance = RS[S^T][R^T]
def computeCov3D(scale, mod, rot):
    # create scaling matrix
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # normalize quaternion to get valid rotation
    # we use rotation matrix
    R = rot

    # compute 3d world covariance matrix Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D


def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.

    t = transformPoint4x3(mean, viewmatrix)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    J = np.array(
        [
            [focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])],
            [0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
            [0, 0, 0],
        ]
    )
    W = viewmatrix[:3, :3]
    T = np.dot(J, W)

    cov = np.dot(T, cov3D)
    cov = np.dot(cov, T.T)

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high
    # Discard 3rd row and column
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    return [cov[0, 0], cov[0, 1], cov[1, 1]]

if __name__ == "__main__":
    deg = 3
    pos = np.array([2, 0, -2])
    campos = np.array([0, 0, 5])
    sh = np.random.random((16, 3))
    computeColorFromSH(deg, pos, campos, sh)