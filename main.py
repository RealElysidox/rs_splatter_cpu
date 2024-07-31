import math
from util import *
from renderer import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # set guassian
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]])
    n = len(pts)
    SpheHarmonics = np.random.random((n, 16, 3))
    # print(SpheHarmonics)
    opacities = np.ones((n, 1))
    scales = np.ones((n, 3))
    rotations = np.array([np.eye(3)] * n)

    # set camera
    cam_pos = np.array([0, 0, 5])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    viewmatrix = get_world2view_mat(R=R, t=cam_pos)
    projmatrix = get_proj_mat_J(**proj_param)
    projmatrix = np.dot(projmatrix, viewmatrix)
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # render
    rasterizer = Rasterizer()
    out_color = rasterizer.splat(
        P=len(pts),
        D=3,
        M=16,
        background=np.array([0, 0, 0]),
        width=700,
        height=700,
        means3D=pts,
        SpheHarmonics=SpheHarmonics,
        opacities=opacities,
        scales=scales,
        scale_modifier=1,
        rotations=rotations,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        cam_pos=cam_pos,
        tan_fovx=tanfovx,
        tan_fovy=tanfovy,
    )

    plt.imshow(out_color)
    plt.show()
