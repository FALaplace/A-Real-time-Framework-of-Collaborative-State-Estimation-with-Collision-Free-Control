import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation_matrix_from_vectors(vec1, vec2):
    """计算从vec1旋转到vec2的旋转矩阵"""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):  # 180度翻转的特殊情况
        return -np.eye(3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


def plot_cone(ax, start, end, radius=0.1, resolution=30, color="green", alpha=0.8):
    """
    在ax上绘制从start指向end的圆锥
    """
    start, end = np.array(start), np.array(end)
    direction = end - start
    height = np.linalg.norm(direction)

    # 在z轴上生成标准圆锥
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, 2)  # 只要底面和顶点
    theta, z = np.meshgrid(theta, z)
    R = radius * (1 - z / height)

    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Z = z

    # 旋转对齐方向
    cone_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
    Rmat = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
    cone_points_rot = Rmat @ cone_points

    # 平移
    Xr = cone_points_rot[0].reshape(X.shape) + start[0]
    Yr = cone_points_rot[1].reshape(Y.shape) + start[1]
    Zr = cone_points_rot[2].reshape(Z.shape) + start[2]

    # 绘制
    ax.plot_surface(Xr, Yr, Zr, color=color, alpha=alpha)


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    start = [0, 0, 0]
    end = [1, 1, 2]
    plot_cone(ax, start, end, radius=0.2, color="orange")

    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(0, 3)
    plt.show()
