import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.data_generator import generate_ring_dataset


def generate_kernel_trick_gif(n_inner=35, n_outer=45, noise=0.0, random_seed=7,
                              output_path="outputs/phase1_kernel_trick.gif",
                              total_frames=180, fps=10):
    """Generate the kernel trick concept animation GIF with given parameters."""

    X, y = generate_ring_dataset(n_inner=n_inner, n_outer=n_outer,
                                  noise=noise, random_seed=random_seed)
    X_blue = X[y == 0]
    X_red = X[y == 1]
    z_blue = X_blue[:, 0] ** 2 + X_blue[:, 1] ** 2
    z_red = X_red[:, 0] ** 2 + X_red[:, 1] ** 2
    c_sep = float((np.max(z_blue) + np.min(z_red)) / 2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    blue_scatter = ax.scatter([], [], [], c="blue", s=30, label="Inner (blue)")
    red_scatter = ax.scatter([], [], [], c="red", s=30, label="Outer (red)")

    u = np.linspace(-3, 3, 25)
    v = np.linspace(-3, 3, 25)
    U, V = np.meshgrid(u, v)
    Z_surf = U ** 2 + V ** 2
    surf = ax.plot_surface(U, V, Z_surf, alpha=0, color="cyan")

    pX, pY = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(-3, 3, 2))
    pZ = np.full_like(pX, c_sep)
    plane = ax.plot_surface(pX, pY, pZ, alpha=0, color="yellow")

    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.sqrt(c_sep) * np.cos(theta)
    circle_y = np.sqrt(c_sep) * np.sin(theta)
    circle_z = np.zeros_like(theta)
    circle_line, = ax.plot([], [], [], color="yellow", linewidth=2.5)

    title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, ha="center",
                            fontsize=14, fontweight="bold", color="white",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    ax.legend(loc="upper left")

    breaks = [0, 0.10, 0.18, 0.28, 0.45, 0.58, 0.72, 0.85, 1.0]
    labels = [
        "2D data on z=0 plane",
        "No straight line can separate them.",
        "Feature mapping: z = x^2 + y^2",
        "Lifting points to 3D...",
        "Paraboloid surface",
        "Separating hyperplane z = c",
        "Projection: x^2 + y^2 = c",
        "Camera rotation",
        "In 3D: linear | In 2D: nonlinear",
    ]

    interval = 1000 / fps

    def update(frame):
        t = frame / total_frames
        idx = 0
        for i, b in enumerate(breaks):
            if t >= b:
                idx = i
        title_text.set_text(labels[idx])

        if t < breaks[2]:
            lift = 0
        elif t < breaks[3] + 0.05:
            lift = min(1.0, (t - breaks[2]) / (breaks[3] - breaks[2]))
        else:
            lift = 1.0

        blue_z = lift * z_blue
        red_z = lift * z_red
        blue_scatter._offsets3d = (X_blue[:, 0], X_blue[:, 1], blue_z)
        red_scatter._offsets3d = (X_red[:, 0], X_red[:, 1], red_z)

        alpha_pts = min(1.0, t / breaks[1])
        blue_scatter.set_alpha(alpha_pts)
        red_scatter.set_alpha(alpha_pts)

        surf_alpha = 0
        if t >= breaks[3] + 0.02 and t < breaks[5] + 0.05:
            prog = min(1.0, (t - breaks[3]) / (breaks[4] - breaks[3]))
            surf_alpha = prog * 0.22
        elif t >= breaks[5] + 0.05:
            surf_alpha = 0.22
        surf.set_alpha(surf_alpha)

        plane_alpha = 0
        if t >= breaks[4] + 0.03 and t < breaks[6]:
            prog = min(1.0, (t - breaks[4]) / (breaks[5] - breaks[4]))
            plane_alpha = prog * 0.35
        elif t >= breaks[6]:
            plane_alpha = 0.35
        plane.set_alpha(plane_alpha)

        if t >= breaks[5] + 0.03:
            circle_line.set_data(circle_x, circle_y)
            circle_line.set_3d_properties(circle_z)
        else:
            circle_line.set_data([], [])
            circle_line.set_3d_properties([])

        if t < breaks[1]:
            ax.view_init(elev=90, azim=-90)
        elif t < breaks[6]:
            ax.view_init(elev=25, azim=-45)
        elif t < breaks[7]:
            ax.view_init(elev=90, azim=-90)
        elif t < breaks[8]:
            prog = (t - breaks[7]) / (breaks[8] - breaks[7])
            ax.view_init(elev=25, azim=-45 + prog * 360)
        else:
            ax.view_init(elev=25, azim=-45)

        return blue_scatter, red_scatter, surf, plane, circle_line, title_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval,
                         blit=False)
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path
