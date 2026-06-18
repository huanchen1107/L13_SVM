import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.insert(0, ".")
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface


def main():
    X, y = generate_ring_dataset(n_inner=35, n_outer=45, random_seed=7, noise=0.08)
    model = train_svm(X, y, kernel="rbf", C=10.0, gamma=1.0)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy, grid = make_decision_grid(
        x_range=(x_min, x_max), y_range=(y_min, y_max), resolution=100
    )
    Z = compute_decision_surface(model, grid, xx.shape)
    sv = model.support_vectors_

    # --- 2D plot ---
    fig2d, ax2d = plt.subplots(figsize=(7, 6))
    ax2d.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=30, label="Inner (blue)")
    ax2d.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=30, label="Outer (red)")
    ax2d.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none",
                 edgecolors="black", linewidths=1.5, label="Support vectors")
    ax2d.contour(xx, yy, Z, levels=[0], colors="yellow", linewidths=2)
    ax2d.contour(xx, yy, Z, levels=[-1, 1], colors="gray",
                 linestyles="dashed", linewidths=1)
    ax2d.set_title("2D Decision Boundary (RBF kernel, C=10, gamma=1)")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.legend(loc="upper right")
    ax2d.set_aspect("equal")
    fig2d.tight_layout()
    fig2d.savefig("outputs/phase2_2d_decision_boundary.png", dpi=150)
    plt.close(fig2d)

    # --- 3D decision function surface ---
    fig3d = plt.figure(figsize=(10, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(xx, yy, Z, cmap=cm.RdYlBu, alpha=0.7, linewidth=0,
                             antialiased=True)
    ax3d.contour(xx, yy, Z, levels=[0], colors="yellow", linewidths=2, zdir="z", offset=Z.min())

    # Scatter training points at z = decision_function value
    Z_train = model.decision_function(X)
    ax3d.scatter(X[y == 0, 0], X[y == 0, 1], Z_train[y == 0],
                 c="blue", s=20, depthshade=True, label="Inner")
    ax3d.scatter(X[y == 1, 0], X[y == 1, 1], Z_train[y == 1],
                 c="red", s=20, depthshade=True, label="Outer")
    Z_sv = model.decision_function(sv)
    ax3d.scatter(sv[:, 0], sv[:, 1], Z_sv,
                 s=80, facecolors="none", edgecolors="black",
                 linewidths=1.5, depthshade=False, label="Support vectors")

    ax3d.set_title("3D Decision Function Surface f(x, y)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("f(x, y)")
    ax3d.legend(loc="upper left")
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label="f(x, y)")
    fig3d.tight_layout()
    fig3d.savefig("outputs/phase2_3d_decision_surface.png", dpi=150)
    plt.close(fig3d)

    print("Phase 2 complete. Plots saved to outputs/")
    print(f"  Support vectors: {len(sv)}")
    print(f"  Training accuracy: {model.score(X, y):.3f}")


if __name__ == "__main__":
    main()
