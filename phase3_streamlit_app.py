import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os
sys.path.insert(0, ".")
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface
from utils.animation_utils import generate_kernel_trick_gif


st.set_page_config(page_title="SVM Kernel Trick 3D Demo", layout="wide")
st.title("Interactive SVM Kernel Trick 3D Demo")

# Sidebar controls
st.sidebar.header("Parameters")
kernel = st.sidebar.selectbox(
    "Kernel", options=["linear", "poly", "rbf", "sigmoid"], index=2
)
C = st.sidebar.slider("C (regularization)", min_value=0.1, max_value=100.0,
                       value=10.0, step=0.1)

show_gamma = kernel in ("rbf", "poly", "sigmoid")
gamma = st.sidebar.slider("Gamma", min_value=0.01, max_value=10.0, value=1.0,
                           step=0.01, disabled=not show_gamma) if show_gamma else "scale"

show_degree = kernel == "poly"
degree = st.sidebar.slider("Degree", min_value=2, max_value=6, value=3, step=1,
                            disabled=not show_degree) if show_degree else 3

noise = st.sidebar.slider("Noise", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
n_points = st.sidebar.slider("Number of points", min_value=40, max_value=300,
                              value=120, step=10)
random_seed = st.sidebar.number_input("Random seed", value=7, step=1)

grid_res = min(150, max(40, n_points // 2))

# Generate data
@st.cache_data
def cached_generate(n_points, noise, random_seed):
    n_inner = int(n_points * 0.45)
    n_outer = n_points - n_inner
    return generate_ring_dataset(n_inner=n_inner, n_outer=n_outer,
                                  noise=noise, random_seed=random_seed)

X, y = cached_generate(n_points, noise, random_seed)

# Train model
gamma_val = gamma if show_gamma else "scale"
model = train_svm(X, y, kernel=kernel, C=C, gamma=gamma_val, degree=degree)

# Decision grid
margin = 0.5
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xx, yy, grid = make_decision_grid(
    x_range=(x_min, x_max), y_range=(y_min, y_max), resolution=grid_res
)

Z = None
has_decision_function = hasattr(model, "decision_function")
if has_decision_function:
    Z = compute_decision_surface(model, grid, xx.shape)
elif hasattr(model, "predict_proba"):
    Z_proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    Z = Z_proba - 0.5
else:
    Z = model.predict(grid).astype(float).reshape(xx.shape)
    Z = Z - 0.5

sv = model.support_vectors_ if hasattr(model, "support_vectors_") else np.empty((0, 2))
acc = model.score(X, y)

# Create tabs for different views
tab1, tab2 = st.tabs([
    "Interactive Parameters & Models", 
    "Concept Animation (Phase 1)",
])

with tab1:
    # Concept panel
    with st.expander("Concept", expanded=False):
        st.markdown("""
        - **2D circular data** cannot be separated by a straight line.
        - **Kernel methods** allow SVM to learn nonlinear decision boundaries.
        - **RBF kernel** uses similarity to support vectors to form a flexible boundary.
        """)

    # 2D plot
    st.subheader("2D Decision Boundary")
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(
        x=X[y == 0, 0], y=X[y == 0, 1], mode="markers",
        marker=dict(color="blue", size=6), name="Inner (class 0)"
    ))
    fig2d.add_trace(go.Scatter(
        x=X[y == 1, 0], y=X[y == 1, 1], mode="markers",
        marker=dict(color="red", size=6), name="Outer (class 1)"
    ))
    if len(sv) > 0:
        fig2d.add_trace(go.Scatter(
            x=sv[:, 0], y=sv[:, 1], mode="markers",
            marker=dict(color="black", size=10, symbol="circle-open", line=dict(width=2)),
            name="Support vectors"
        ))

    if has_decision_function:
        # Decision boundary f=0
        fig2d.add_trace(go.Contour(
            x=xx[0, :], y=yy[:, 0], z=Z,
            contours=dict(start=0, end=0, size=0.01, coloring="lines"),
            line=dict(color="yellow", width=2.5), name="f(x,y)=0",
            showscale=False,
        ))
        # Margin contours f=-1, f=+1
        fig2d.add_trace(go.Contour(
            x=xx[0, :], y=yy[:, 0], z=Z,
            contours=dict(start=-1, end=1, size=1, coloring="lines"),
            line=dict(color="gray", width=1, dash="dash"),
            name="margins (f=±1)", showscale=False,
        ))
    elif kernel == "linear":
        fig2d.add_trace(go.Contour(
            x=xx[0, :], y=yy[:, 0], z=Z,
            contours=dict(start=0, end=0, size=0.01, coloring="lines"),
            line=dict(color="yellow", width=2.5), name="boundary",
            showscale=False,
        ))

    fig2d.update_layout(
        xaxis_title="x", yaxis_title="y",
        width=600, height=550,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig2d, width="stretch")

    # 3D plot
    st.subheader("3D Decision Function Surface")
    fig3d = go.Figure()

    if has_decision_function:
        fig3d.add_trace(go.Surface(
            x=xx, y=yy, z=Z, colorscale="RdYlBu",
            opacity=0.8, name="f(x, y)", showscale=True,
            colorbar=dict(title="f(x, y)"),
        ))

    Z_train = model.decision_function(X) if has_decision_function else np.zeros(len(X))
    fig3d.add_trace(go.Scatter3d(
        x=X[y == 0, 0], y=X[y == 0, 1], z=Z_train[y == 0],
        mode="markers", marker=dict(color="blue", size=3),
        name="Inner"
    ))
    fig3d.add_trace(go.Scatter3d(
        x=X[y == 1, 0], y=X[y == 1, 1], z=Z_train[y == 1],
        mode="markers", marker=dict(color="red", size=3),
        name="Outer"
    ))
    if len(sv) > 0 and has_decision_function:
        Z_sv = model.decision_function(sv)
        fig3d.add_trace(go.Scatter3d(
            x=sv[:, 0], y=sv[:, 1], z=Z_sv,
            mode="markers",
            marker=dict(color="black", size=6, symbol="circle-open", line=dict(width=2)),
            name="Support vectors"
        ))
    if has_decision_function:
        fig3d.add_trace(go.Surface(
            x=xx, y=yy, z=np.zeros_like(xx),
            opacity=0.15, colorscale=[[0, "gray"], [1, "gray"]],
            showscale=False, name="z = 0",
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.8),
        ),
        width=700, height=600,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig3d, width="stretch")

    # Support vector metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Support vectors", len(sv))
    col2.metric("Training accuracy", f"{acc:.3f}")
    col3.metric("Kernel", kernel)
    col4.metric("C", f"{C:.2f}")
    col5.metric("Gamma", f"{gamma:.2f}" if show_gamma else "N/A")

    # Teaching notes
    st.subheader("Teaching Notes")
    notes = []
    if show_gamma and isinstance(gamma, (int, float)):
        if gamma < 0.2:
            notes.append("Gamma is small: the boundary is smoother and each point has wider influence.")
        if gamma > 3:
            notes.append("Gamma is large: the boundary becomes very flexible and may overfit.")
    if C < 1:
        notes.append("C is small: the model allows more mistakes to keep a wider margin.")
    if C > 20:
        notes.append("C is large: the model tries harder to classify training data correctly.")
    if kernel == "linear":
        notes.append("Linear kernel can only find a straight line. It will fail badly on circular data.")
    if notes:
        for note in notes:
            st.info(note)
    else:
        st.info("Try adjusting gamma, C, or kernel to see teaching tips appear here.")

with tab2:
    st.subheader("Phase 1: Conceptual 2D to 3D Feature Mapping")
    st.markdown("""
    In 2D space, the concentric blue and red dots are not linearly separable.
    By applying the feature mapping **phi(x, y) = (x, y, x^2 + y^2)**,
    each point is lifted into 3D based on its distance from the origin.
    In this 3D space, the classes become separable by a flat **separating hyperplane** (z = c).
    Projecting back to 2D gives the circular decision boundary (x^2 + y^2 = c).
    """)

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    anim_n_inner = col_a1.slider("Blue (inner) points", 10, 100, 35, 5, key="anim_n_inner")
    anim_n_outer = col_a2.slider("Red (outer) points", 10, 100, 45, 5, key="anim_n_outer")
    anim_noise = col_a3.slider("Noise", 0.0, 0.3, 0.0, 0.02, key="anim_noise")
    anim_seed = col_a4.number_input("Random seed", 0, 999, 7, key="anim_seed")

    if st.button("Regenerate Animation", type="primary"):
        with st.spinner("Generating animation... (may take ~30s)"):
            out = generate_kernel_trick_gif(
                n_inner=anim_n_inner, n_outer=anim_n_outer,
                noise=anim_noise, random_seed=anim_seed,
            )
        st.success(f"Saved: {out}")
        st.rerun()

    manim_video = "media/videos/phase1_manim_kernel_trick/480p15/SVMKernelTrick3D.mp4"
    fallback_gif = "outputs/phase1_kernel_trick.gif"

    if os.path.exists(manim_video):
        st.video(manim_video)
        st.caption("Phase 1 Concept Animation (Manim Community Edition)")
    elif os.path.exists(fallback_gif):
        st.image(fallback_gif)
        st.caption("Phase 1 Concept Animation (Matplotlib)")
    else:
        st.warning("Animation not found. Click 'Regenerate' above.")
