# SVM Kernel Trick 3D Interactive Demo

## Project Overview

A complete educational demonstration for the Support Vector Machine kernel trick. The project includes a Manim animation showing the conceptual 2D-to-3D feature mapping, a real sklearn RBF SVM decision surface visualisation, and an interactive Streamlit/Plotly web app for students to explore kernel parameters.

**Target audience:** high school CS students, introductory ML students, and teachers for classroom presentation.

## Educational Story

1. "2D inner blue points and outer red points cannot be separated by a straight line."
2. "Through feature mapping, the data is lifted to 3D."
3. "In the 3D feature space, a hyperplane can separate the classes."
4. "The 3D hyperplane projects back to 2D as a nonlinear (circular) decision boundary."
5. "A real RBF SVM decision function surface is then visualised to show how the kernel works in practice."
6. "Finally, an interactive interface lets students adjust C, gamma, and kernel to build intuition."

## Phase 1: Manim Kernel Trick Animation

Animated explanation of how z = x^2 + y^2 makes circularly separable data linearly separable in 3D.

- Shows 2D data on the z = 0 plane
- Animates lifting points to 3D via φ(x, y) = (x, y, x^2 + y^2)
- Displays the paraboloid surface and a separating horizontal hyperplane
- Projects the circle x^2 + y^2 = c back to the 2D plane
- Camera rotation for spatial understanding

## Phase 2: Real RBF SVM Decision Surface

Trains a real sklearn SVC with RBF kernel and visualises:

- 2D scatter plot with decision boundary (f = 0) and margin contours (f = ±1)
- 3D decision function surface f(x, y) with training points placed at their decision function values
- Support vectors highlighted in both plots

## Phase 3: Interactive Streamlit Demo

Web app with sidebar controls for:

- **Kernel:** linear, poly, rbf, sigmoid
- **C:** 0.1 to 100 (regularisation strength)
- **Gamma:** 0.01 to 10 (RBF/poly/sigmoid kernel width)
- **Degree:** 2 to 6 (polynomial kernel degree)
- **Noise:** 0 to 0.5 (dataset noise)
- **Number of points:** 40 to 300
- **Random seed**

Displays interactive 2D decision boundary, 3D decision function surface, support vector metrics, and dynamic teaching notes.

## Installation

```bash
pip install -r requirements.txt
```

> **Windows note:** `manim` depends on `moderngl` and `glcontext`, which require Visual C++ Build Tools to compile. If `pip install manim` fails, either:
> - Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and retry, or
> - Use conda: `conda install -c conda-forge manim`

## Run Commands

### Phase 1 — Manim animation

Low-quality preview:
```bash
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
```

High-quality render:
```bash
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
```

### Phase 2 — RBF SVM decision surface

```bash
python phase2_rbf_decision_surface.py
```

Output images are saved to `outputs/`.

### Phase 3 — Interactive Streamlit app

```bash
streamlit run phase3_streamlit_app.py
```

## Important Mathematical Note

The mapping z = x^2 + y^2 in Phase 1 is a visual and educational feature mapping used to explain why nonlinear data can become linearly separable in a higher-dimensional feature space. A real RBF kernel does not explicitly map data to only 3D; it corresponds to a high-dimensional or infinite-dimensional feature space. Therefore, the RBF decision surface shown in Phase 2 and Phase 3 visualises the decision function f(x, y), not the full feature space itself.

## Teaching Suggestions

- Start with the Manim animation to build conceptual intuition about why higher dimensions help.
- Use Phase 3 to let students experiment: try gamma = 0.1 vs gamma = 5, C = 0.1 vs C = 50.
- Switch to the linear kernel to show it fails on circular data — reinforcing the need for kernels.
- Discuss how gamma controls the "reach" of each support vector and how C trades off margin width vs training errors.
