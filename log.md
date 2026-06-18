# SVM Kernel Trick 3D Interactive Demo — Session Log

## Project Summary

A complete educational demonstration for the Support Vector Machine kernel trick. Three phases: Manim 3D animation, sklearn RBF SVM visualization, and interactive Streamlit/Plotly dashboard.

**Live Demo:** https://svmdemo.streamlit.app/
**Repo:** https://github.com/huanchen1107/L13_SVM

---

## Files Created

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies: numpy, scikit-learn, matplotlib, streamlit, plotly, pandas, manim |
| `.gitignore` | Ignores `__pycache__/`, `*.pyc`, `outputs/*.png`, `.DS_Store` |
| `utils/data_generator.py` | `generate_ring_dataset()` — blue inner + red outer ring data with noise |
| `utils/svm_utils.py` | `train_svm()`, `make_decision_grid()`, `compute_decision_surface()` |
| `phase1_manim_kernel_trick.py` | Manim ThreeDScene — 2D→3D kernel trick concept animation |
| `phase1_matplotlib_animation.py` | Matplotlib fallback animation (no Manim required) — same concept |
| `phase2_rbf_decision_surface.py` | Real sklearn SVC(RBF) — 2D boundary + 3D surface plots |
| `phase3_streamlit_app.py` | Interactive Plotly dashboard with sidebar parameter controls |
| `README.md` | Project docs with screenshots, commands, teaching guide |

---

## Implementation Order

1. Created repo structure (`utils/`, `assets/`, `outputs/`)
2. Created `requirements.txt`
3. Implemented `utils/data_generator.py` — ring dataset generator (blue inner, red outer)
4. Implemented `utils/svm_utils.py` — SVM training, grid generation, decision surface computation
5. Implemented **Phase 1** (Manim animation) — `SVMKernelTrick3D` scene:
   - Title → 2D data on z=0 → formula φ(x,y)=(x,y,x²+y²) → lift points to 3D → paraboloid surface → separating hyperplane z=c → circle projection → camera rotation → summary
6. Implemented **Phase 2** (RBF SVM) — Trains `SVC(kernel='rbf', C=10, gamma=1)`:
   - 2D: scatter + decision boundary (f=0) + margin contours (f=±1) + support vectors
   - 3D: decision function surface + training points + support vectors
   - Verified: `acc=1.000`, `14 support vectors`, linear kernel `acc=0.562` (fails as expected)
7. Implemented **Phase 3** (Streamlit) — Interactive app:
   - Sidebar: kernel (linear/poly/rbf/sigmoid), C (0.1–100), gamma (0.01–10, conditional), degree (2–6, poly only), noise (0–0.5), n_points (40–300), random seed
   - 2D plot: Plotly contour + scatter + support vectors
   - 3D plot: Plotly Surface + Scatter3d + z=0 reference plane
   - Metrics: support vectors count, accuracy, kernel, C, gamma
   - Dynamic teaching notes based on gamma/C values
   - Caching via `@st.cache_data`
8. Created `README.md` with all sections: overview, educational story, phases, installation, run commands, screenshots, parameter guide, mathematical note, teaching suggestions, repo structure

---

## Dependencies & Issues

### Manim Installation (Resolved)
- **Issue:** `manim` depends on `moderngl` and `glcontext`, which need C++ compilation. No pre-built wheels for Python 3.14 on Windows.
- **Attempts:** Tried `pip install manim --only-binary :all:` (failed, no wheels), conda (not installed), winget (not installed)
- **Fallback:** Created `phase1_matplotlib_animation.py` using matplotlib FuncAnimation → GIF output
- **Resolution:** Downloaded and installed Microsoft Visual C++ Build Tools silently via `vs_buildtools.exe --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools`. Then `pip install manim` succeeded, and `moderngl`/`glcontext` compiled from source.

### Streamlit Deprecation
- `use_container_width` deprecated → replaced with `width="stretch"`

### Streamlit CORS Error
- User encountered `chrome-error://chromewebdata` — browser sandbox blocking localhost
- Tried ports 8559, 8501, 8502, 8777; enabled `--server.enableCORS false --server.enableXsrfProtection false`
- Likely an environment-specific browser sandbox issue

---

## Discussion Q&A

| Q | A |
|---|----|
| "What's your plan?" | Build repo structure → utils → Phase 1 Manim → Phase 2 sklearn → Phase 3 Streamlit → README → tests |
| "What shall I do?" | Option A: conda install manim. Option B: install C++ build tools then pip |
| "Do option A" | Conda not installed on system |
| "OK" | Proceeded with remaining setup |
| "Run Streamlit" | Launched at localhost:8559 |
| "The process stocks, restart" | Killed processes, restarted |
| "Chrome error" (unsafe load) | Tried CORS disable, different ports. Browser sandbox issue |
| "Deploy to streamlit.app" | Initialized git, committed, pushed to huanchen1107/L13_SVM.git |
| "Write README to summarize" | Rewrote README with tables, badges, parameter guide, repo structure |
| "Add Live Demo link" | Added https://svmdemo.streamlit.app/ badge and inline link |
| "Add workflow figure" | Added `assets/workflow.png` to README; image didn't show — un-gitignored assets/, committed |
| "Add live demo again at beginning" | Added bold link right after H1 title |
| "How about Phase 1?" | Wrote matplotlib fallback animation; later installed manim via VC++ build tools |
| "Show manim animation" | Installed build tools → pip install manim → ran `manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D` |

---

## Verification Results

- Phase 2: Runs successfully, `acc=1.000`, `14 SV`, plots saved to `outputs/`
- Phase 3: Syntax valid, all imports work, Streamlit launches
- Phase 1: Manim installed and rendering
- Linear kernel baseline: `acc=0.562` on circular data (confirms need for kernels)
- All Python files pass `ast.parse()` syntax check

---

## Commands

```bash
# Install
pip install -r requirements.txt

# Phase 1 — Manim
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D

# Phase 1 — Matplotlib fallback
python phase1_matplotlib_animation.py

# Phase 2
python phase2_rbf_decision_surface.py

# Phase 3
streamlit run phase3_streamlit_app.py
```
