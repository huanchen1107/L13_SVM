from manim import *
import numpy as np
import sys
sys.path.insert(0, ".")
from utils.data_generator import generate_ring_dataset


class SVMKernelTrick3D(ThreeDScene):
    def construct(self):
        X, y = generate_ring_dataset(n_inner=35, n_outer=45, random_seed=7, noise=0.0)
        X_blue = X[y == 0]
        X_red = X[y == 1]
        z_blue = X_blue[:, 0] ** 2 + X_blue[:, 1] ** 2
        z_red = X_red[:, 0] ** 2 + X_red[:, 1] ** 2

        # Separation plane z = c, halfway between max blue and min red
        c_sep = float((np.max(z_blue) + np.min(z_red)) / 2)

        # --- 3D axes ---
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 8, 1],
            x_length=7,
            y_length=7,
            z_length=5,
        )
        z_label = axes.get_z_axis_label("z")

        # --- Step 1: Title --- # self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        title = Text("SVM Kernel Trick: From 2D to 3D", font_size=32)
        subtitle = Text("Nonlinear in 2D, linear in feature space", font_size=22, color=GRAY)
        subtitle.next_to(title, DOWN)
        title_group = VGroup(title, subtitle)
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(Write(title), Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title_group))

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        # --- Step 2: Show 2D data on z=0 plane --- # self.play(Create(axes), Write(z_label))
        self.play(Create(axes), Write(z_label))
        self.wait(0.5)

        # Create 2D dots at z=0
        dots_blue = VGroup()
        dots_red = VGroup()
        for x, y in X_blue:
            dots_blue.add(Dot3D(point=[x, y, 0], color=BLUE, radius=0.06))
        for x, y in X_red:
            dots_red.add(Dot3D(point=[x, y, 0], color=RED, radius=0.06))

        all_2d_dots = VGroup(dots_blue, dots_red)
        self.play(LaggedStart(
            *[FadeIn(dot, scale=0.5) for dot in all_2d_dots],
            lag_ratio=0.02,
        ), run_time=2)

        no_line = Text("No straight line can separate them in 2D.", font_size=24, color=YELLOW)
        no_line.to_corner(UL)
        self.add_fixed_in_frame_mobjects(no_line)
        self.play(Write(no_line))
        self.wait(1)
        self.play(FadeOut(no_line))

        # --- Step 3: Show mapping formula --- # formula = MathTex(
        formula = MathTex(
            r"\phi(x, y) = (x,\; y,\; x^2 + y^2)",
            font_size=36,
        )
        self.add_fixed_in_frame_mobjects(formula)
        self.play(Write(formula))
        self.wait(1.5)
        self.remove_fixed_in_frame_mobjects(formula)

        # --- Step 4: Animate lifting to 3D --- # self.play(formula.animate.to_corner(UR).scale(0.7))
        self.play(formula.animate.to_edge(UP).scale(0.65))
        self.wait(0.3)

        t_tracker = ValueTracker(0)

        def make_updater(x, y, z_target):
            return lambda d: d.move_to([x, y, t_tracker.get_value() * z_target])

        for dot, x, y, zt in zip(dots_blue, X_blue[:, 0], X_blue[:, 1], z_blue):
            dot.add_updater(make_updater(x, y, zt))
        for dot, x, y, zt in zip(dots_red, X_red[:, 0], X_red[:, 1], z_red):
            dot.add_updater(make_updater(x, y, zt))

        self.play(t_tracker.animate.set_value(1), run_time=3)
        self.wait(0.5)

        # Remove updaters so dots stay in place
        for dot in dots_blue:
            dot.clear_updaters()
        for dot in dots_red:
            dot.clear_updaters()

        # --- Step 5: Show paraboloid surface z = x^2 + y^2 --- # paraboloid = Surface(
        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u ** 2 + v ** 2),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(31, 31),
            fill_opacity=0.22,
            stroke_width=0.5,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        paraboloid.set_style(fill_opacity=0.22, stroke_width=0.5)
        self.play(FadeIn(paraboloid), run_time=1.5)
        self.wait(1)

        # --- Step 6: Show separating hyperplane z = c --- # hyperplane = Surface(
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, c_sep),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(2, 2),
            fill_opacity=0.35,
            fill_color=YELLOW,
            stroke_width=0,
        )
        self.play(FadeIn(hyperplane))

        hp_label = MathTex(r"z = c", font_size=30, color=YELLOW)
        hp_label.rotate(PI / 2, axis=RIGHT)
        hp_label.move_to(axes.c2p(2.5, 2.5, c_sep))
        self.add_fixed_orientation_mobjects(hp_label)
        self.play(Write(hp_label))
        self.wait(1)

        # --- Step 7: Project back to 2D --- # self.play(FadeOut(paraboloid), FadeOut(hyperplane))
        self.play(FadeOut(paraboloid), FadeOut(hyperplane), FadeOut(hp_label))

        proj_text = MathTex(
            r"z = c,\; z = x^2 + y^2 \;\Rightarrow\; x^2 + y^2 = c",
            font_size=30,
            color=YELLOW,
        )
        self.add_fixed_in_frame_mobjects(proj_text)
        self.play(Write(proj_text))
        self.wait(2)

        # Draw circle on z=0 plane
        circle_radius = np.sqrt(c_sep)
        circle_2d = Circle(
            radius=circle_radius,
            color=YELLOW,
            stroke_width=4,
        )
        circle_2d.move_to(axes.c2p(0, 0, 0))
        circle_2d.rotate(PI / 2, axis=RIGHT)
        self.play(Create(circle_2d), run_time=1.5)
        self.wait(1)

        self.play(FadeOut(proj_text))

        # --- Step 8: Camera rotation --- # self.begin_ambient_camera_rotation(rate=0.18)
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(6)
        self.stop_ambient_camera_rotation()

        # --- Step 9: Final summary --- # summary = VGroup(
        summary = VGroup(
            Text("In 3D: linear hyperplane", font_size=28, color=YELLOW),
            Text("In 2D: nonlinear decision boundary", font_size=28, color=YELLOW),
            Text("This is the intuition behind the kernel trick.", font_size=24, color=GRAY),
        )
        summary.arrange(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(summary)
        self.play(Write(summary), run_time=2)
        self.wait(3)
