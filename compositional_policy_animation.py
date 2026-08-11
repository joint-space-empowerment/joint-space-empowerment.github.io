import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

cmap_type = "bone"

FONT_AXIS_LABEL = 18
FONT_PANEL_TITLE = 20
FONT_ANNOTATION = 18
FONT_AXIS_TITLE = 18
FONT_PANEL_LABEL = 28

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# Reduced grid for animation performance
N = 60
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(x, y)

# Derive base plane from seed=42 (matches the static figure)
np.random.seed(42)
A_BASE = np.random.uniform(-0.4, 0.4)
B_BASE = np.random.uniform(-0.4, 0.4)

# Uncertainty sample planes
Z_SAMPLES = []
for i in range(3):
    np.random.seed(10 + i)
    a = np.random.uniform(-0.4, 0.4)
    b = np.random.uniform(-0.4, 0.4)
    Z_SAMPLES.append(a * X + b * Y)

_light = np.array([5.0, -5.0, 5.0])

def compute_shading(Z, alpha=1.0):
    dist_sq = (X - _light[0])**2 + (Y - _light[1])**2 + (Z - _light[2])**2
    bright = 1.0 / (1.0 + 0.015 * dist_sq)
    bright = (bright - bright.min()) / (bright.max() - bright.min())
    bright = 0.5 + 0.5 * bright
    grey = np.zeros((*bright.shape, 4))
    grey[..., :3] = bright[..., np.newaxis]
    grey[..., 3] = 0.8 * alpha
    return grey

def get_distribution(sigma_x, sigma_y):
    Prob = np.exp(-((X**2 / (2 * sigma_x**2)) + (Y**2 / (2 * sigma_y**2))))
    return Prob / np.max(Prob)

def smooth_step(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + (b - a) * t

def add_axes_arrows(ax, length=2.5, z_offset=0, alpha=1.0):
    lw = 2.5
    o = [0, 0, z_offset]
    kw = dict(arrow_length_ratio=0.0, linewidth=lw, alpha=0.8 * alpha)
    ax.quiver(*o, 0, 0, length * 1.4, color="black", **kw)
    ax.text(o[0], o[1], o[2] + length * 1.4 + 0.4, "muscle $M$",
            fontsize=FONT_AXIS_LABEL, ha="center", alpha=alpha)
    ax.quiver(*o, length, 0, 0, color="black", **kw)
    ax.text(o[0] + length + 0.4, o[1], o[2] - 0.4, "muscle 2",
            fontsize=FONT_AXIS_LABEL, ha="center", alpha=alpha)
    ax.quiver(*o, 0, -length, 0, color="black", **kw)
    ax.text(o[0], o[1] - length - 0.4, o[2] - 0.4, "muscle 1",
            fontsize=FONT_AXIS_LABEL, ha="center", alpha=alpha)

def draw_left(ax, Z, alpha=1.0):
    ax.cla()
    ax.axis("off")
    for Z_s in Z_SAMPLES:
        ax.plot_surface(X, Y, Z_s, color="lightblue", alpha=0.15 * alpha,
                        rstride=3, cstride=3, linewidth=0.5, shade=False, zorder=1)
    grey = compute_shading(Z, alpha=alpha)
    ax.plot_surface(X, Y, Z, facecolors=grey, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=2)
    add_axes_arrows(ax, length=3.3, z_offset=-0.85, alpha=alpha)
    ax.set_title(r"$\bf{manifold}$" + "\n(task-agnostic)",
                 fontsize=FONT_PANEL_TITLE, y=0.95, linespacing=1.4, alpha=alpha)
    ax.text2D(0.23, 0.6, "posterior\nsamples", transform=ax.transAxes,
              ha="center", fontsize=FONT_ANNOTATION, color="steelblue", alpha=alpha)
    ax.view_init(elev=25, azim=-45)
    ax.set_zlim(-1.5, 4.4)
    ax.set_box_aspect([1, 1, 1], zoom=1.1)

def draw_middle(ax, Prob, alpha=1.0):
    ax.cla()
    if alpha <= 0:
        ax.axis("off")
        return
    ax.imshow(Prob, extent=[-2, 2, -2, 2], origin="lower", cmap=cmap_type,
              interpolation="bicubic", aspect="equal", alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("latent action 1", fontsize=FONT_AXIS_TITLE, labelpad=10, alpha=alpha)
    ax.set_ylabel("latent action 2", fontsize=FONT_AXIS_TITLE, labelpad=10, alpha=alpha)
    ax.set_title(r"$\bf{base\ distribution}$" + "\n(task-specific)",
                 fontsize=FONT_PANEL_TITLE, y=1.14, linespacing=1.4, pad=0, alpha=alpha)

def draw_right(ax, Z, colors, alpha=1.0):
    ax.cla()
    ax.axis("off")
    if alpha <= 0:
        ax.set_zlim(-1.5, 4.4)
        ax.set_box_aspect([1, 1, 1], zoom=1.1)
        return
    for Z_s in Z_SAMPLES:
        ax.plot_surface(X, Y, Z_s, color="grey", alpha=0.15 * alpha,
                        rstride=3, cstride=3, linewidth=0, shade=False, zorder=1)
    c = colors.copy()
    c[..., 3] *= alpha
    ax.plot_surface(X, Y, Z, facecolors=c, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=2)
    add_axes_arrows(ax, length=3.3, z_offset=-0.85, alpha=alpha)
    ax.set_title("manifold policy\n", fontsize=FONT_PANEL_TITLE,
                 fontweight="bold", y=0.962, alpha=alpha)
    ax.view_init(elev=25, azim=-45)
    ax.set_zlim(-1.5, 4.4)
    ax.set_box_aspect([1, 1, 1], zoom=1.1)


# -------------------------------------------------------
# Phase boundaries (frames)
# -------------------------------------------------------
FPS = 20
REVEAL_DUR = 30   # frames to fade in each panel
HOLD1_DUR  = 30   # hold after all panels visible
DIST_DUR   = 120  # distribution varies
PLANE_DUR  = 120  # plane orientation varies
HOLD2_DUR  = 40   # final hold

F_MID_START   = REVEAL_DUR                       # 30
F_RIGHT_START = REVEAL_DUR * 2                   # 60
F_HOLD1_END   = REVEAL_DUR * 3 + HOLD1_DUR      # 120
F_DIST_END    = F_HOLD1_END + DIST_DUR           # 240
F_PLANE_END   = F_DIST_END + PLANE_DUR           # 360
TOTAL_FRAMES  = F_PLANE_END + HOLD2_DUR          # 400


def get_alphas(frame):
    alpha1 = 1.0
    alpha2 = smooth_step((frame - F_MID_START) / REVEAL_DUR)
    alpha3 = smooth_step((frame - F_RIGHT_START) / REVEAL_DUR)
    return alpha1, alpha2, alpha3


def get_params(frame):
    sigma_x, sigma_y = 0.5, 1.0
    a, b = A_BASE, B_BASE

    if F_HOLD1_END <= frame < F_DIST_END:
        t = (frame - F_HOLD1_END) / DIST_DUR
        # Two full oscillations of sigma_x, one-and-a-half of sigma_y
        sigma_x = 0.5 + 0.45 * np.sin(2 * np.pi * t * 2)
        sigma_y = 1.0 + 0.5 * np.sin(2 * np.pi * t * 1.5)
        sigma_x = max(0.18, sigma_x)
        sigma_y = max(0.18, sigma_y)

    if frame >= F_DIST_END:
        # Keep last distribution values from end of dist phase
        t_dist = 1.0
        sigma_x = 0.5 + 0.45 * np.sin(2 * np.pi * t_dist * 2)
        sigma_y = 1.0 + 0.5 * np.sin(2 * np.pi * t_dist * 1.5)
        sigma_x = max(0.18, sigma_x)
        sigma_y = max(0.18, sigma_y)

    if F_DIST_END <= frame < F_PLANE_END:
        t = (frame - F_DIST_END) / PLANE_DUR
        # Sweep to a steep plane and back
        a_tgt, b_tgt = 0.38, -0.36
        if t < 0.5:
            s = smooth_step(t * 2)
            a = lerp(A_BASE, a_tgt, s)
            b = lerp(B_BASE, b_tgt, s)
        else:
            s = smooth_step((t - 0.5) * 2)
            a = lerp(a_tgt, A_BASE, s)
            b = lerp(b_tgt, B_BASE, s)

    Z = a * X + b * Y
    return sigma_x, sigma_y, Z


def main():
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.1, 0.5, 0.1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_axes([0.29, 0.205, 0.4, 0.52])
    ax3 = fig.add_subplot(gs[4], projection="3d")

    # Fixed panel labels
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    label_y = pos1.y1 + 0.03
    la = fig.text(pos1.x0 + 0.005, label_y - 0.07, "a",
                  fontsize=FONT_PANEL_LABEL, fontweight="bold", ha="left", va="bottom")
    lb = fig.text(pos2.x0 - 0.025, label_y - 0.07, "b",
                  fontsize=FONT_PANEL_LABEL, fontweight="bold", ha="left", va="bottom")

    def update(frame):
        alpha1, alpha2, alpha3 = get_alphas(frame)
        sigma_x, sigma_y, Z = get_params(frame)

        Prob = get_distribution(sigma_x, sigma_y)
        norm = plt.Normalize(Prob.min(), Prob.max())
        colors = plt.colormaps[cmap_type](norm(Prob))

        draw_left(ax1, Z, alpha=alpha1)
        draw_middle(ax2, Prob, alpha=alpha2)
        draw_right(ax3, Z, colors, alpha=alpha3)

        la.set_alpha(alpha1)
        lb.set_alpha(max(alpha2, alpha3))
        return []

    ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                  interval=1000 // FPS, blit=False)

    print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps "
          f"({TOTAL_FRAMES / FPS:.1f}s)...")

    try:
        out = "compositional_policy_animation.mp4"
        ani.save(out, writer="ffmpeg", fps=FPS, dpi=100,
                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        print(f"Saved: {out}")
    except Exception as e:
        print(f"ffmpeg failed ({e}), falling back to GIF...")
        out = "compositional_policy_animation.gif"
        ani.save(out, writer="pillow", fps=FPS)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()