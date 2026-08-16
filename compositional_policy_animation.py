import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

cmap_type = "bone"

FONT_AXIS_LABEL = 18
FONT_PANEL_TITLE = 20
FONT_AXIS_TITLE = 18
FONT_PANEL_LABEL = 28

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYMBOLS_PDF = os.path.join(SCRIPT_DIR, "static", "images",
                            "manifold_policy_bone_symbols_FINAL.pdf")

# Reduced grid for animation performance
N = 60
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(x, y)

def _sample_orientation(seed):
    np.random.seed(seed)
    a = np.random.uniform(-0.4, 0.4)
    b = np.random.uniform(-0.4, 0.4)
    return a, b

# Derive base plane from seed=42 (matches the static figure)
A_BASE, B_BASE = _sample_orientation(42)

# Alternative "posterior samples" of the manifold's orientation -- the
# single displayed plane periodically swaps to one of these (see
# get_plane_schedule) instead of continuously rotating.
ORIENTATIONS = [(A_BASE, B_BASE), _sample_orientation(10), _sample_orientation(11)]

_light = np.array([5.0, -5.0, 5.0])

def compute_shading(Z, alpha=1.0):
    dist_sq = (X - _light[0])**2 + (Y - _light[1])**2 + (Z - _light[2])**2
    bright = 1.0 / (1.0 + 0.015 * dist_sq)
    bright = (bright - bright.min()) / (bright.max() - bright.min())
    bright = 0.5 + 0.5 * bright
    grey = np.zeros((*bright.shape, 4))
    grey[..., :3] = bright[..., np.newaxis]
    grey[..., 3] = alpha
    return grey

def get_distribution(sigma_x, sigma_y):
    Prob = np.exp(-((X**2 / (2 * sigma_x**2)) + (Y**2 / (2 * sigma_y**2))))
    return Prob / np.max(Prob)

def smooth_step(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

# -------------------------------------------------------
# Extract the 4 hand-set LaTeX expressions straight out of the final
# composited PDF (static/images/manifold_policy_bone_symbols_FINAL.pdf)
# as transparent-background crops, instead of re-typesetting them with
# matplotlib's usetex. Each is rasterized at high DPI, then converted to
# RGBA by treating luminance as inverse alpha (black text on a white
# page -> black text with anti-aliased alpha), so it composites cleanly
# over the animation frames.
#
# Pixel boxes below were measured by hand against a 150-dpi render of
# that PDF (final_fig.png cropped by pdfcrop); DX/DY/CANVAS_* calibrate
# that crop back to the *uncropped* figure's coordinate system, which is
# the same coordinate system the animation figure itself uses (both are
# savefig'd from a figsize=(20, 6) figure).
# -------------------------------------------------------
_CAL_DPI = 150
_CAL_DX, _CAL_DY = 392, 84
_CAL_CANVAS_W, _CAL_CANVAS_H = 3000, 900  # uncropped (20, 6) figure @ 150 dpi

# name: (x0, x1, y0, y1, padL, padR, padT, padB), all in final_fig.png px @ 150 dpi
_LATEX_REGIONS = {
    "M":   (247, 435, 717, 769, 8, 8, 8, 8),
    "piz": (979, 1175, 717, 769, 8, 8, 8, 8),
    "pia": (1871, 2071, 717, 769, 8, 8, 8, 8),
    # tight negative right/bottom pad: the mapping arrow starts just
    # below-right of this label and bleeds in otherwise
    "g":   (1397, 1619, 336, 384, 8, -4, 8, 0),
}


def extract_latex_overlays(pdf_path, fig_w_in, fig_h_in, out_dpi, extract_dpi=600):
    with tempfile.TemporaryDirectory() as tmpdir:
        hires_png = os.path.join(tmpdir, "hires.png")
        subprocess.run(
            ["gs", "-sDEVICE=png16m", f"-r{extract_dpi}", "-o", hires_png, pdf_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        hires = np.asarray(Image.open(hires_png).convert("RGB"), dtype=np.float32)

    scale = extract_dpi / _CAL_DPI
    overlays = {}
    for name, (x0, x1, y0, y1, padL, padR, padT, padB) in _LATEX_REGIONS.items():
        X0, X1 = int((x0 - padL) * scale), int((x1 + padR) * scale)
        Y0, Y1 = int((y0 - padT) * scale), int((y1 + padB) * scale)
        crop = hires[Y0:Y1, X0:X1]
        alpha = np.clip(255 - crop.mean(axis=2), 0, 255).astype(np.uint8)
        rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        img = Image.fromarray(rgba, "RGBA")

        cx150 = ((x0 - padL) + (x1 + padR)) / 2
        cy150 = ((y0 - padT) + (y1 + padB)) / 2
        w150 = (x1 + padR) - (x0 - padL)
        h150 = (y1 + padB) - (y0 - padT)
        fx = (cx150 + _CAL_DX) / _CAL_CANVAS_W
        fy = 1 - (cy150 + _CAL_DY) / _CAL_CANVAS_H

        target_w = max(1, round(w150 / _CAL_DPI * out_dpi))
        target_h = max(1, round(h150 / _CAL_DPI * out_dpi))
        img = img.resize((target_w, target_h), Image.LANCZOS)

        px = fx * fig_w_in * out_dpi
        py = (1 - fy) * fig_h_in * out_dpi
        overlays[name] = (img, (round(px - target_w / 2), round(py - target_h / 2)))

    return overlays


def composite_overlays(frame_rgba, overlays, alphas):
    """Alpha-composite the extracted LaTeX crops onto a rendered frame.

    frame_rgba: HxWx4 uint8 array from fig.canvas.buffer_rgba().
    alphas: {name: fade_alpha in [0, 1]}.
    """
    img = Image.fromarray(frame_rgba, "RGBA")
    for name, (overlay, xy) in overlays.items():
        fade = alphas[name]
        if fade <= 0:
            continue
        if fade >= 1:
            layer = overlay
        else:
            r, g, b, a = overlay.split()
            a = a.point(lambda p, f=fade: round(p * f))
            layer = Image.merge("RGBA", (r, g, b, a))
        img.alpha_composite(layer, dest=xy)
    return img

# mplot3d autoscales X/Y from whatever's plotted that frame. The surface
# is sometimes skipped entirely (mid plane-swap, alpha 0), which used to
# shrink/shift the autoscaled box relative to frames where it's drawn --
# the whole panel (axes, labels, box_aspect zoom) would visibly jump.
# Pinning explicit, constant limits every frame removes that dependency.
# The value matches what autoscale always produced anyway: the quiver
# arms don't register in mplot3d's autoscale at all (only their own
# 3.3-unit-long dummy bbox does), so the box was always sized from just
# the surface's [-2, 2] grid plus its default 10% margin.
PLOT_XY_LIM = 2.2

# Same as add_central_axes() in compositional_policy_figure_FINAL.py: plain
# quiver/text calls with no explicit zorder, so mplot3d's own automatic
# depth sort (computed_zorder stays at its default, True) decides whether
# each arm renders in front of or behind the surface. That's exactly what
# gives the correct-looking result in the static figure, computed fresh
# from that frame's real geometry -- no custom occlusion math needed here
# either, now that each held orientation is a static frame (the plane
# fades out/in around every swap, so there's never a frame where the sort
# has to stay consistent with a *different* orientation's).
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

def draw_left(ax, Z, alpha=1.0, plane_alpha=1.0):
    ax.cla()
    ax.axis("off")
    surf_alpha = alpha * plane_alpha
    if surf_alpha > 0:
        grey = compute_shading(Z, alpha=surf_alpha)
        ax.plot_surface(X, Y, Z, facecolors=grey, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False,
                        zorder=2)
    add_axes_arrows(ax, length=3.3, z_offset=-0.85, alpha=alpha)
    ax.set_title(r"$\bf{manifold}$" + "\n(task-agnostic)",
                 fontsize=FONT_PANEL_TITLE, y=0.95, linespacing=1.4, alpha=alpha)
    ax.view_init(elev=25, azim=-45)
    ax.set_xlim(-PLOT_XY_LIM, PLOT_XY_LIM)
    ax.set_ylim(-PLOT_XY_LIM, PLOT_XY_LIM)
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

def draw_right(ax, Z, colors, alpha=1.0, plane_alpha=1.0):
    ax.cla()
    ax.axis("off")
    if alpha <= 0:
        ax.set_xlim(-PLOT_XY_LIM, PLOT_XY_LIM)
        ax.set_ylim(-PLOT_XY_LIM, PLOT_XY_LIM)
        ax.set_zlim(-1.5, 4.4)
        ax.set_box_aspect([1, 1, 1], zoom=1.1)
        return
    surf_alpha = alpha * plane_alpha
    if surf_alpha > 0:
        c = colors.copy()
        c[..., 3] *= surf_alpha
        ax.plot_surface(X, Y, Z, facecolors=c, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False,
                        zorder=2)
    add_axes_arrows(ax, length=3.3, z_offset=-0.85, alpha=alpha)
    ax.set_title("manifold policy\n", fontsize=FONT_PANEL_TITLE,
                 fontweight="bold", y=0.962, alpha=alpha)
    ax.view_init(elev=25, azim=-45)
    ax.set_xlim(-PLOT_XY_LIM, PLOT_XY_LIM)
    ax.set_ylim(-PLOT_XY_LIM, PLOT_XY_LIM)
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

    a, b, plane_alpha = get_plane_schedule(frame, default=(a, b))

    Z = a * X + b * Y
    return sigma_x, sigma_y, Z, a, b, plane_alpha


# -------------------------------------------------------
# Instead of continuously rotating the manifold plane, hold it at one
# orientation, then periodically swap to a different "posterior sample"
# orientation: fade the plane out, swap, then fade it back in.
# -------------------------------------------------------
PLANE_HOLD_DUR  = 30  # frames holding each orientation (first/last halved)
PLANE_TRANS_DUR = 10  # frames per swap (fade-out + fade-in)

# BASE -> O1 -> O2 -> BASE, i.e. 3 swaps between the 4 legs below
_PLANE_LEGS = [ORIENTATIONS[0], ORIENTATIONS[1], ORIENTATIONS[2], ORIENTATIONS[0]]


def _plane_transition(t):
    """t in [0, 1) across one swap window -> plane_alpha (dips to 0 at t=0.5)."""
    if t < 0.5:
        return 1 - smooth_step(t * 2)
    return smooth_step((t - 0.5) * 2)


def get_plane_schedule(frame, default):
    if not (F_DIST_END <= frame < F_PLANE_END):
        return (*default, 1.0)

    l = frame - F_DIST_END
    half_hold = PLANE_HOLD_DUR / 2
    leg = half_hold  # end of the first (halved) hold
    for i in range(len(_PLANE_LEGS) - 1):
        if l < leg:
            orient = _PLANE_LEGS[i]
            return (*orient, 1.0)
        trans_end = leg + PLANE_TRANS_DUR
        if l < trans_end:
            t = (l - leg) / PLANE_TRANS_DUR
            dip = _plane_transition(t)
            orient = _PLANE_LEGS[i] if t < 0.5 else _PLANE_LEGS[i + 1]
            return (*orient, dip)
        is_last = i == len(_PLANE_LEGS) - 2
        hold = half_hold if is_last else PLANE_HOLD_DUR
        leg = trans_end + hold

    return (*_PLANE_LEGS[-1], 1.0)


# Content only ever occupies roughly x:[260, 1799] / y:[62, 575] of the
# 2000x600 canvas (measured across every phase of the animation) -- the
# rest is unused margin from the default matplotlib layout. Cropped here
# (with a little padding) rather than by touching the figure's margins,
# since all the panel/label/overlay positions are calibrated in absolute
# figure-fraction coordinates and would need re-deriving otherwise.
CROP_X0, CROP_Y0, CROP_W, CROP_H = 245, 47, 1570, 544


def main():
    FIG_W, FIG_H, DPI = 20, 6, 100
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
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

    # 'b' used to be anchored to ax2's *bounding box* left edge, but the
    # imshow panel (aspect='equal') is centered inside that box and is
    # narrower than it -- so the label sat too far left. Anchor it to the
    # actual visible left edge of the square panel instead.
    square_w = pos2.height * (FIG_H / FIG_W)
    square_left = pos2.x0 + (pos2.width - square_w) / 2
    lb = fig.text(square_left - 0.025, label_y - 0.07, "b",
                  fontsize=FONT_PANEL_LABEL, fontweight="bold", ha="left", va="bottom")

    # The mapping arrow between the two 'b' panels is plain graphics (not
    # LaTeX), so it's still drawn directly; the 4 math captions are
    # composited in per-frame from extract_latex_overlays() below.
    arrow_g = patches.FancyArrowPatch((0.587, 0.4633), (0.6747, 0.4633),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=20,
                                       linewidth=2.5, color="black")
    fig.add_artist(arrow_g)

    print("Extracting LaTeX captions from", SYMBOLS_PDF)
    overlays = extract_latex_overlays(SYMBOLS_PDF, FIG_W, FIG_H, out_dpi=DPI)

    print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps "
          f"({TOTAL_FRAMES / FPS:.1f}s)...")

    out = "compositional_policy_animation.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{FIG_W * DPI}x{FIG_H * DPI}", "-pix_fmt", "rgb24",
         "-r", str(FPS), "-i", "-", "-an",
         "-vf", f"crop={CROP_W}:{CROP_H}:{CROP_X0}:{CROP_Y0}",
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    for frame in range(TOTAL_FRAMES):
        alpha1, alpha2, alpha3 = get_alphas(frame)
        sigma_x, sigma_y, Z, a, b, plane_alpha = get_params(frame)

        Prob = get_distribution(sigma_x, sigma_y)
        norm = plt.Normalize(Prob.min(), Prob.max())
        colors = plt.colormaps[cmap_type](norm(Prob))

        draw_left(ax1, Z, alpha=alpha1, plane_alpha=plane_alpha)
        draw_middle(ax2, Prob, alpha=alpha2)
        draw_right(ax3, Z, colors, alpha=alpha3, plane_alpha=plane_alpha)

        la.set_alpha(alpha1)
        lb.set_alpha(max(alpha2, alpha3))
        arrow_g.set_alpha(alpha3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        composited = composite_overlays(
            buf, overlays,
            {"M": alpha1, "piz": alpha2, "pia": alpha3, "g": alpha3},
        )
        ffmpeg.stdin.write(composited.convert("RGB").tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()