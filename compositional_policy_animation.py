import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Sans, matching the webpage's title font, used only for the
# pure-text title cards rendered by render_title_frame() -- the
# scientific figure panels above keep the serif/Times styling set
# globally above.
GOOGLE_SANS_REGULAR = os.path.join(SCRIPT_DIR, "static", "fonts", "GoogleSans-Regular.ttf")
GOOGLE_SANS_BOLD = os.path.join(SCRIPT_DIR, "static", "fonts", "GoogleSans-Bold.ttf")
fm.fontManager.addfont(GOOGLE_SANS_REGULAR)
fm.fontManager.addfont(GOOGLE_SANS_BOLD)
TITLE_FONT = fm.FontProperties(fname=GOOGLE_SANS_REGULAR)
TITLE_FONT_BOLD = fm.FontProperties(fname=GOOGLE_SANS_BOLD)
SYMBOLS_PDF = os.path.join(SCRIPT_DIR, "static", "images",
                            "manifold_policy_bone_symbols_FINAL.pdf")

# Reduced grid for animation performance. mplot3d's plot_surface does
# real per-facet work in Python (depth handling, color assembly) that
# scales with N^2 and dominates render time far more than pixel count
# does -- lowering DPI alone barely moves total runtime, so --preview
# drops this too (see set_grid_resolution()).
N = 60
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(x, y)


def set_grid_resolution(n):
    """Rebind the module-level grid globals. Every function that plots
    the surface reads X/Y as globals at call time (not at def time), so
    calling this before main() changes what they all draw with."""
    global N, x, y, X, Y
    N = n
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

def get_distribution(sigma_x, sigma_y, mu_x=0.0, mu_y=0.0):
    Prob = np.exp(-(((X - mu_x)**2 / (2 * sigma_x**2)) + ((Y - mu_y)**2 / (2 * sigma_y**2))))
    return Prob / np.max(Prob)

def smooth_step(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + (b - a) * t

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

def draw_left(ax, Z, alpha=1.0, plane_alpha=1.0, label_alpha=0.0):
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
    if label_alpha > 0:
        # Shown constantly during the "exploration" phase, to mark that
        # the displayed plane is a sample from the manifold's posterior.
        ax.text2D(0.23, 0.6, "posterior\nsamples", transform=ax.transAxes,
                  ha="center", fontsize=FONT_ANNOTATION, color="steelblue",
                  alpha=min(alpha, label_alpha))
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
#
# After the initial reveal, the video splits its two sources of
# variation into two separately-explained sections (each introduced by
# its own title card):
#   1. "training": the posterior-mean manifold's orientation and the
#     base distribution are both learned. This has two sub-cases, each
#     preceded by a brief indicator card:
#       - "sequential": panel a appears alone with its orientation
#         sweeping continuously (continuous_orientation); once it
#         settles, panels b/c fade in and the base distribution sweeps
#         (continuous_distribution) while the orientation stays fixed.
#       - "simultaneous": all three panels are already visible and the
#         orientation + distribution sweep together
#         (continuous_orientation_and_distribution).
#   2. "exploration": epistemic uncertainty over the manifold, at a
#     fixed state -- the displayed manifold periodically swaps between
#     posterior samples (plane_switch_schedule) while the base
#     distribution stays fixed.
# -------------------------------------------------------
FPS = 20
# Frame content/counts below are all still authored against FPS (so every
# *_DUR/*_SEC constant means what it says), but the encoded video is played
# back at OUTPUT_FPS instead -- packing the same frames into less wall-clock
# time speeds up the whole video uniformly (here, 1.5x = 50% faster)
# without touching any animation timing/logic.
OUTPUT_FPS = FPS * 1.5
REVEAL_DUR       = 30    # frames to fade in each panel
HOLD1_DUR        = 30    # hold after all panels visible
TITLE_DUR_SEC    = 5     # each mid-video title card
SUBTITLE_DUR_SEC = 3   # brief "sequential"/"simultaneous" indicator cards
SEQ_A_DUR        = 120   # sequential case, part 1: panel a alone, orientation sweeps
SEQ_B_FADE_DUR   = 30    # sequential case, part 2: panels b/c fade-in stretch
SEQ_B_DUR        = 120   # sequential case, part 2 total (fade-in + distribution sweep)
CONTINUOUS_DUR   = 120   # simultaneous case: manifold + distribution drift together
SWITCH_DUR       = 120   # posterior-sample manifold switching, distribution fixed ("exploration")
HOLD2_DUR        = 40    # final hold

TITLE_DUR = round(FPS * TITLE_DUR_SEC)
SUBTITLE_DUR = round(FPS * SUBTITLE_DUR_SEC)

F_MID_START   = REVEAL_DUR                   # 30
F_RIGHT_START = REVEAL_DUR * 2               # 60
F_HOLD1_END   = REVEAL_DUR * 3 + HOLD1_DUR   # 120

TOTAL_FRAMES_NO_EXPLORATION = (
    F_HOLD1_END
    + TITLE_DUR                                # training overview title
    + SUBTITLE_DUR + SEQ_A_DUR + SEQ_B_DUR     # sequential case
    + SUBTITLE_DUR + CONTINUOUS_DUR            # simultaneous case
    + HOLD2_DUR                                # final hold (no exploration section)
)
TOTAL_FRAMES = (
    TOTAL_FRAMES_NO_EXPLORATION
    + TITLE_DUR + SWITCH_DUR                   # exploration (optional, see --exploration)
)


def get_alphas(frame):
    alpha1 = 1.0
    alpha2 = smooth_step((frame - F_MID_START) / REVEAL_DUR)
    alpha3 = smooth_step((frame - F_RIGHT_START) / REVEAL_DUR)
    return alpha1, alpha2, alpha3


# -------------------------------------------------------
# "Exploration": hold the manifold at one orientation, then periodically
# swap to a different posterior-sample orientation: fade the plane out,
# swap, then fade it back in. The base distribution is NOT varied during
# this phase (see main()).
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


def plane_switch_schedule(local_frame):
    """local_frame in [0, SWITCH_DUR) -> (a, b, plane_alpha)."""
    half_hold = PLANE_HOLD_DUR / 2
    leg = half_hold  # end of the first (halved) hold
    for i in range(len(_PLANE_LEGS) - 1):
        if local_frame < leg:
            return (*_PLANE_LEGS[i], 1.0)
        trans_end = leg + PLANE_TRANS_DUR
        if local_frame < trans_end:
            t = (local_frame - leg) / PLANE_TRANS_DUR
            dip = _plane_transition(t)
            orient = _PLANE_LEGS[i] if t < 0.5 else _PLANE_LEGS[i + 1]
            return (*orient, dip)
        is_last = i == len(_PLANE_LEGS) - 2
        hold = half_hold if is_last else PLANE_HOLD_DUR
        leg = trans_end + hold

    return (*_PLANE_LEGS[-1], 1.0)


# -------------------------------------------------------
# "Training": the manifold orientation and the base distribution both
# sweep smoothly and *do not return* to where they started -- training
# moves the posterior mean somewhere new, it doesn't loop. The sequential
# and simultaneous cases each get their own distinct final
# orientation/distribution (rather than reusing one another's) so the
# two demonstrations don't just look like the same motion restaged.
# -------------------------------------------------------

def _orientation_sweep(t, target):
    """One smooth (single, not staged-through-waypoints) sweep from the
    base orientation to `target`, moving simultaneously in both
    parameters throughout -- not a relay of separate a-then-b legs."""
    s = smooth_step(t)
    a_tgt, b_tgt = target
    return lerp(A_BASE, a_tgt, s), lerp(B_BASE, b_tgt, s)


# Both targets are hand-picked (not just a random posterior sample) so
# that |Δa| and |Δb| from the base orientation are comparable -- if one
# parameter swings much further than the other (e.g. a near-symmetric
# sign flip in just one of them), that swing visually dominates and the
# whole sweep reads as a simple single-axis rock instead of the plane
# actually tumbling through orientation space.
SEQ_ORIENT_FINAL = (-0.38, -0.35)  # sequential case
SIM_ORIENT_FINAL = (0.36, -0.28)   # simultaneous case -- a different final tilt


def continuous_orientation(t):
    """Sequential case's orientation sweep. t in [0, 1] -> (a, b)."""
    return _orientation_sweep(t, SEQ_ORIENT_FINAL)


def continuous_orientation_simultaneous(t):
    """Simultaneous case's orientation sweep -- a different final tilt
    than the sequential case's. t in [0, 1] -> (a, b)."""
    return _orientation_sweep(t, SIM_ORIENT_FINAL)


def _distribution_sweep(t, sigma_end, mu_end):
    """One smooth sweep from the shared isotropic-wide, centered start
    to (sigma_end, mu_end)."""
    s = smooth_step(t)
    sigma_x = lerp(DIST_SIGMA_START, sigma_end[0], s)
    sigma_y = lerp(DIST_SIGMA_START, sigma_end[1], s)
    mu_x = lerp(0.0, mu_end[0], s)
    mu_y = lerp(0.0, mu_end[1], s)
    return sigma_x, sigma_y, mu_x, mu_y


# Both cases start from the same isotropic, wide, centered distribution
# but converge (monotonically, not periodically) to their own distinct
# anisotropic end state, so the two demonstrations end up looking
# different from one another, not just differently staged versions of
# the same end point.
DIST_SIGMA_START = 1.3  # isotropic, wide, shared starting point

# Sequential: wide along latent action 1 (a third of the isotropic
# start), narrow along latent action 2; mean drifts to the upper right.
SEQ_DIST_SIGMA_END = (DIST_SIGMA_START / 3, 0.18)
SEQ_DIST_MU_END = (1.0, 1.0)

# Simultaneous: the opposite anisotropy (narrow along latent action 1,
# wide along latent action 2); mean drifts to the lower left instead.
SIM_DIST_SIGMA_END = (0.18, DIST_SIGMA_START / 3)
SIM_DIST_MU_END = (-1.0, -1.0)


def continuous_distribution(t):
    """Sequential case's distribution sweep. t in [0, 1] ->
    (sigma_x, sigma_y, mu_x, mu_y)."""
    return _distribution_sweep(t, SEQ_DIST_SIGMA_END, SEQ_DIST_MU_END)


def continuous_distribution_simultaneous(t):
    """Simultaneous case's distribution sweep -- a different end mean
    and (oppositely) anisotropic covariance than the sequential case's.
    t in [0, 1] -> (sigma_x, sigma_y, mu_x, mu_y)."""
    return _distribution_sweep(t, SIM_DIST_SIGMA_END, SIM_DIST_MU_END)


def continuous_orientation_and_distribution(t):
    """t in [0, 1] -> (a, b, sigma_x, sigma_y, mu_x, mu_y): the
    simultaneous case's orientation and distribution sweeps, composed
    together (each is that case's own distinct sweep, not the
    sequential case's). The orientation sweep runs in reverse (target ->
    base) while the distribution still sweeps start -> end, so the two
    don't just mirror each other."""
    a, b = continuous_orientation_simultaneous(1.0 - t)
    sigma_x, sigma_y, mu_x, mu_y = continuous_distribution_simultaneous(t)
    return a, b, sigma_x, sigma_y, mu_x, mu_y


# "Uncertainty-guided exploration": the base distribution held fixed
# while the manifold swaps between posterior samples -- its own distinct
# mean/anisotropic covariance too, different from either training case's
# end state and from the (0, 0)/isotropic distribution shown before any
# training happens.
EXPLORE_SIGMA = (0.9, 0.3)
EXPLORE_MU = (0.6, -0.9)


# Content only ever occupies roughly x:[260, 1799] / y:[62, 575] of the
# 2000x600 canvas (measured across every phase of the animation) -- the
# rest is unused margin from the default matplotlib layout. Cropped here
# (with a little padding) rather than by touching the figure's margins,
# since all the panel/label/overlay positions are calibrated in absolute
# figure-fraction coordinates and would need re-deriving otherwise.
# Measured at REF_DPI; scaled to whatever DPI is actually rendered at
# (see scaled_crop()), since matplotlib fonts/linewidths are specified
# in points and already scale with DPI automatically -- only these
# pixel offsets need to be told to follow along.
REF_DPI = 100
CROP_X0_REF, CROP_Y0_REF, CROP_W_REF, CROP_H_REF = 245, 47, 1570, 544


def scaled_crop(dpi):
    s = dpi / REF_DPI
    x0 = round(CROP_X0_REF * s)
    y0 = round(CROP_Y0_REF * s)
    # keep even (yuv420p requires it); round each dimension independently
    w = 2 * round(CROP_W_REF * s / 2)
    h = 2 * round(CROP_H_REF * s / 2)
    return x0, y0, w, h

# Title cards: black text on white, held for TITLE_DUR_SEC (or
# INTRO_DURATION_SEC for the one before the animation starts).
INTRO_HEADING = "" # "Compositional Manifold Policies"
INTRO_TEXT = ("We construct policies by composing task-agnostic\n"
              "manifolds with task-specific base distributions")
INTRO_FONTSIZE = 30
INTRO_DURATION_SEC = 4
INTRO_TEXT_Y_FRAC = 0.35  # distance down from the crop window's top edge (0=top, 1=bottom)

TITLE_TRAIN_HEADING = "Policy Search"
TITLE_TRAIN_TEXT = ("The manifold and base distribution can be learned\n"
                     "sequentially or simultaneously")
TITLE_EXPLORE_HEADING = "Uncertainty-guided exploration of action space"
TITLE_EXPLORE_TEXT = ("We maintain a distribution over the optimal manifold\n"
                       "and sample from it for action generation")
TITLE_FONTSIZE = INTRO_FONTSIZE
TITLE_TEXT_Y_FRAC = INTRO_TEXT_Y_FRAC  # same shifted-up height as the intro card
TITLE_HEADING_GAP = 0.11  # heading baseline above center, in axes-fraction of the crop height
TITLE_UNDERLINE_GAP = 0.012  # gap between heading text and its underline

# Brief (no heading) indicator cards marking which training sub-case is
# about to play.
SUBTITLE_SEQUENTIAL_TEXT = "Sequential Manifold and Task Learning"
SUBTITLE_SIMULTANEOUS_TEXT = "Simultaneous Manifold and Task Learning"


def render_title_frame(text, fig_w, fig_h, dpi, fontsize=INTRO_FONTSIZE, y_frac=0.5,
                        heading=None, crop=None):
    """Render a title card on the same (uncropped) canvas size as the
    main animation, with the text centered on where the crop window will
    end up -- so after the shared crop filter it lands at the intended
    position, at the same final pixel size as every other frame.

    If given, `heading` is drawn bold and underlined above `text`.
    Matplotlib's mathtext has no \\underline support, so the rule is
    drawn by hand: render the heading, measure its actual rendered
    width via the renderer, then draw a line spanning that width.

    `crop` is the (x0, y0, w, h) the caller will crop this frame to
    (defaults to the reference crop at REF_DPI, for standalone/preview
    calls); pass scaled_crop(dpi) when rendering at a different DPI.
    """
    crop_x0, crop_y0, crop_w, crop_h = crop or (CROP_X0_REF, CROP_Y0_REF, CROP_W_REF, CROP_H_REF)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cx = (crop_x0 + crop_w / 2) / (fig_w * dpi)
    cy = 1 - (crop_y0 + crop_h * y_frac) / (fig_h * dpi)

    if heading:
        heading_y = cy + TITLE_HEADING_GAP
        h_artist = ax.text(cx, heading_y, heading, ha="center", va="bottom",
                            fontsize=fontsize, color="black", fontproperties=TITLE_FONT)
        fig.canvas.draw()
        bbox = h_artist.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, _) = ax.transAxes.inverted().transform(
            [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        underline_y = y0 - TITLE_UNDERLINE_GAP
        ax.plot([x0, x1], [underline_y, underline_y], color="black",
                linewidth=2, transform=ax.transAxes)

    ax.text(cx, cy, text, ha="center", va="top" if heading else "center",
            fontsize=fontsize, color="black", linespacing=1.5, fontproperties=TITLE_FONT)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(frame).convert("RGB")
    plt.close(fig)
    return img


def main(dpi=REF_DPI, out_name="compositional_policy_animation.mp4", include_exploration=False):
    FIG_W, FIG_H, DPI = 20, 6, dpi
    crop = scaled_crop(DPI)
    crop_x0, crop_y0, crop_w, crop_h = crop
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

    total_frames = TOTAL_FRAMES if include_exploration else TOTAL_FRAMES_NO_EXPLORATION
    print(f"Rendering {total_frames} frames, played back at {OUTPUT_FPS:.0f} fps "
          f"({total_frames / OUTPUT_FPS:.1f}s)...")

    out = os.path.join(SCRIPT_DIR, "static", "videos", out_name)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{FIG_W * DPI}x{FIG_H * DPI}", "-pix_fmt", "rgb24",
         "-r", str(OUTPUT_FPS), "-i", "-", "-an",
         "-vf", f"crop={crop_w}:{crop_h}:{crop_x0}:{crop_y0}",
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    def write_title(text, y_frac, n_frames, heading=None):
        frame_bytes = render_title_frame(text, FIG_W, FIG_H, DPI, fontsize=TITLE_FONTSIZE,
                                          y_frac=y_frac, heading=heading, crop=crop).tobytes()
        for _ in range(n_frames):
            ffmpeg.stdin.write(frame_bytes)

    def write_scene(a, b, sigma_x, sigma_y, alpha1, alpha2, alpha3, plane_alpha,
                     label_alpha=0.0, mu_x=0.0, mu_y=0.0, show_panel_labels=True):
        Z = a * X + b * Y
        Prob = get_distribution(sigma_x, sigma_y, mu_x, mu_y)
        norm = plt.Normalize(Prob.min(), Prob.max())
        colors = plt.colormaps[cmap_type](norm(Prob))

        draw_left(ax1, Z, alpha=alpha1, plane_alpha=plane_alpha, label_alpha=label_alpha)
        draw_middle(ax2, Prob, alpha=alpha2)
        draw_right(ax3, Z, colors, alpha=alpha3, plane_alpha=plane_alpha)

        la.set_alpha(alpha1 if show_panel_labels else 0.0)
        lb.set_alpha(max(alpha2, alpha3) if show_panel_labels else 0.0)
        arrow_g.set_alpha(alpha3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        composited = composite_overlays(
            buf, overlays,
            {"M": alpha1, "piz": alpha2, "pia": alpha3, "g": alpha3},
        )
        ffmpeg.stdin.write(composited.convert("RGB").tobytes())

    # Intro card
    write_title(INTRO_TEXT, INTRO_TEXT_Y_FRAC, round(FPS * INTRO_DURATION_SEC), heading=INTRO_HEADING)

    # Reveal + hold: panels fade in at the base orientation/distribution
    for frame in range(F_HOLD1_END):
        alpha1, alpha2, alpha3 = get_alphas(frame)
        write_scene(A_BASE, B_BASE, 0.5, 1.0, alpha1, alpha2, alpha3, plane_alpha=1.0,
                    show_panel_labels=False)

    # Title (training) -> two sub-cases: sequential, then simultaneous
    write_title(TITLE_TRAIN_TEXT, TITLE_TEXT_Y_FRAC, TITLE_DUR, heading=TITLE_TRAIN_HEADING)

    # Sequential case, part 1: panel a alone, orientation sweeps through
    # several distinct tilts, ending somewhere new (not back at base);
    # b/c stay hidden throughout.
    write_title(SUBTITLE_SEQUENTIAL_TEXT, TITLE_TEXT_Y_FRAC, SUBTITLE_DUR)
    for local_frame in range(SEQ_A_DUR):
        t = local_frame / SEQ_A_DUR
        a, b = continuous_orientation(t)
        write_scene(a, b, 0.5, 1.0, 1.0, 0.0, 0.0, plane_alpha=1.0, show_panel_labels=False)

    # Sequential case, part 2: panels b/c fade in at the orientation
    # part 1 ended on (held fixed from here on), then the base
    # distribution sweeps on its own -- starting from the same
    # isotropic-wide state continuous_distribution(0) sweeps from, so
    # there's no jump once it takes over after the fade.
    seq_a, seq_b = continuous_orientation(1.0)
    fade_sigma_x, fade_sigma_y, fade_mu_x, fade_mu_y = continuous_distribution(0.0)
    for local_frame in range(SEQ_B_DUR):
        fade = smooth_step(local_frame / SEQ_B_FADE_DUR)
        if local_frame < SEQ_B_FADE_DUR:
            sigma_x, sigma_y, mu_x, mu_y = fade_sigma_x, fade_sigma_y, fade_mu_x, fade_mu_y
        else:
            t = (local_frame - SEQ_B_FADE_DUR) / (SEQ_B_DUR - SEQ_B_FADE_DUR)
            sigma_x, sigma_y, mu_x, mu_y = continuous_distribution(t)
        write_scene(seq_a, seq_b, sigma_x, sigma_y, 1.0, fade, fade, plane_alpha=1.0,
                    mu_x=mu_x, mu_y=mu_y, show_panel_labels=False)

    # Simultaneous case: all three panels already visible, orientation
    # and distribution sweep together.
    write_title(SUBTITLE_SIMULTANEOUS_TEXT, TITLE_TEXT_Y_FRAC, SUBTITLE_DUR)
    for local_frame in range(CONTINUOUS_DUR):
        t = local_frame / CONTINUOUS_DUR
        a, b, sigma_x, sigma_y, mu_x, mu_y = continuous_orientation_and_distribution(t)
        write_scene(a, b, sigma_x, sigma_y, 1.0, 1.0, 1.0, plane_alpha=1.0, mu_x=mu_x, mu_y=mu_y,
                    show_panel_labels=False)

    if include_exploration:
        # Title (exploration) -> posterior-sample manifold switching, base
        # distribution held fixed. The "posterior samples" label is shown
        # constantly (not just around each swap) throughout this phase, and
        # kept on into the final hold below so it doesn't vanish right as
        # the phase ends.
        write_title(TITLE_EXPLORE_TEXT, TITLE_TEXT_Y_FRAC, TITLE_DUR, heading=TITLE_EXPLORE_HEADING)
        for local_frame in range(SWITCH_DUR):
            a, b, plane_alpha = plane_switch_schedule(local_frame)
            write_scene(a, b, *EXPLORE_SIGMA, 1.0, 1.0, 1.0, plane_alpha,
                        label_alpha=1.0, mu_x=EXPLORE_MU[0], mu_y=EXPLORE_MU[1])

        # Final hold, back at the base orientation but still at the
        # exploration phase's distribution (so there's no jump into the
        # hold right as it ends)
        for _ in range(HOLD2_DUR):
            write_scene(A_BASE, B_BASE, *EXPLORE_SIGMA, 1.0, 1.0, 1.0, plane_alpha=1.0,
                        label_alpha=1.0, mu_x=EXPLORE_MU[0], mu_y=EXPLORE_MU[1])
    else:
        # No exploration section: just hold on the simultaneous case's
        # final state (no jump into the hold right as the loop ends).
        a, b, sigma_x, sigma_y, mu_x, mu_y = continuous_orientation_and_distribution(1.0)
        for _ in range(HOLD2_DUR):
            write_scene(a, b, sigma_x, sigma_y, 1.0, 1.0, 1.0, plane_alpha=1.0,
                        mu_x=mu_x, mu_y=mu_y, show_panel_labels=False)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"Saved: {out}")


if __name__ == "__main__":
    # `--preview` renders at a quarter of the reference DPI and a much
    # coarser surface grid, to a separate file, for fast iteration on
    # timing/text/staging -- rerun without the flag for the real output.
    # `--exploration` adds the final "uncertainty-guided exploration"
    # section back in; it's left out by default.
    include_exploration = "--exploration" in sys.argv
    if "--preview" in sys.argv:
        set_grid_resolution(15)
        main(dpi=REF_DPI // 4, out_name="compositional_policy_animation.preview.mp4",
             include_exploration=include_exploration)
    else:
        main(include_exploration=include_exploration)