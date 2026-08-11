# ValueError: 'parula' is not a valid value for cmap; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
# 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
# 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
# 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
# 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
# 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone',
# 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
# 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
# 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
# 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
# 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
# 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
# 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# cmap_type = "YlGnBu_r"
# cmap_type = 'Blues_r'
cmap_type = "bone"
# cmap_type = "magma"
# cmap_type = 'afmhot'

import subprocess

# -------------------------------------------------------
# Font sizes: every bit of text drawn in the figure is controlled
# from here.
# -------------------------------------------------------
FONT_AXIS_LABEL = 18  # "muscle 1" / "muscle 2" / "muscle M" tick labels
FONT_PANEL_TITLE = 20  # panel titles (manifold / base distribution / policy)
FONT_ANNOTATION = 18  # "posterior samples" annotation
FONT_AXIS_TITLE = 18  # ax2's "latent action 1" / "latent action 2" axis titles
FONT_PANEL_LABEL = 28  # bold "a" / "b" panel labels

FONT_LARGE = 12
FONT_MEDIUM = 11
FONT_SMALL = 9


def set_plot_style():
    font_large = 12
    font_medium = 100
    font_small = 9
    plt.rcParams.update(
        {
            "font.size": FONT_MEDIUM,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": FONT_LARGE,
            "axes.labelsize": FONT_LARGE,
            "xtick.labelsize": FONT_MEDIUM,
            "ytick.labelsize": FONT_MEDIUM,
            "legend.fontsize": FONT_LARGE,
            "lines.linewidth": 2.0,
            "mathtext.fontset": "cm",
        }
    )


set_plot_style()


def plot_additive_gp_policy_final_tight():
    # Setup Figure
    fig = plt.figure(figsize=(20, 6))

    # Grid spec for the OUTER plots
    # Middle column is tiny (0.01) just to create a 'gap' for our manual plot
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.1, 0.5, 0.1, 1], wspace=0.3)

    # -------------------------------------------------------
    # Data Generation
    # -------------------------------------------------------
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)

    # Function to generate a random plane (linear manifold)
    def generate_surface_sample(seed, perturbation=0.0, **kwargs):
        np.random.seed(seed)
        a = np.random.uniform(-0.4, 0.4)
        b = np.random.uniform(-0.4, 0.4)
        if perturbation > 0:
            a += np.random.normal(0, perturbation * 0.5)
            b += np.random.normal(0, perturbation * 0.5)
        return a * X + b * Y

    # 1. The "Mean" Manifold (Best guess)
    Z_mean = generate_surface_sample(seed=42, perturbation=0.0)

    # 2. The "Uncertainty" Samples (Alternative hypotheses)
    # We generate 3 samples slightly deviated from the mean
    Z_samples = []
    for i in range(3):
        Z_samples.append(generate_surface_sample(seed=10 + i, perturbation=0.0))

    # Base Dist params
    sigma_x = 0.5
    sigma_y = 1.0
    Prob = np.exp(-((X**2 / (2 * sigma_x**2)) + (Y**2 / (2 * sigma_y**2))))
    Prob = Prob / np.max(Prob)

    norm = plt.Normalize(Prob.min(), Prob.max())
    colors = plt.colormaps[cmap_type](norm(Prob))

    # -------------------------------------------------------
    # Helper: Central "Inverted Y" Axes
    # -------------------------------------------------------
    def add_central_axes(ax, length=2.5, z_offset=0, txt_description=""):
        ax_color = "black"
        ax_alpha = 0.8
        lw = 2.5
        o = [0, 0, z_offset]

        # Z-Axis
        ax.quiver(
            o[0],
            o[1],
            o[2],
            0,
            0,
            length * 1.4,
            color=ax_color,
            alpha=ax_alpha,
            arrow_length_ratio=0.0,
            linewidth=lw,
        )
        ax.text(
            o[0],
            o[1],
            o[2] + length * 1.4 + 0.4,
            "muscle $M$",
            fontsize=FONT_AXIS_LABEL,
            ha="center",
        )

        # X-Axis
        ax.quiver(
            o[0],
            o[1],
            o[2],
            length,
            0,
            0,
            color=ax_color,
            alpha=ax_alpha,
            arrow_length_ratio=0.0,
            linewidth=lw,
        )
        ax.text(
            o[0] + length + 0.4,
            o[1],
            o[2] - 0.4,
            "muscle 2",
            fontsize=FONT_AXIS_LABEL,
            ha="center",
        )

        # Y-Axis
        ax.quiver(
            o[0],
            o[1],
            o[2],
            0,
            -length,
            0,
            color=ax_color,
            alpha=ax_alpha,
            arrow_length_ratio=0.0,
            linewidth=lw,
        )
        ax.text(
            o[0],
            o[1] - length - 0.4,
            o[2] - 0.4,
            "muscle 1",
            fontsize=FONT_AXIS_LABEL,
            ha="center",
        )

        ax.scatter([0], [0], [z_offset], color="black", s=50, alpha=0.0)

        # ax.text(
        #     o[0], o[1], o[2] - 3, txt_description, fontsize=21, ha="center", usetex=True
        # )

    # -------------------------------------------------------
    # Panel 1: Learned Manifold
    # -------------------------------------------------------
    ax1 = fig.add_subplot(gs[0], projection="3d")

    # A. Plot "Uncertainty" Samples first (Ghosted)
    for Z_samp in Z_samples:
        ax1.plot_surface(
            X,
            Y,
            Z_samp,
            color="lightblue",
            alpha=0.15,
            rstride=4,
            cstride=4,
            linewidth=0.5,
            antialiased=True,
            shade=False,
            rasterized=True,
            zorder=1,
        )

    # Compute distance-based shading from a point light source.
    # Even a perfectly flat plane gets brightness variation this way.
    _light = np.array([5.0, -5.0, 5.0])
    _dist_sq = (X - _light[0]) ** 2 + (Y - _light[1]) ** 2 + (Z_mean - _light[2]) ** 2
    _bright = 1.0 / (1.0 + 0.015 * _dist_sq)
    _bright = (_bright - _bright.min()) / (_bright.max() - _bright.min())
    _shade_contrast = (
        0.5  # 0 = flat grey, 1 = black-to-white; increase for stronger gradient
    )
    _bright = (1.0 - _shade_contrast) + _shade_contrast * _bright
    _grey = np.zeros((*_bright.shape, 4))
    _grey[..., :3] = _bright[..., np.newaxis]
    _grey[..., 3] = 0.8

    ax1.plot_surface(
        X,
        Y,
        Z_mean,
        facecolors=_grey,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        rasterized=True,
        zorder=2,
    )

    add_central_axes(
        ax1, length=3.3, z_offset=-0.85, txt_description=r"$\mathcal{M}_\theta(s_t)$"
    )
    # ax1.set_title("learned manifold $\mathcal{M}(s_t)$\n(task-agnostic)", fontsize=16, fontweight='bold', pad=20)
    ax1.set_title(
        r"$\bf{manifold}$" + "\n(task-agnostic)",
        fontsize=FONT_PANEL_TITLE,
        y=0.95,
        linespacing=1.4,
    )  # default is about 1.2
    # ax1.set_title(r"learned manifold", fontsize=16, fontweight='bold', pad=20)
    ax1.axis("off")
    ax1.view_init(elev=25, azim=-45)
    ax1.set_zlim(-1.5, 4.4)
    ax1.set_box_aspect([1, 1, 1], zoom=1.1)  # zoom fills whitespace inside 3D box

    ax1.text2D(
        0.23,
        0.6,
        "posterior\nsamples",
        transform=ax1.transAxes,
        ha="center",
        fontsize=FONT_ANNOTATION,
        color="steelblue",
        fontweight="normal",
    )

    # -------------------------------------------------------
    # Symbols
    # -------------------------------------------------------
    # ax_plus = fig.add_subplot(gs[1])
    # ax_plus.text(
    #     -0.3, 0.43, "+", fontsize=45, ha="center", va="center", fontweight="bold"
    # )
    # ax_plus.axis("off")

    # ax_eq = fig.add_subplot(gs[3])
    # ax_eq.text(
    #     1.8,
    #     0.43,
    #     # r"$\xrightarrow{\;g_{\theta^*}(\cdot,\,s)\;}$
    #     "=",
    #     fontsize=45,
    #     ha="center",
    #     va="center",
    #     fontweight="bold",
    # )
    # ax_eq.axis("off")

    # -------------------------------------------------------
    # Panel 3: Manifold Policy
    # -------------------------------------------------------
    ax3 = fig.add_subplot(gs[4], projection="3d")

    # A. Plot "Uncertainty" Samples (Ghosted with low saturation color)
    for Z_samp in Z_samples:
        # Use a faint version of the colormap or just uniform faint color
        ax3.plot_surface(
            X,
            Y,
            Z_samp,
            color="grey",
            alpha=0.15,
            rstride=4,
            cstride=4,
            linewidth=0,
            shade=False,
            zorder=1,
        )

    surf = ax3.plot_surface(
        X,
        Y,
        Z_mean,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        rasterized=True,
        zorder=2,
    )
    add_central_axes(
        ax3,
        length=3.3,
        z_offset=-0.85,
        txt_description=r"$\pi_{\mathrm{a}}(a_t | s_t)$",
    )
    # ax3.set_title("manifold policy $\pi_{\mathrm{a}}(a_t | s_t)$\n(task-specific)", fontsize=16, fontweight='bold', pad=20)
    ax3.set_title(
        "manifold policy\n", fontsize=FONT_PANEL_TITLE, fontweight="bold", y=0.962
    )
    ax3.axis("off")
    ax3.view_init(elev=25, azim=-45)
    ax3.set_zlim(-1.5, 4.4)
    ax3.set_box_aspect([1, 1, 1], zoom=1.1)  # zoom fills whitespace inside 3D box

    # -------------------------------------------------------
    # Panel 2: Base Distribution (Absolute Positioning)
    # -------------------------------------------------------
    # [left, bottom, width, height]
    # Adjusted 'bottom' to 0.36 to sit nicely with the new margins
    ax2 = fig.add_axes([0.29, 0.205, 0.4, 0.52])

    ax2.imshow(
        Prob,
        extent=[-2, 2, -2, 2],
        origin="lower",
        cmap=cmap_type,
        interpolation="bicubic",
        aspect="equal",
    )

    ax2.axis("on")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax2.set_xlabel("latent action 1", fontsize=FONT_AXIS_TITLE, labelpad=10)
    ax2.set_ylabel("latent action 2", fontsize=FONT_AXIS_TITLE, labelpad=10)
    # ax2.set_title("base distribution $\pi_{\mathrm{z}}(z_t | s_t)$\n(task-specific)", fontsize=16, fontweight='bold', pad=38.5)
    ax2.set_title(
        r"$\bf{base\ distribution}$" + "\n(task-specific)",
        fontsize=FONT_PANEL_TITLE,
        y=1.14,
        linespacing=1.4,  # default is about 1.2
        pad=0,
    )
    # ax2.text(
    #     0,
    #     -3.03,
    #     r"$\pi_{\mathrm{z}}(z_t | s_t)$",
    #     fontsize=21,
    #     ha="center",
    #     usetex=True,
    # )

    # -------------------------------------------------------
    # Panel labels: 'a' marks the left (manifold) panel, 'b' marks
    # the combined middle+right (base distribution -> policy) group.
    # -------------------------------------------------------
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    label_y = pos1.y1 + 0.03
    fig.text(
        pos1.x0 + 0.005,
        label_y - 0.07,
        "a",
        fontsize=FONT_PANEL_LABEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        pos2.x0 - 0.025,
        label_y - 0.07,
        "b",
        fontsize=FONT_PANEL_LABEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # -------------------------------------------------------
    # FIX: Manually Trim Margins
    # -------------------------------------------------------
    # This replaces tight_layout().
    # Top=1.0 means plots go all the way to the top edge.
    # Bottom=0.0 means plots go all the way to the bottom edge.
    # plt.subplots_adjust(left=0.0, right=1., top=1.3, bottom=-0.4, wspace=0.)

    filename = "manifold_policy_" + cmap_type + ".pdf"
    plt.savefig(filename)
    subprocess.run(["pdfcrop", filename, filename])
    plt.close()  # Good practice to close figure after saving

    # plt.show()


if __name__ == "__main__":
    plot_additive_gp_policy_final_tight()
