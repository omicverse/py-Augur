from __future__ import annotations

from typing import Union, Optional, Literal, Sequence
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap


def register_function(*args, **kwargs):
    """No-op decorator for compatibility with omicverse-style registration."""
    def decorator(func):
        return func
    return decorator


@register_function(
    aliases=["棒棒糖图", "lollipop", "cell_type_prioritization"],
    category="pl",
    description="Create a lollipop plot showing cell type priorities (AUC values)",
    examples=[
        "# Basic lollipop plot",
        "pyaugur.plot_lollipop(augur_results)",
        "# Top 10 cell types",
        "pyaugur.plot_lollipop(augur_results, top_n=10)",
    ],
    related=["plot_augur", "plot_umap"],
)
def plot_lollipop(
    augur_results: dict,
    *,
    top_n: Optional[int] = None,
    figsize: tuple = (8, 6),
    color: str = "#4C72B0",
    title: Optional[str] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""
    Create a lollipop plot showing cell type priorities (AUC values).

    Parameters
    ----------
    augur_results : dict
        Output from ``calculate_auc()``. Must contain ``'AUC'`` or ``'CCC'`` key.
    top_n : int or None
        Number of top cell types to display. ``None`` shows all. (None)
    figsize : tuple
        Figure size ``(width, height)``. ((8, 6))
    color : str
        Color for lollipop markers and stems. ('#4C72B0')
    title : str or None
        Plot title. Defaults to "Cell Type Prioritization". (None)
    show : bool or None
        Whether to call ``plt.show()``. If ``None`, defers to global setting.
    save : bool, str, or None
        If ``True``, saves to default path. If ``str``, saves to that path.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on. Creates new figure if ``None``.
    return_fig : bool or None
        If ``True``, forces return of the figure object.
    **kwargs
        Additional keyword arguments passed to ``ax.hlines`` and ``ax.scatter``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes, or None
    """
    if "AUC" in augur_results:
        auc_df = augur_results["AUC"].copy()
        metric_col = "auc"
    elif "CCC" in augur_results:
        auc_df = augur_results["CCC"].copy()
        metric_col = "ccc"
    else:
        raise ValueError("Results must contain 'AUC' or 'CCC' key")

    if top_n is not None:
        auc_df = auc_df.head(top_n)

    auc_df = auc_df.sort_values(metric_col, ascending=True).reset_index(drop=True)

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Draw stems
    ax.hlines(
        y=auc_df["cell_type"], xmin=0, xmax=auc_df[metric_col],
        color=color, linewidth=1.5, alpha=0.7, **kwargs,
    )

    # Draw dots
    ax.scatter(
        auc_df[metric_col], auc_df["cell_type"], color=color, s=80,
        zorder=3, edgecolors="white", linewidths=0.5, **kwargs,
    )

    # Labels
    ax.set_xlabel("AUC" if metric_col == "auc" else "CCC", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(title or "Cell Type Prioritization", fontsize=14, fontweight="bold")

    # Add value labels
    for i, row in auc_df.iterrows():
        ax.text(
            row[metric_col] + 0.01, i, "%.3f" % row[metric_col],
            va="center", fontsize=9,
        )

    # Set x-axis limits
    ax.set_xlim(0, auc_df[metric_col].max() * 1.15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if save:
        plt.savefig(save if isinstance(save, str) else "lollipop.pdf", bbox_inches="tight")
    if show:
        plt.show()
    if created_ax or return_fig:
        return fig, ax
    return ax


@register_function(
    aliases=["UMAP图", "umap_overlay", "优先级UMAP"],
    category="pl",
    description="Superimpose cell type prioritization scores onto a UMAP plot",
    examples=[
        "# Default UMAP overlay with axis arrows",
        "pyaugur.plot_umap(adata, augur_results)",
        "# Rank mode with labels, no frame",
        "pyaugur.plot_umap(adata, augur_results, mode='rank', top_n=5, frameon=False)",
    ],
    related=["plot_lollipop", "plot_augur"],
)
def plot_umap(
    input,
    augur_results: dict,
    *,
    mode: Literal["default", "rank"] = "default",
    cell_type_col: str = "cell_type",
    label_col: str = "label",
    figsize: tuple = (5, 5),
    point_size: Optional[float] = None,
    alpha: float = 0.8,
    palette: str = "cividis",
    top_n: int = 0,
    frameon: Union[str, bool] = "small",
    legend_fontsize: int = 12,
    legend_loc: str = "right margin",
    colorbar_loc: str = "right",
    title: Optional[str] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""
    Superimpose cell type prioritizations onto a UMAP plot.

    Visualize the global landscape of the perturbation response by coloring
    cells according to their cell type's AUC score (continuous color scale),
    matching R Augur's ``plot_umap`` behavior. Styled after omicverse's
    embedding plots: axis arrows (``frameon='small'``), inset colorbar,
    rasterized scatter points.

    Parameters
    ----------
    input : AnnData
        AnnData object with ``X_umap`` in ``.obsm``.
    augur_results : dict
        Output from ``calculate_auc()``.
    mode : str
        ``'default'`` for raw AUC values, ``'rank'`` for relative rank percentage.
    cell_type_col : str
        Column name for cell types in ``input.obs``. ('cell_type')
    label_col : str
        Column name for condition labels in ``input.obs``. ('label')
    figsize : tuple
        Figure size. ((5, 5))
    point_size : float or None
        Point size for scatter plot. If ``None``, auto-calculated as
        ``120000 / n_cells``. (None)
    alpha : float
        Point transparency. (0.8)
    palette : str
        Color palette name. Options: ``'viridis'``, ``'cividis'``, ``'plasma'``,
        ``'magma'``, ``'inferno'``, or any matplotlib colormap name. ('cividis')
    top_n : int
        Number of top prioritized cell types to label (0 for no labels). (0)
    frameon : str or bool
        ``'small'`` draws axis arrows at bottom-left corner (default).
        ``False`` hides the frame entirely. ``True`` keeps the regular frame.
    legend_fontsize : int
        Font size for axis labels and legend. (12)
    legend_loc : str
        Legend location. ('right margin')
    colorbar_loc : str
        Colorbar location. ``'right'`` for inset colorbar, ``None`` to hide. ('right')
    title : str or None
        Plot title. Defaults to "Cell Type Prioritization". (None)
    show : bool or None
        Whether to call ``plt.show()``.
    save : bool, str, or None
        If ``True``, saves to default path. If ``str``, saves to that path.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on.
    return_fig : bool or None
        If ``True``, forces return of the figure object.
    **kwargs
        Additional keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes, or None
    """
    if not hasattr(input, "obsm") or "X_umap" not in input.obsm:
        raise ValueError("Input must be AnnData with 'X_umap' in .obsm")

    # Get AUC results
    if "AUC" in augur_results:
        aucs = augur_results["AUC"].copy()
    elif "CCC" in augur_results:
        aucs = augur_results["CCC"].copy()
    else:
        raise ValueError("Results must contain 'AUC' or 'CCC' key")

    metric_col = "auc" if "auc" in aucs.columns else "ccc"

    # Calculate fill values based on mode
    if mode == "rank":
        aucs["rank"] = aucs[metric_col].rank()
        aucs["rank_pct"] = aucs["rank"] / len(aucs)
        aucs["rank_pct"] = (aucs["rank_pct"] - aucs["rank_pct"].min()) / \
                          (aucs["rank_pct"].max() - aucs["rank_pct"].min())
        aucs["fill"] = aucs["rank_pct"]
        legend_name = "Rank (%)"
    else:
        aucs["fill"] = aucs[metric_col]
        legend_name = "AUC" if metric_col == "auc" else "CCC"

    # Get UMAP coordinates and metadata
    umap_coords = input.obsm["X_umap"]
    meta = input.obs.copy()
    n_cells = umap_coords.shape[0]

    # Map AUC scores to each cell based on cell type
    auc_map = dict(zip(aucs["cell_type"], aucs["fill"]))
    cell_auc = meta[cell_type_col].map(auc_map).values

    # Auto point size (omicverse convention)
    if point_size is None:
        point_size = 120000 / n_cells

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get colormap
    if palette in ["viridis", "cividis", "plasma", "magma", "inferno"]:
        cmap = plt.cm.__getattribute__(palette)
    else:
        cmap = plt.cm.get_cmap(palette) if isinstance(palette, str) else palette

    # Plot points colored by AUC score (rasterized for performance)
    scatter = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=cell_auc, cmap=cmap, s=point_size, alpha=alpha,
        edgecolors="none", rasterized=True, **kwargs,
    )

    # Colorbar: inset (omicverse style) or side
    if colorbar_loc is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from matplotlib.ticker import MaxNLocator

        cax = inset_axes(ax, width="2%", height="30%", loc="lower right", borderpad=0)
        cb = plt.colorbar(scatter, cax=cax, orientation="vertical")
        cb.locator = MaxNLocator(nbins=3, integer=False)
        cb.update_ticks()
        cb.set_label(legend_name, fontsize=legend_fontsize - 2)
        cb.outline.set_visible(False)

    # Add labels for top N cell types
    if top_n > 0:
        labeled_types = aucs.nlargest(top_n, "fill")["cell_type"].values
        for ct in labeled_types:
            mask = meta[cell_type_col].values == ct
            if mask.any():
                median_x = np.median(umap_coords[mask, 0])
                median_y = np.median(umap_coords[mask, 1])
                ax.annotate(
                    ct, (median_x, median_y),
                    fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white", alpha=0.8, edgecolor="gray",
                    ),
                )

    # Frame handling (omicverse style)
    if frameon is False:
        ax.axis("off")
    elif frameon == "small":
        ax.axis("off")
        # Draw axis arrows at bottom-left corner
        x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
        y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
        x_range = (x_max - x_min) / 6
        y_range = (y_max - y_min) / 6
        arrow_scale = 10
        arrow_width = 0.01

        # X-axis arrow
        ax.arrow(
            x=x_min - x_range / 5, y=y_min,
            dx=x_range + x_range / arrow_scale, dy=0,
            width=arrow_width, color="k",
            head_width=y_range * 2 / arrow_scale,
            head_length=x_range * 2 / arrow_scale,
            overhang=0.5,
        )
        # Y-axis arrow
        ax.arrow(
            x=x_min, y=y_min - y_range / 5,
            dx=0, dy=y_range + y_range / arrow_scale,
            width=arrow_width, color="k",
            head_width=x_range * 2 / arrow_scale,
            head_length=y_range * 2 / arrow_scale,
            overhang=0.5,
        )
        # Axis labels at arrow bases
        ax.text(x_min, y_min - y_range / 2, "UMAP1",
                fontsize=legend_fontsize, ha="center", va="center")
        ax.text(x_min - x_range / 2, y_min, "UMAP2",
                fontsize=legend_fontsize, ha="center", va="center", rotation="vertical")
    else:
        ax.set_xlabel("UMAP1", fontsize=legend_fontsize)
        ax.set_ylabel("UMAP2", fontsize=legend_fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.set_xticks([])
        ax.set_yticks([])

    # Title
    if title is None:
        title = "Cell Type Prioritization"
    ax.set_title(title, fontsize=legend_fontsize + 2, fontweight="bold")
    ax.set_aspect("equal")

    if save:
        plt.savefig(save if isinstance(save, str) else "umap.pdf", bbox_inches="tight")
    if show:
        plt.show()
    if created_ax or return_fig:
        return fig, ax
    return ax


@register_function(
    aliases=["特征重要性", "feature_importance", "基因重要性"],
    category="pl",
    description="Plot the most important features (genes) for a cell type",
    examples=[
        "# Default: top prioritized cell type",
        "pyaugur.plot_important_features(augur_results)",
        "# Specific cell type, top 20 genes",
        "pyaugur.plot_important_features(augur_results, cell_type='T cells', top_n=20)",
    ],
    related=["plot_augur", "plot_lollipop"],
)
def plot_important_features(
    augur_results: dict,
    *,
    cell_type: Optional[str] = None,
    top_n: int = 10,
    figsize: tuple = (10, 6),
    color: str = "#DD8452",
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""
    Plot the most important features (genes) for a cell type.

    Parameters
    ----------
    augur_results : dict
        Output from ``calculate_auc()``. Must contain ``'feature_importance'`` key.
    cell_type : str or None
        Cell type to show features for. Defaults to top prioritized cell type. (None)
    top_n : int
        Number of top features to show. (10)
    figsize : tuple
        Figure size. ((10, 6))
    color : str
        Bar color. ('#DD8452')
    show : bool or None
        Whether to call ``plt.show()``.
    save : bool, str, or None
        If ``True``, saves to default path. If ``str``, saves to that path.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on.
    return_fig : bool or None
        If ``True``, forces return of the figure object.
    **kwargs
        Additional keyword arguments passed to ``ax.barh``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes, or None
    """
    if "feature_importance" not in augur_results:
        raise ValueError("Results must contain 'feature_importance' key")

    imp_df = augur_results["feature_importance"].copy()

    # Select cell type
    if cell_type is None:
        if "AUC" in augur_results:
            cell_type = augur_results["AUC"]["cell_type"].iloc[0]
        elif "CCC" in augur_results:
            cell_type = augur_results["CCC"]["cell_type"].iloc[0]

    imp_ct = imp_df[imp_df["cell_type"] == cell_type]

    # Aggregate across subsamples and folds
    imp_agg = imp_ct.groupby("gene")["importance"].mean().reset_index()
    imp_agg = imp_agg.sort_values("importance", ascending=False).head(top_n)
    imp_agg = imp_agg.sort_values("importance", ascending=True).reset_index(drop=True)

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot
    y_pos = np.arange(len(imp_agg))
    ax.barh(y_pos, imp_agg["importance"], color=color, alpha=0.8, height=0.7, **kwargs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_agg["gene"], fontsize=9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(
        "Top %d Important Features - %s" % (top_n, cell_type),
        fontsize=14, fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.grid(False)

    if save:
        plt.savefig(save if isinstance(save, str) else "important_features.pdf", bbox_inches="tight")
    if show:
        plt.show()
    if created_ax or return_fig:
        return fig, ax
    return ax


@register_function(
    aliases=["综合图", "augur_combined", "augur总览"],
    category="pl",
    description="Create a combined lollipop + feature importance visualization",
    examples=[
        "# Combined plot",
        "pyaugur.plot_augur(augur_results)",
        "# Top 5 cell types",
        "pyaugur.plot_augur(augur_results, top_n=5)",
    ],
    related=["plot_lollipop", "plot_important_features"],
)
def plot_augur(
    augur_results: dict,
    *,
    top_n: Optional[int] = None,
    figsize: tuple = (12, 10),
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, None]:
    r"""
    Create a combined visualization of Augur results.

    Produces a 1x2 subplot with a lollipop plot on the left and a feature
    importance bar chart on the right.

    Parameters
    ----------
    augur_results : dict
        Output from ``calculate_auc()``.
    top_n : int or None
        Number of top cell types to display. (None)
    figsize : tuple
        Figure size. ((12, 10))
    show : bool or None
        Whether to call ``plt.show()``.
    save : bool, str, or None
        If ``True``, saves to default path. If ``str``, saves to that path.
    return_fig : bool or None
        If ``True``, forces return of the figure object.
    **kwargs
        Additional keyword arguments passed to individual plot functions.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Lollipop plot
    plot_lollipop(augur_results, top_n=top_n, ax=axes[0], **kwargs)

    # Feature importance plot
    plot_important_features(augur_results, top_n=top_n or 10, ax=axes[1], **kwargs)

    plt.tight_layout()

    if save:
        plt.savefig(save if isinstance(save, str) else "augur_combined.pdf", bbox_inches="tight")
    if show:
        plt.show()
    if return_fig or show is None:
        return fig
    return None


@register_function(
    aliases=["散点图比较", "scatterplot", "AUC对比图"],
    category="pl",
    description="Compare two cell type prioritization results as a scatterplot",
    examples=[
        "# Compare two Augur results",
        "pyaugur.plot_scatterplot(augur1, augur2)",
        "# With top labels",
        "pyaugur.plot_scatterplot(augur1, augur2, top_n=5)",
    ],
    related=["plot_differential_prioritization", "plot_lollipop"],
)
def plot_scatterplot(
    augur1: dict,
    augur2: dict,
    *,
    top_n: int = 0,
    figsize: tuple = (6, 6),
    point_size: float = 20,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""
    Compare two cell type prioritizations as a scatterplot.

    Plots AUCs from the first result on the x-axis and the second on the
    y-axis. Points are colored by delta-AUC using a coolwarm colormap.
    Useful for understanding how a parameter or preprocessing choice
    influences cell type prioritization.

    Parameters
    ----------
    augur1 : dict
        First set of Augur results from ``calculate_auc()``.
    augur2 : dict
        Second set of Augur results from ``calculate_auc()``.
    top_n : int
        Number of cell types with the largest AUC shift to label. (0)
    figsize : tuple
        Figure size. ((6, 6))
    point_size : float
        Scatter point size. (20)
    show : bool or None
        Whether to call ``plt.show()``.
    save : bool, str, or None
        If ``True``, saves to default path. If ``str``, saves to that path.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on.
    return_fig : bool or None
        If ``True``, forces return of the figure object.
    **kwargs
        Additional keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes, or None
    """
    if "AUC" not in augur1 or "AUC" not in augur2:
        raise ValueError("Both results must contain 'AUC' key")

    auc1 = augur1["AUC"].copy()
    auc2 = augur2["AUC"].copy()

    # Combine
    df = auc1.merge(auc2, on="cell_type", suffixes=(".x", ".y"))
    df["delta"] = df["auc.y"] - df["auc.x"]
    df["abs_delta"] = df["delta"].abs()

    labels = df.sort_values("abs_delta", ascending=False).head(top_n) if top_n > 0 else pd.DataFrame()

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Coolwarm colormap centered at 0
    delta_range = max(abs(df["delta"].min()), abs(df["delta"].max()))
    if delta_range > 0:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-delta_range, vcenter=0, vmax=delta_range)
    else:
        norm = None

    scatter = ax.scatter(
        df["auc.x"], df["auc.y"],
        c=df["delta"], cmap="coolwarm", norm=norm,
        s=point_size, edgecolors="black", linewidths=0.3,
        **kwargs,
    )

    # Diagonal reference line
    lim_min = min(df["auc.x"].min(), df["auc.y"].min()) - 0.02
    lim_max = max(df["auc.x"].max(), df["auc.y"].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="dotted",
            color="gray", linewidth=0.8)

    # Add labels for top shifted cell types
    if top_n > 0 and len(labels) > 0:
        for _, row in labels.iterrows():
            ax.annotate(
                row["cell_type"], (row["auc.x"], row["auc.y"]),
                fontsize=7, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.3),
            )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$ AUC", fontsize=10)
    cbar.outline.set_visible(False)

    ax.set_xlabel("AUC 1", fontsize=12)
    ax.set_ylabel("AUC 2", fontsize=12)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if save:
        plt.savefig(save if isinstance(save, str) else "scatterplot.pdf", bbox_inches="tight")
    if show:
        plt.show()
    if created_ax or return_fig:
        return fig, ax
    return ax


@register_function(
    aliases=["差异优先级", "differential_prioritization", "差异分析图"],
    category="pl",
    description="Plot differential prioritization results highlighting significant cell types",
    examples=[
        "# Plot differential prioritization",
        "pyaugur.plot_differential_prioritization(dp_results)",
        "# With top labels",
        "pyaugur.plot_differential_prioritization(dp_results, top_n=5)",
    ],
    related=["plot_scatterplot", "calculate_differential_prioritization"],
)
def plot_differential_prioritization(
    results: pd.DataFrame,
    *,
    top_n: int = 0,
    pval_threshold: float = 0.05,
    condition1_color: str = "#2166ac",
    condition2_color: str = "#b2182b",
    ns_color: str = "#cccccc",
    figsize: tuple = (6, 6),
    point_size: float = 10,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""
    Plot the results of a differential prioritization analysis.

    Creates a scatterplot of AUC from condition 1 vs condition 2, highlighting
    cell types with statistically significant differences.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ``calculate_differential_prioritization()``.
        Must contain columns: ``cell_type``, ``auc.x``, ``auc.y``, ``pval``, ``z``.
    top_n : int
        Number of significant cell types to label. (0)
    pval_threshold : float
        Significance threshold for highlighting. (0.05)
    condition1_color : str
        Color for cell types prioritized in condition 1. ('#2166ac')
    condition2_color : str
        Color for cell types prioritized in condition 2. ('#b2182b')
    ns_color : str
        Color for non-significant cell types. ('#cccccc')
    figsize : tuple
        Figure size. ((6, 6))
    point_size : float
        Scatter point size. (10)
    show : bool or None
        Whether to call ``plt.show()``.
    save : bool, str, or None
        If ``True``, saves to default path. If ``str``, saves to that path.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on.
    return_fig : bool or None
        If ``True``, forces return of the figure object.
    **kwargs
        Additional keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes, or None
    """
    required_cols = ["cell_type", "auc.x", "auc.y", "pval", "z"]
    missing = [c for c in required_cols if c not in results.columns]
    if missing:
        raise ValueError(f"Results missing required columns: {missing}")

    df = results.dropna(subset=["auc.x", "auc.y"]).copy()

    # Classify cell types
    df["color_group"] = "n.s."
    df.loc[(df["pval"] < pval_threshold) & (df["z"] > 0), "color_group"] = "condition 2"
    df.loc[(df["pval"] < pval_threshold) & (df["z"] <= 0), "color_group"] = "condition 1"

    color_map = {
        "condition 1": condition1_color,
        "condition 2": condition2_color,
        "n.s.": ns_color,
    }

    # Select labels
    sig_df = df[df["pval"] < pval_threshold].copy()
    sig_df["abs_z"] = sig_df["z"].abs()
    labels = sig_df.sort_values("abs_z", ascending=False).head(top_n)

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot each group
    for group, color in color_map.items():
        mask = df["color_group"] == group
        if mask.any():
            ax.scatter(
                df.loc[mask, "auc.x"], df.loc[mask, "auc.y"],
                c=color, s=point_size, label=group if group != "n.s." else "n.s.",
                edgecolors="none", alpha=0.8, **kwargs,
            )

    # Diagonal reference line
    lim_min = min(df["auc.x"].min(), df["auc.y"].min()) - 0.02
    lim_max = max(df["auc.x"].max(), df["auc.y"].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="dotted",
            color="gray", linewidth=0.8)

    # Add labels
    if top_n > 0 and len(labels) > 0:
        for _, row in labels.iterrows():
            ax.annotate(
                row["cell_type"], (row["auc.x"], row["auc.y"]),
                fontsize=7, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.3),
            )

    ax.set_xlabel("AUC 1", fontsize=12)
    ax.set_ylabel("AUC 2", fontsize=12)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=condition1_color,
                    markersize=6, label="condition 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=condition2_color,
                    markersize=6, label="condition 2"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ns_color,
                    markersize=6, label="n.s."),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    if save:
        plt.savefig(
            save if isinstance(save, str) else "differential_prioritization.pdf",
            bbox_inches="tight",
        )
    if show:
        plt.show()
    if created_ax or return_fig:
        return fig, ax
    return ax
