from __future__ import annotations

try:
    import open3d as o3d
except ImportError:
    class _MockO3D:
        class geometry:
            PointCloud = object
            AxisAlignedBoundingBox = object
            TriangleMesh = object
            VoxelGrid = object
        class utility:
            Vector3dVector = object
        class visualization:
            @staticmethod
            def draw_geometries(*args, **kwargs):
                pass
    o3d = _MockO3D()

try:
    import laspy
except ImportError:
    laspy = None

"""
Visualization module — matplotlib-based plotting helpers for point clouds,
DEMs, histograms, cross-sections, and density maps.
"""


from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_point_cloud(
    pcd: o3d.geometry.PointCloud,
    color_by: str = "elevation",
    *,
    figsize: tuple = (10, 8),
    point_size: float = 0.5,
    colormap: str = "viridis",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Create a 2-D scatter plot of a point cloud.

    Args:
        pcd: Input point cloud.
        color_by: ``"elevation"`` (colour by Z), ``"rgb"`` (use cloud
            colours), or ``"intensity"``.
        figsize: Figure size in inches.
        point_size: Marker size.
        colormap: Matplotlib colourmap (used when *color_by* is
            ``"elevation"``).
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    points = np.asarray(pcd.points)
    fig, ax = plt.subplots(figsize=figsize)

    if color_by == "rgb" and pcd.has_colors():
        colors = np.asarray(pcd.colors)
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=point_size)
    else:
        sc = ax.scatter(
            points[:, 0], points[:, 1],
            c=points[:, 2], s=point_size, cmap=colormap,
        )
        plt.colorbar(sc, ax=ax, label="Elevation")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dem(
    dem_array: np.ndarray,
    metadata: Dict,
    *,
    colormap: str = "terrain",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a colourised DEM image.

    Args:
        dem_array: 2-D elevation array.
        metadata: Metadata dict from DTM/DSM generation.
        colormap: Matplotlib colourmap.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        dem_array, origin="lower", cmap=colormap,
        extent=[
            metadata["x_min"], metadata["x_max"],
            metadata["y_min"], metadata["y_max"],
        ],
    )
    plt.colorbar(im, ax=ax, label="Elevation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title), 

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_histogram(
    stats: Dict,
    *,
    title: str = "Elevation Distribution",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot an elevation histogram from :func:`~lidarlens.elevation_histogram`
    output.

    Args:
        stats: Dict with ``counts`` and ``bin_edges``.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    edges = stats["bin_edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    ax.bar(centers, stats["counts"], width=(edges[1] - edges[0]) * 0.9, color="#3498db")
    ax.set_xlabel("Elevation")
    ax.set_ylabel("Point Count")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cross_section(
    profile: Dict,
    *,
    title: str = "Elevation Profile",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a terrain cross-section from :func:`~lidarlens.cross_section` output.

    Args:
        profile: Dict with ``distance`` and ``elevation`` arrays.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(profile["distance"], profile["elevation"], s=0.5, c="#2ecc71")
    ax.fill_between(
        profile["distance"], profile["elevation"],
        alpha=0.3, color="#2ecc71",
    )
    ax.set_xlabel("Distance along profile")
    ax.set_ylabel("Elevation")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_density_map(
    density: np.ndarray,
    metadata: Dict,
    *,
    colormap: str = "hot",
    title: str = "Point Density",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a point-density heat-map.

    Args:
        density: 2-D density array from :func:`~lidarlens.point_density`.
        metadata: Metadata dict.
        colormap: Matplotlib colourmap.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        density, origin="lower", cmap=colormap,
        extent=[
            metadata["x_min"], metadata["x_max"],
            metadata["y_min"], metadata["y_max"],
        ],
    )
    plt.colorbar(im, ax=ax, label="Points per cell")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_classification(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    *,
    class_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (10, 8),
    point_size: float = 0.5,
    title: str = "Classification",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a point cloud coloured by classification labels.

    Args:
        pcd: Input point cloud.
        labels: Per-point integer labels.
        class_names: Optional mapping of label → name for the legend.
        figsize: Figure size.
        point_size: Marker size.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    from lidarlens.classification import ASPRS_CLASSES

    if class_names is None:
        class_names = ASPRS_CLASSES

    points = np.asarray(pcd.points)
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = class_names.get(int(lbl), f"Class {lbl}")
        ax.scatter(
            points[mask, 0], points[mask, 1],
            c=[cmap(i)], s=point_size, label=name,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(markerscale=5, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
