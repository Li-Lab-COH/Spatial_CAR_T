"""Shared plotting style for the spatial CAR-T project.

Single source of truth for cell-type colors. Import in any notebook/script:

    import sys
    sys.path.append("/Users/janzules/Roselab/Spatial/CAR_T/code")
    from misc.plot_style import CELL_TYPE_COLORS, get_cell_colors

To color a stacked bar / pie / scatter consistently:

    colors = get_cell_colors(df.columns)
    df.plot(kind="bar", stacked=True, color=colors)
"""

CELL_TYPE_COLORS = {
    "Unknown":            "#d3d3d3",  # light grey

    "Cancer_cell":        "#d62728",  # red

    # T cells — blues
    "CD8_T":              "#1f77b4",
    "CD4_T":              "#6baed6",
    "Treg":               "#08519c",  # dark navy, distinct from CD4

    # NK lineage — greens
    "NK":                 "#2ca02c",
    "NKT":                "#74c476",

    # B cells — purple
    "B":                  "#9467bd",

    # Stromal / vascular
    "Endothelial":        "#b30000",  # deep red
    "Fibroblast":         "#8c564b",
    "Erythrocyte":        "#54322c",

    # Monocytes — oranges (mono lineage owns this family)
    "Classical_Mono":     "#fdae6b",
    "Nonclassical_Mono":  "#e6550d",

    # Macrophages — pinks/magentas (moved out of orange to avoid mono overlap)
    "Intermediate_Mac":   "#fcc5c0",
    "M1_like_Mac":        "#e377c2",
    "M2_like_Mac":        "#ad3a8e",
    "Spp1_Mac":           "#2ad1c9",  # teal — intentional highlight

    # Neutrophils — yellows/olives
    "N1_like_Neu":        "#dbdb8d",
    "N2_like_Neu":        "#bcbd22",

    # DCs — cyans
    "cDC":                "#17becf",
    "pDC":                "#9edae5",

    # Collapsed/fallback bucket
    "Other":              "#7f7f7f",
}

DEFAULT_COLOR = "#999999"


def get_cell_colors(cell_types, default=DEFAULT_COLOR):
    """Return a list of colors aligned with ``cell_types`` order.

    Unknown labels fall back to ``default`` (mid-grey) so plotting never errors.
    """
    return [CELL_TYPE_COLORS.get(ct, default) for ct in cell_types]
