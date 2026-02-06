import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ClusterWarning, fcluster, linkage
from scipy.spatial.distance import squareform

# --- Parsing Import ---
from .parse_files import parse_files

# Machine Learning Imports
try:
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except ImportError:
    silhouette_score = None
    PCA = None

# Optimization Import
try:
    import fastcluster
    HAS_FASTCLUSTER = True
except ImportError:
    HAS_FASTCLUSTER = False

from . import __version__

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%y-%b-%d %H:%M:%S",
    level=logging.INFO,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Genomic Distance Matrix")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input matrix file name"
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=[
            "snp-dists",
            "skani",
            "fastani",
            "mash",
            "sourmash",
            "melted",
            "poppunk",
            "ezaai",
            "nwk",
            "kwip",
            "ska",
            "bindash",
            "dashing",
            "pyani_identity",
            "pyani_errors",
            "iqtree",
            "gene_presence_absence",
            "pathogen_detection",
            "vcf",
        ],
        default="snp-dists",
        help="Input file format.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="heatcluster_matrix.png",
        help="Output figure filename. Default: heatcluster_matrix.png",
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default="heatcluster_sorted.csv",
        help="Output sorted matrix CSV filename. Default: heatcluster_sorted.csv",
    )

    # ML / Clustering Extraction
    parser.add_argument(
        "-l",
        "--cluster-out",
        type=str,
        default=None,
        help="Output filename for cluster assignments (e.g., clusters.csv).",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        help="Split tree into K groups (e.g. 5). Overrides --cluster-t.",
    )
    parser.add_argument(
        "--cluster-t",
        type=float,
        help="Split tree by distance threshold (e.g. 10 SNPs or 0.05 ANI).",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically detect the optimal K using Silhouette Scores.",
    )

    # PCA
    parser.add_argument(
        "--pca", action="store_true", help="Generate a PCA scatter plot."
    )
    parser.add_argument(
        "--pca-out",
        type=str,
        default="heatcluster_pca.png",
        help="Filename for PCA plot.",
    )

    # Visual Appearance
    parser.add_argument(
        "--title", type=str, default="Heatmap", help="Title of the plot."
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Reds_r",
        help="Matplotlib colormap (e.g., Reds_r, viridis).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI (dots per inch) for output image. Default: 300.",
    )
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Do not show numbers inside the heatmap cells.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the heatmap image (Computation only).",
    )

    # Dimensions & Text Size
    parser.add_argument("--width", type=float, help="Force figure width in inches.")
    parser.add_argument("--height", type=float, help="Force figure height in inches.")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale all font sizes by this factor.",
    )

    # Thresholds
    parser.add_argument("--vmin", type=float, help="Minimum value for color scale.")
    parser.add_argument("--vmax", type=float, help="Maximum value for color scale.")
    parser.add_argument("--hide-below", type=float, help="Mask values lower than this.")
    parser.add_argument(
        "--hide-above", type=float, help="Mask values higher than this."
    )

    parser.add_argument(
        "--no-cluster", action="store_true", help="Disable hierarchical clustering"
    )
    parser.add_argument(
        "--dendrogram",
        action="store_true",
        help="Show the dendrogram (tree) and borders",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def determine_figsize(
    num_samples: int,
    user_w: Optional[float] = None,
    user_h: Optional[float] = None,
    font_scale: float = 1.0,
) -> Tuple[float, float, Tuple[float, float]]:
    # Base Heuristics
    if num_samples >= 120:
        base_annot, base_label, base_size = 1.0, 2.0, (18, 15)
    elif num_samples >= 80:
        base_annot, base_label, base_size = 2.0, 3.0, (18, 15)
    elif num_samples >= 40:
        base_annot, base_label, base_size = 3.0, 4.0, (11, 8)
    elif num_samples >= 30:
        base_annot, base_label, base_size = 4.0, 6.0, (11, 8)
    else:
        base_annot, base_label, base_size = 6.0, 7.0, (11, 8)

    final_w = user_w if user_w else base_size[0]
    final_h = user_h if user_h else base_size[1]

    final_annot = round(base_annot * font_scale, 2)
    final_label = round(base_label * font_scale, 2)

    logging.info(f"Dimensions: {final_w}x{final_h} inches. Label Size: {final_label}.")
    return final_annot, final_label, (final_w, final_h)


def calculate_linkage(df: pd.DataFrame):
    """
    Computes hierarchical clustering linkage matrix.
    Uses fastcluster (O(N^2)) if available, else scipy (O(N^3)).
    """
    # Important: Convert Square DataFrame to Condensed Distance Matrix (1D Array)
    condensed_matrix = squareform(df.values, checks=False)

    if HAS_FASTCLUSTER:
        return fastcluster.linkage(condensed_matrix, method="average")
    else:
        return linkage(condensed_matrix, method="average")


def run_analysis(df: pd.DataFrame, args, font_tuple: tuple):
    matplotlib.use("agg")
    font_size, label_size, fig_size = font_tuple

    # --- LEVEL 1: HIERARCHICAL CLUSTERING (TREE BUILDING) ---
    linkage_matrix = None
    use_clustering = not args.no_cluster
    
    if use_clustering and df.shape[0] > 1:
        logging.info("Performing hierarchical clustering...")
        linkage_matrix = calculate_linkage(df)

    # --- LEVEL 2: FLAT CLUSTERING & AUTO-K (TREE CUTTING) ---
    final_cluster_labels = None
    
    # Logic: Do we need to calculate clusters?
    need_clusters = (
        args.cluster_out is not None
        or args.auto_k
        or args.cluster_k
        or args.cluster_t
    )

    if need_clusters and linkage_matrix is not None:
        # OPTION A: Auto-Detect K
        if args.auto_k:
            if silhouette_score is None:
                logging.error(
                    "scikit-learn not found. Install it to use --auto-k: pip install scikit-learn"
                )
                sys.exit(1)

            logging.info("Running Silhouette Analysis to find optimal K...")
            best_k = 2
            best_score = -1
            max_possible_k = min(10, len(df) - 1)

            if max_possible_k < 2:
                logging.warning(
                    "Not enough samples for Auto-K. Defaulting to K=2."
                )
            else:
                # Prepare Distance Matrix for Silhouette Validation
                diag_mean = np.diag(df).mean()
                if diag_mean > 90:
                    dist_matrix_np = (100 - df).to_numpy().copy()
                elif diag_mean <= 1.0:
                    dist_matrix_np = (1.0 - df).to_numpy().copy()
                else:
                    dist_matrix_np = df.to_numpy().copy()

                np.fill_diagonal(dist_matrix_np, 0.0)
                dist_matrix_np[dist_matrix_np < 0] = 0

                for k in range(2, max_possible_k + 1):
                    labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
                    if len(set(labels)) < 2:
                        continue
                    score = silhouette_score(
                        dist_matrix_np, labels, metric="precomputed"
                    )
                    logging.info(f"  K={k}, Silhouette Score={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_k = k

                logging.info(
                    f"Optimal K detected: {best_k} (Score: {best_score:.4f})"
                )
                args.cluster_k = best_k

        # OPTION B: Extract Labels based on final K or T
        if args.cluster_k:
            logging.info(f"Cutting tree into {args.cluster_k} clusters...")
            final_cluster_labels = fcluster(
                linkage_matrix, t=args.cluster_k, criterion="maxclust"
            )
        elif args.cluster_t:
            logging.info(f"Cutting tree at threshold {args.cluster_t}...")
            final_cluster_labels = fcluster(
                linkage_matrix, t=args.cluster_t, criterion="distance"
            )
        elif args.cluster_out:  # Fallback
            logging.warning("No clustering method provided. Defaulting to K=2.")
            final_cluster_labels = fcluster(
                linkage_matrix, t=2, criterion="maxclust"
            )

        # Save to CSV
        if final_cluster_labels is not None:
            cluster_df = pd.DataFrame(
                {"Sample": df.index, "Cluster_ID": final_cluster_labels}
            )
            out_filename = args.cluster_out if args.cluster_out else "heatcluster_clusters.csv"
            cluster_df.to_csv(out_filename, index=False)
            logging.info(f"Saved cluster assignments to {out_filename}")

    # --- LEVEL 3: PCA / VALIDATION ---
    if args.pca:
        if PCA is None:
            logging.error("scikit-learn not found. Install it to use --pca.")
        else:
            logging.info("Generating PCA plot...")
            try:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(df)
                pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df.index)

                if final_cluster_labels is not None:
                    pca_df["Cluster"] = final_cluster_labels.astype(str)
                    hue_col = "Cluster"
                else:
                    hue_col = None

                plt.figure(figsize=(10, 8))
                
                # Setup kwargs to conditionally include palette
                plot_args = {
                    "data": pca_df,
                    "x": "PC1",
                    "y": "PC2",
                    "hue": hue_col,
                    "s": 100,
                    "edgecolor": "black",
                    "alpha": 0.8,
                }
                
                # Only pass palette if we are actually coloring by cluster
                if hue_col:
                    plot_args["palette"] = "tab10"

                sns.scatterplot(**plot_args)

                plt.title(f"PCA (PCoA) - {args.title}")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

                if hue_col:
                    plt.legend(
                        bbox_to_anchor=(1.05, 1), loc="upper left", title="Cluster"
                    )

                plt.savefig(args.pca_out, bbox_inches="tight", dpi=args.dpi)
                logging.info(f"Saved PCA plot to {args.pca_out}")
                plt.close()
            except Exception as e:
                logging.error(f"Failed during PCA generation: {e}")

    # --- LEVEL 4: VISUALIZATION (HEATMAP) ---
    if args.no_plot:
        logging.info("Skipping heatmap generation (--no-plot requested).")
        # Ensure we still export the sorted matrix CSV if possible
        if linkage_matrix is not None:
            # We need to calculate the sort order manually if not plotting
            from scipy.cluster.hierarchy import leaves_list
            reordered_index = leaves_list(linkage_matrix)
            sorted_df = df.iloc[reordered_index, reordered_index]
            sorted_df.to_csv(args.csv)
            logging.info(f"Saved sorted matrix to {args.csv}")
        return

    # Logic: Annotations
    if args.no_annot:
        annot_data = False
    else:
        annot_data = df.map(
            lambda x: "10K+" if x >= 10000 else str(x) if pd.notnull(x) else ""
        )

    # Logic: Masking
    mask = None
    if args.hide_below is not None or args.hide_above is not None:
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        if args.hide_below is not None:
            mask = mask | (df < args.hide_below)
        if args.hide_above is not None:
            mask = mask | (df > args.hide_above)

    logging.info(f"Generating heatmap: {args.output}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ClusterWarning)

            if linkage_matrix is not None:
                # Use the PRE-CALCULATED linkage
                # Note: We pass the same linkage for row and col since it's a symmetric matrix
                g = sns.clustermap(
                    df,
                    annot=annot_data,
                    row_linkage=linkage_matrix,
                    col_linkage=linkage_matrix,
                    fmt="",
                    cmap=args.cmap,
                    linewidths=0,
                    figsize=fig_size,
                    cbar_kws={"label": "Value"},
                    xticklabels=True,
                    yticklabels=True,
                    tree_kws={"linewidths": 1.5},
                    vmin=args.vmin,
                    vmax=args.vmax,
                    mask=mask,
                    annot_kws={"size": font_size},
                )
            else:
                # Simple Sorting (No clustering)
                logging.info("Using simple sorting (no clustering)...")
                df_sorted = df.loc[df.sum(axis=1).sort_values().index]
                if df_sorted.shape[0] == df_sorted.shape[1]:
                    df_sorted = df_sorted.reindex(columns=df_sorted.index)
                
                # Manual heatmap creation since clustermap always clusters or requires linkage
                fig, ax = plt.subplots(figsize=fig_size)
                g = sns.heatmap(
                    df_sorted,
                    annot=annot_data,
                    fmt="",
                    cbar_kws={"fraction": 0.01},
                    cmap=args.cmap,
                    linewidths=0,
                    ax=ax,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    mask=mask,
                    annot_kws={"size": font_size},
                )
                ax.set_title(args.title, fontsize=label_size + 4)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=label_size)
                plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=label_size)
                plt.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
                
                df_sorted.to_csv(args.csv)
                logging.info(f"Saved sorted matrix to {args.csv}")
                return

            # Continue Clustermap formatting
            if not args.dendrogram:
                g.ax_row_dendrogram.set_visible(False)
                g.ax_col_dendrogram.set_visible(False)
                g.ax_heatmap.set_frame_on(False)

            g.ax_heatmap.set_xticklabels(
                g.ax_heatmap.get_xmajorticklabels(),
                fontsize=label_size,
                rotation=45,
                ha="right",
            )
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_ymajorticklabels(), fontsize=label_size, rotation=0
            )
            g.fig.suptitle(args.title, fontsize=label_size + 4, y=1.02)

            plt.savefig(args.output, bbox_inches="tight", dpi=args.dpi)

            # Save Sorted Matrix (based on the plot's final order)
            if linkage_matrix is not None:
                reordered_index = g.dendrogram_row.reordered_ind
                sorted_df = df.iloc[reordered_index, reordered_index]
                sorted_df.to_csv(args.csv)
                logging.info(f"Saved sorted matrix to {args.csv}")

    except Exception as e:
        logging.error(f"Failed to create heatmap: {e}")
        sys.exit(1)
    finally:
        plt.close("all")


def main():
    parser = get_parser()
    args = parser.parse_args()

    try:
        plt.get_cmap(args.cmap)
    except ValueError:
        logging.error(f"Colormap '{args.cmap}' is not recognized.")
        sys.exit(1)

    input_file = Path(args.input)
    if not input_file.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    output_path = Path(args.output)
    valid_exts = [".png", ".pdf", ".svg", ".jpg", ".jpeg"]
    if output_path.suffix.lower() not in valid_exts:
        logging.error(f"Invalid output format '{output_path.suffix}'.")
        sys.exit(1)

    logging.info(f"Parsing and formatting {args.input}")
    df = parse_files(args.input, args.format)

    logging.info(f"Processing {len(df.columns)} samples")
    font_tuple = determine_figsize(
        len(df.columns), args.width, args.height, args.font_scale
    )

    run_analysis(df, args, font_tuple)
    logging.info("Done")


if __name__ == "__main__":
    main()
