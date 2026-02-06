import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from heatcluster.cli import main
from heatcluster.parse_files import parse_files

def run_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["heatcluster"] + args)
    main()

def test_auto_k_correctness(monkeypatch, tmp_path):
    """
    Create a synthetic matrix with 2 VERY distinct groups.
    Verify that --auto-k correctly identifies K=2.
    """
    # 1. Create Synthetic Data: Group A (Samples 0-2) and Group B (Samples 3-5)
    # Distance within groups = 0 or 1
    # Distance between groups = 100
    data = [
        [0, 1, 1, 100, 100, 100],
        [1, 0, 1, 100, 100, 100],
        [1, 1, 0, 100, 100, 100],
        [100, 100, 100, 0, 1, 1],
        [100, 100, 100, 1, 0, 1],
        [100, 100, 100, 1, 1, 0],
    ]
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    df = pd.DataFrame(data, index=names, columns=names)
    
    # Save to file
    matrix_file = tmp_path / "synthetic_k2.csv"
    df.to_csv(matrix_file) # Pandas saves with index by default

    # 2. Run with --auto-k
    cluster_out = tmp_path / "clusters.csv"
    args = [
        "-i", str(matrix_file),
        "-f", "snp-dists", # Standard CSV format
        "--auto-k",
        "-l", str(cluster_out),
        "--no-plot"
    ]
    
    run_cli(monkeypatch, args)

    # 3. Verify Result
    res_df = pd.read_csv(cluster_out)
    
    # Get cluster assignments for Group A and Group B
    group_a_cluster = res_df[res_df["Sample"].str.startswith("A")]["Cluster_ID"].iloc[0]
    group_b_cluster = res_df[res_df["Sample"].str.startswith("B")]["Cluster_ID"].iloc[0]

    # They must be different clusters
    assert group_a_cluster != group_b_cluster
    # Total unique clusters should be exactly 2
    assert len(res_df["Cluster_ID"].unique()) == 2


def test_parser_identity_inversion(tmp_path):
    """
    UNIT TEST: Verify that Identity matrices (100% match = 100) 
    are correctly inverted to Distance matrices (100% match = 0).
    """
    # 1. Create a dummy FastANI file (Identity format)
    # A vs A = 100% (Identity) -> Should become 0 (Distance)
    # A vs B = 95%  (Identity) -> Should become 5 (Distance)
    ani_file = tmp_path / "fastani.txt"
    with open(ani_file, "w") as f:
        f.write("A\tA\t100.0\t100\t100\n")
        f.write("A\tB\t95.0\t100\t100\n")
        f.write("B\tA\t95.0\t100\t100\n")
        f.write("B\tB\t100.0\t100\t100\n")

    # 2. Call the parsing function directly (bypassing CLI)
    df = parse_files(str(ani_file), "fastani")

    # 3. Check Math
    assert df.loc["A", "A"] == 0.0, "Diagonal should be 0.0"
    assert df.loc["A", "B"] == 5.0, "95.0 Identity should be 5.0 Distance"


def test_auto_k_insufficient_samples(monkeypatch, tmp_path):
    """
    Test edge case: --auto-k requires at least 3 samples to be meaningful.
    If we provide only 2, it should fallback gracefully (usually to K=2) 
    without crashing.
    """
    # Create 2x2 matrix
    matrix_file = tmp_path / "tiny.csv"
    with open(matrix_file, "w") as f:
        f.write("names,A,B\n")
        f.write("A,0,10\n")
        f.write("B,10,0\n")

    args = [
        "-i", str(matrix_file),
        "--auto-k",
        "--no-plot"
    ]

    # Should run without error (exit code 0)
    try:
        run_cli(monkeypatch, args)
    except SystemExit as e:
        assert e.code == 0

def test_dimensions_arguments(monkeypatch, tmp_path):
    """
    Ensure custom width/height/font-scale arguments are respected 
    and don't cause matplotlib errors.
    """
    # Use existing small matrix
    matrix_file = Path("test/small_matrix.csv")
    out_png = tmp_path / "custom_dims.png"

    args = [
        "-i", str(matrix_file),
        "-o", str(out_png),
        "--width", "20",
        "--height", "5",
        "--font-scale", "2.5"
    ]

    run_cli(monkeypatch, args)
    assert out_png.exists()