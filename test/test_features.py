import sys
import pytest
import pandas as pd
from pathlib import Path
from heatcluster.cli import main

# Path to a known good file for general testing
# Assumes pytest from the root directory
SMALL_MATRIX = Path("test/small_matrix.csv")

def run_cli(monkeypatch, args):
    """Helper to run the CLI with specific arguments."""
    monkeypatch.setattr(sys, "argv", ["heatcluster"] + args)
    main()

def test_version(monkeypatch, capsys):
    """Test that the version flag works and prints something."""
    monkeypatch.setattr(sys, "argv", ["heatcluster", "--version"])
    
    # Expect SystemExit because argparse exits after printing version
    with pytest.raises(SystemExit):
        main()
    
    captured = capsys.readouterr()
    assert "heatcluster" in captured.out or "heatcluster" in captured.err

def test_invalid_input_file(monkeypatch):
    """Test that the program exits gracefully if input file is missing."""
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, ["-i", "non_existent_ghost_file.csv"])
    assert e.value.code == 1

def test_no_plot_mode(monkeypatch, tmp_path):
    """
    Test the --no-plot flag. 
    Should produce a sorted CSV matrix but NOT a PNG image.
    """
    out_csv = tmp_path / "sorted.csv"
    out_png = tmp_path / "heatmap.png"

    args = [
        "-i", str(SMALL_MATRIX),
        "-o", str(out_png),
        "-c", str(out_csv),
        "--no-plot"
    ]
    
    run_cli(monkeypatch, args)

    assert out_csv.exists(), "Sorted CSV should be created even in no-plot mode"
    assert not out_png.exists(), "PNG should NOT be created in no-plot mode"

def test_pca_generation(monkeypatch, tmp_path):
    """Test that the --pca flag successfully generates a second image."""
    out_png = tmp_path / "heatmap.png"
    out_pca = tmp_path / "pca_plot.png"

    args = [
        "-i", str(SMALL_MATRIX),
        "-o", str(out_png),
        "--pca",
        "--pca-out", str(out_pca)
    ]

    run_cli(monkeypatch, args)

    assert out_png.exists()
    assert out_pca.exists(), "PCA plot file was not created"

def test_cluster_k_enforcement(monkeypatch, tmp_path):
    """
    Test that forcing --cluster-k produces a cluster CSV 
    with the correct structure.
    """
    cluster_csv = tmp_path / "clusters.csv"
    
    args = [
        "-i", str(SMALL_MATRIX),
        "--cluster-k", "2",
        "-l", str(cluster_csv),
        "--no-plot" # faster
    ]

    run_cli(monkeypatch, args)

    assert cluster_csv.exists()
    
    # Verify content logic
    df = pd.read_csv(cluster_csv)
    assert "Cluster_ID" in df.columns
    assert "Sample" in df.columns
    # With K=2, we expect exactly 2 unique values in Cluster_ID
    assert len(df["Cluster_ID"].unique()) == 2

def test_bad_diagonal_integrity(monkeypatch, tmp_path):
    """
    Test that the tool rejects matrices where the diagonal is not zero 
    (i.e., sample A vs sample A != 0).
    """
    # 1. Create a bad matrix
    bad_matrix_file = tmp_path / "bad_matrix.csv"
    with open(bad_matrix_file, "w") as f:
        f.write("snp-dists,A,B\n")
        f.write("A,100,5\n") # Error: A vs A is 100, should be 0
        f.write("B,5,0\n")

    # 2. Run CLI and expect failure
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, ["-i", str(bad_matrix_file)])
    
    assert e.value.code == 1

def test_masking_does_not_crash(monkeypatch, tmp_path):
    """Test that --hide-below/above flags run without crashing."""
    out_png = tmp_path / "masked.png"
    
    args = [
        "-i", str(SMALL_MATRIX),
        "-o", str(out_png),
        "--hide-below", "10",
        "--hide-above", "1000"
    ]
    
    # Just asserting it doesn't crash (exit code 0)
    try:
        run_cli(monkeypatch, args)
    except SystemExit as e:
        assert e.code == 0

    assert out_png.exists()