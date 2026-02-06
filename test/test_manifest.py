import csv
import sys
import pytest
from pathlib import Path
from heatcluster.cli import main

# Define the path to your manifest file
# This assumes the test file and manifest.csv are in the same 'test/' directory
MANIFEST_PATH = Path(__file__).parent / "manifest.csv"

def load_test_cases():
    """
    Reads the CSV manifest and returns a list of rows.
    If the file doesn't exist, returns empty list (skips tests).
    """
    if not MANIFEST_PATH.exists():
        return []
    
    cases = []
    with open(MANIFEST_PATH, "r") as f:
        # csv.DictReader maps the header row to keys: 'fmt', 'test_file', 'test_name'
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace just in case
            clean_row = {k: v.strip() for k, v in row.items()}
            cases.append(clean_row)
    return cases

@pytest.mark.parametrize("case", load_test_cases())
def test_formats_from_manifest(case, tmp_path, monkeypatch):
    """
    Dynamically tests every row in the manifest.csv file.
    """
    fmt = case["fmt"]
    input_rel_path = case["test_file"]
    prefix = case["test_name"]

    # 1. Resolve Input Path
    # Assumes pytest is run from the project root
    input_path = Path(input_rel_path)
    
    if not input_path.exists():
        pytest.skip(f"Skipping {fmt}: Input file '{input_path}' not found.")

    # 2. Define Output Paths
    # Use tmp_path (pytest fixture) to avoid cluttering with test artifacts
    output_png = tmp_path / f"{prefix}.png"
    output_csv = tmp_path / f"{prefix}_sorted.csv"
    output_clusters = tmp_path / f"{prefix}_clusters.csv"

    # 3. Construct Command Line Arguments
    # Mimics: heatcluster -i <file> -f <fmt> -o <png> -c <csv> -l <clusters> --auto-k
    argv = [
        "heatcluster",
        "-i", str(input_path),
        "-f", fmt,
        "-o", str(output_png),
        "-c", str(output_csv),
        "-l", str(output_clusters),
        "--auto-k"  # Force clustering to ensure that logic is tested too
    ]

    # 4. Mock sys.argv so 'main' reads our arguments
    monkeypatch.setattr(sys, "argv", argv)

    # 5. Run HeatCluster
    print(f"\nTesting {fmt} with {input_path}...")
    try:
        main()
    except SystemExit as e:
        # main() calls sys.exit(1) on failure, check that we didn't get that.
        if e.code != 0 and e.code is not None:
            pytest.fail(f"HeatCluster crashed with exit code {e.code}")

    # 6. Assertions (Verify files were created)
    assert output_png.exists(), f"Output PNG was not created for {fmt}"
    assert output_csv.exists(), f"Output CSV was not created for {fmt}"
    assert output_clusters.exists(), f"Cluster CSV was not created for {fmt}"