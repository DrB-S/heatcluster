import gzip
import logging
import sys
from pathlib import Path
import pickle
from scipy.spatial.distance import squareform, pdist
from Bio import Phylo

import pandas as pd
import numpy as np


def melted_to_pivot(df):
    df = df.pivot_table(index=0, columns=1, values=2, aggfunc="first")
    df = df.combine_first(df.T)
    return df


def check_diagonal_integrity(df: pd.DataFrame, tolerance: float = 1e-5):
    """
    Verifies that the diagonal (self-vs-self) is approximately 0.
    Returns True if valid, False if invalid (and logs warnings).
    """
    # 1. Extract the diagonal values
    diagonal_values = np.diag(df)

    # 2. Check if they are close to 0
    # Using tolerance because 100.0 - 99.999999 could be something like 1e-9, not absolute 0
    is_zero = np.abs(diagonal_values) <= tolerance

    if np.all(is_zero):
        logging.info("Data Integrity Check Passed: All self-comparisons are ~0.0")
        return True

    # 3. If check fails, identify the culprits
    failed_indices = np.where(~is_zero)[0]
    failed_samples = df.index[failed_indices]
    failed_values = diagonal_values[failed_indices]

    logging.warning(
        f"Data Integrity Warning: {len(failed_samples)} samples have non-zero self-distances."
    )

    # Print the first few failures to help the user debug
    for i in range(min(5, len(failed_samples))):
        sample = failed_samples[i]
        val = failed_values[i]
        logging.warning(f"  - {sample}: distance to self is {val:.4f} (Expected 0.0)")

    return False


def read_fastani_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: FastANI")
    try:
        df = pd.read_csv(file_path, sep="\t", header=None)

        if df.shape[1] < 3:
            logging.error(
                f"FastANI file has {df.shape[1]} columns. Expected at least 3."
            )
            sys.exit(1)
        logging.info("Converting FastANI list to square matrix...")
        df.iloc[:, 0] = df.iloc[:, 0].str.split("/").str[-1]
        df.iloc[:, 1] = df.iloc[:, 1].str.split("/").str[-1]
        df = melted_to_pivot(df)
        logging.info("Inverting FastANI values...")
        df = 100.0 - df
        df = df.fillna(100)
        return df
    except Exception as e:
        logging.error(f"Error reading FastANI matrix: {e}")
        sys.exit(1)


def read_mash_matrix(file_path: str) -> pd.DataFrame:
    # Query
    # Reference
    # Mash distance
    # P-value
    # Shared hashes / sketch size

    logging.info("Reading format: MASH")
    try:
        df = pd.read_csv(file_path, sep="\t", header=None)

        if df.shape[1] < 3:
            logging.error(f"MASH file has {df.shape[1]} columns. Expected at least 3.")
            sys.exit(1)
        logging.info("Converting MASH list to square matrix...")
        df.iloc[:, 0] = df.iloc[:, 0].str.split("/").str[-1]
        df.iloc[:, 1] = df.iloc[:, 1].str.split("/").str[-1]
        df = melted_to_pivot(df)
        df = df.fillna(df.values.max())
        return df
    except Exception as e:
        logging.error(f"Error reading MASH matrix: {e}")
        sys.exit(1)


def read_skani_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: Skani")
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, skiprows=1)
        if df.shape[1] < 3:
            logging.error(f"Skani file has {df.shape[1]} columns. Expected at least 3.")
            sys.exit(1)

        df = df.set_index(0)
        df.columns = df.index
        logging.info("Inverting Skani values...")
        df = 100.0 - df
        df = df.fillna(100)
        return df
    except Exception as e:
        logging.error(f"Error reading Skani matrix: {e}")
        sys.exit(1)


def read_snp_dists_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: Standard (snp-dists)")
    try:
        df = pd.read_csv(file_path, sep=None, engine="python", header="infer")
        if df.shape[1] > 1:
            df.set_index(df.columns[0], inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading snp-dists matrix: {e}")
        sys.exit(1)


def read_sourmash_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: Sourmash")
    try:
        df = pd.read_csv(file_path)

        # Check for empty or malformed files
        if df.shape[1] < 3:
            logging.error(f"Melted matrix must have at least 3 columns.")
            sys.exit(1)

        df.index = df.columns

        logging.info("Inverting Sourmash values...")
        df = 1.0 - df

        df = df.fillna(1.0)

        return df
    except Exception as e:
        logging.error(f"Error reading Sourmash matrix: {e}")
        sys.exit(1)


def read_poppunk_matrix(file_path: str, metric: str = "core") -> pd.DataFrame:
    """
    Parses PopPUNK binary output.
    file_path: Path to the .dists.npy file.
    metric: 'core' (similar to ANI) or 'accessory' (Jaccard-like).
    """
    logging.info("Reading format: PopPUNK (Numpy/Pickle)")

    path_obj = Path(file_path)

    # 1. Locate the corresponding .pkl file for labels
    # PopPUNK outputs usually look like: name.dists.npy and name.dists.pkl
    pkl_path = path_obj.with_suffix(".pkl")

    if not pkl_path.exists():
        # Sometimes the .dists part is part of the stem, sometimes not.
        # Try removing .dists.npy and replacing with .dists.pkl just in case
        pkl_path = Path(str(path_obj).replace(".npy", ".pkl"))

    if not pkl_path.exists():
        logging.error(f"Could not find companion pickle file: {pkl_path}")
        logging.error("PopPUNK requires both .npy (data) and .pkl (names) files.")
        sys.exit(1)

    try:
        # 2. Load Sample Names
        with open(pkl_path, "rb") as f:
            # PopPUNK pickle format: (rlist, qlist, self, ...)
            # For all-vs-all, rlist and qlist are identical.
            pkl_data = pickle.load(f)
            # The first element is usually the list of reference names
            names = pkl_data[0]

        # 3. Load Distances
        # Shape is (N_pairs, 2). Column 0 = Core, Column 1 = Accessory
        dists = np.load(file_path)

        # Select metric
        col_idx = 0 if metric == "core" else 1
        dist_values = dists[:, col_idx]

        # 4. Convert condensed distance array to square matrix
        # Note: PopPUNK stores lower-triangular distances.
        # squareform reconstructs the symmetric matrix (with 0 diagonal).
        matrix = squareform(dist_values)

        # 5. Create DataFrame
        df = pd.DataFrame(matrix, index=names, columns=names)

        df = df.fillna(df.values.max())

        return df

    except Exception as e:
        logging.error(f"Error reading PopPUNK matrix: {e}")
        sys.exit(1)


def read_aai_matrix(file_path: str) -> pd.DataFrame:
    """
    Strictly parses EzAAI summary output format:
    Label 1 | Label 2 | AAI | ...
    """
    logging.info("Reading format: AAI (EzAAI Summary)")
    try:
        # Read file assuming tab-separation
        df = pd.read_csv(file_path, sep="\t", engine="python")

        # Strict validation of required columns
        required_cols = ["Label 1", "Label 2", "AAI"]
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Input file is missing required columns: {required_cols}")
            logging.error(f"Found columns: {list(df.columns)}")
            sys.exit(1)

        # Extract only the relevant columns
        df = df[["Label 1", "Label 2", "AAI"]]

        # Rename to 0, 1, 2 for the helper function
        df.columns = [0, 1, 2]

        # Pivot to square matrix
        df = melted_to_pivot(df)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Convert Similarity to Distance
        # EzAAI output is 0-100% Similarity. Distance (0 on diagonal).
        logging.info("Inverting AAI values (100 - Similarity)...")
        df = 100.0 - df

        df = df.fillna(100.0)

        return df

    except Exception as e:
        logging.error(f"Error reading EzAAI matrix: {e}")
        sys.exit(1)


def read_nwk_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses a Newick tree file and converts it to a pairwise distance matrix.
    Requires: pip install biopython
    """
    logging.info("Reading format: Newick Tree")

    try:
        # Read the tree
        tree = Phylo.read(file_path, "newick")

        # Get all leaves (terminals)
        terminals = tree.get_terminals()
        names = [t.name for t in terminals]
        n = len(names)

        logging.info(f"Calculated distances for {n} leaves...")
        matrix = np.zeros((n, n))

        # Calculate pairwise distances
        # Note: tree.distance(t1, t2) sums the branch lengths between tips
        for i in range(n):
            for j in range(i + 1, n):
                d = tree.distance(terminals[i], terminals[j])
                matrix[i, j] = d
                matrix[j, i] = d

        df = pd.DataFrame(matrix, index=names, columns=names)
        return df

    except Exception as e:
        logging.error(f"Error reading Newick tree: {e}")
        sys.exit(1)


def read_kwip_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: KWIP")
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")

        df.set_index(df.columns[0], inplace=True)

        df = df.fillna(df.values.max())

        return df
    except Exception as e:
        logging.error(f"Error reading KWIP matrix: {e}")
        sys.exit(1)


def read_ska_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: SKA (distances)")
    try:
        # SKA output is tab-separated with headers
        df = pd.read_csv(file_path, sep="\t")

        # Ensure required columns exist
        required_cols = ["Sample 1", "Sample 2", "Mash-like distance"]
        if not all(col in df.columns for col in required_cols):
            logging.error(f"SKA file missing required columns. Found: {df.columns}")
            sys.exit(1)

        df["Sample 1"] = (
            df["Sample 1"].astype(str).str.replace(r"\.skf$", "", regex=True)
        )
        df["Sample 2"] = (
            df["Sample 2"].astype(str).str.replace(r"\.skf$", "", regex=True)
        )

        # ska only uniquely identifies differences
        df_fwd = df[["Sample 1", "Sample 2", "SNPs"]].copy()
        df_fwd.columns = [0, 1, 2]

        df_rev = df[["Sample 2", "Sample 1", "SNPs"]].copy()
        df_rev.columns = [0, 1, 2]

        df = pd.concat([df_fwd, df_rev], ignore_index=True)

        df = melted_to_pivot(df)

        # ska does not create distance to self
        # create copy for numpy writability
        data_matrix = df.to_numpy(dtype=float, copy=True)

        # Fill diagonal on the numpy array, not the dataframe
        np.fill_diagonal(data_matrix, 0.0)

        # Reconstruct DataFrame
        df = pd.DataFrame(data_matrix, index=df.index, columns=df.columns)

        df = df.fillna(df.values.max())

        return df

    except Exception as e:
        logging.error(f"Error reading SKA matrix: {e}")
        sys.exit(1)


def read_bindash_matrix(file_path: str) -> pd.DataFrame:

    logging.info("Reading format: Bindash")
    try:
        # Bindash typically has no header
        # Column 0: Query Sample (Path)
        # Column 1: Reference Sample (Path)
        # Column 2: Distance (e.g., 2.7644e-04)
        # Column 3: P-value
        # Column 4: Shared Sketches
        df = pd.read_csv(file_path, sep="\t", header=None)

        if df.shape[1] < 3:
            logging.error(
                f"Bindash file has {df.shape[1]} columns. Expected at least 3."
            )
            sys.exit(1)

        for col in [0, 1]:
            df[col] = (
                df[col]
                .astype(str)
                .str.split("/")
                .str[-1]
                .str.replace(r"\.fasta$", "", regex=True)
            )

        # BinDash performs unique pairwise comparisons
        df_fwd = df.iloc[:, [0, 1, 2]].copy()
        df_fwd.columns = [0, 1, 2]

        df_rev = df.iloc[:, [1, 0, 2]].copy()
        df_rev.columns = [0, 1, 2]

        df_combined = pd.concat([df_fwd, df_rev], ignore_index=True)

        df_matrix = melted_to_pivot(df_combined)

        data_matrix = df_matrix.to_numpy(dtype=float, copy=True)

        np.fill_diagonal(data_matrix, 0.0)

        df_final = pd.DataFrame(
            data_matrix, index=df_matrix.index, columns=df_matrix.columns
        )

        # Fill missing pairs (Max distance = 1.0)
        df_final = df_final.fillna(1.0)

        return df_final

    except Exception as e:
        logging.error(f"Error reading Bindash matrix: {e}")
        sys.exit(1)


def read_dashing_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: Dashing (Similarity)")
    try:
        # Dashing output uses whitespace alignment and hyphens for empty/diagonal cells
        # The first token is usually "##Names", which becomes the index name automatically
        df = pd.read_csv(file_path, sep=r"\s+", index_col=0)

        clean_regex = r"(^contigs/|\.fasta$)"
        df.index = df.index.astype(str).str.replace(clean_regex, "", regex=True)
        df.columns = df.columns.astype(str).str.replace(clean_regex, "", regex=True)

        # Replace '-' with NaN for numeric operations
        df = df.replace("-", np.nan)

        df = df.astype(float)

        # Dashing outputs Jaccard similarity (0-1)
        # Converting to distance
        df = 1.0 - df

        # The input is upper-triangular
        # Fill lower triangle using the transposed values from the upper triangle.
        df = df.fillna(df.T)

        # Dashing does not do calculations to self
        data_matrix = df.to_numpy(dtype=float, copy=True)

        np.fill_diagonal(data_matrix, 0.0)

        df_final = pd.DataFrame(data_matrix, index=df.index, columns=df.columns)

        df_final = df_final.fillna(1.0)

        return df_final

    except Exception as e:
        logging.error(f"Error reading Dashing matrix: {e}")
        sys.exit(1)


def read_pyani_identity_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses pyANI percentage identity matrix (0.0 - 1.0).
    Converts to distance (1.0 - Identity).
    """
    logging.info("Reading format: pyANI (Identity)")
    try:
        # PyANI matrices are standard tab-separated square matrices
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Remove .fasta, .fna, etc.
        clean_regex = r"(\.fasta|\.fa|\.fna)$"
        df.index = df.index.astype(str).str.replace(clean_regex, "", regex=True)
        df.columns = df.columns.astype(str).str.replace(clean_regex, "", regex=True)

        # Invert Identity to Distance
        # 1.0 (Identical) -> 0.0 (Distance)
        # 0.0 (No ID) -> 1.0 (Distance)
        df = 1.0 - df

        df = df.fillna(1.0)

        return df

    except Exception as e:
        logging.error(f"Error reading pyANI identity matrix: {e}")
        sys.exit(1)


def read_pyani_sim_errors_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses pyANI similarity errors matrix (Counts of mismatches/gaps).
    Treats these counts as distances directly.
    """
    logging.info("Reading format: pyANI (Similarity Errors)")
    try:
        # Standard tab-separated square matrix
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Clean Index/Column Names
        clean_regex = r"(\.fasta|\.fa|\.fna)$"
        df.index = df.index.astype(str).str.replace(clean_regex, "", regex=True)
        df.columns = df.columns.astype(str).str.replace(clean_regex, "", regex=True)

        # Fill with the maximum value found in the matrix to ensure separate clusters
        df = df.fillna(df.values.max())

        return df

    except Exception as e:
        logging.error(f"Error reading pyANI similarity errors matrix: {e}")
        sys.exit(1)


def read_iqtree_mldist_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses IQ-TREE .mldist files (Phylip format square matrix).
    Format:
      4
      Name1  0.0  0.1  0.2  0.3
      Name2  0.1  0.0  ...
    """
    logging.info("Reading format: IQ-TREE mldist")
    try:
        # Using sep=r"\s+" handles variable whitespace (tabs or spaces)
        df = pd.read_csv(
            file_path, sep=r"\s+", skiprows=1, header=None, index_col=0, engine="python"
        )

        # Since this is a symmetric matrix, columns match the row index
        df.columns = df.index.astype(str)
        df.index = df.index.astype(str)

        df = df.apply(pd.to_numeric, errors="coerce")

        df = df.fillna(df.values.max())

        return df

    except Exception as e:
        logging.error(f"Error reading IQ-TREE mldist matrix: {e}")
        sys.exit(1)


def read_gene_presence_absence_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses Roary/Panaroo/etc 'gene_presence_absence.Rtab' output.
    Calculates Jaccard Distance (1 - Jaccard Index).
    """
    logging.info("Reading format: Gene Presence/Absence (.Rtab)")

    try:
        # Rtab files are tab-separated
        # Rows = Genes, Columns = Samples
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # Convert Samples as Rows for distance calculation
        df = df.T

        # Panaroo sometimes creates 'fragmented' entries or similar,
        # but the Rtab should strictly be 0/1.
        # Force numeric just in case.
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Calculate Jaccard Distance
        # metric='jaccard' computes the proportion of discordant non-zero entries.
        # It handles the fact that "0 vs 0" (gene absent in both)
        # should NOT increase similarity.
        logging.info(f"Calculating Jaccard distances for {len(df)} samples...")

        # pdist returns a condensed distance matrix
        condensed_matrix = pdist(df.values, metric="jaccard")

        # Convert to square matrix
        square_matrix = squareform(condensed_matrix)

        # Reconstruct DataFrame
        df_dist = pd.DataFrame(square_matrix, index=df.index, columns=df.index)

        return df_dist

    except Exception as e:
        logging.error(f"Error processing gene presence/absence matrix: {e}")
        sys.exit(1)


def read_pathogen_detection_snp_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses NCBI Pathogen Detection SNP distance files.
    Columns used: biosample_acc_1, biosample_acc_2, compatible_distance.

    Naming Logic: Strictly uses 'biosample_acc' columns.
    Missing Pairs: Filled with the maximum value found in 'compatible_positions' column.
    """
    logging.info("Reading format: NCBI Pathogen Detection SNP (Biosample Accessions)")
    try:
        df = pd.read_csv(file_path, sep="\t")

        required_cols = [
            "biosample_acc_1",
            "biosample_acc_2",
            "compatible_distance",
            "compatible_positions",
        ]

        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing required columns. Found: {df.columns}")
            sys.exit(1)

        # 1. Determine the fill value for missing pairs
        fill_value = df["compatible_positions"].max()
        logging.info(
            f"Missing pairs will be filled with max compatible_positions: {fill_value}"
        )

        # 2. Extract strictly the Biosample Accessions and Distance
        df_clean = df[
            ["biosample_acc_1", "biosample_acc_2", "compatible_distance"]
        ].copy()
        df_clean.columns = [0, 1, 2]  # Rename for helper function

        # 3. Pivot to square matrix
        df_matrix = melted_to_pivot(df_clean)

        # 4. Finalize Matrix
        data_matrix = df_matrix.to_numpy(dtype=float, copy=True)

        # Self-to-self distance is 0
        np.fill_diagonal(data_matrix, 0.0)

        df_final = pd.DataFrame(
            data_matrix, index=df_matrix.index, columns=df_matrix.columns
        )

        # Fill missing values (unconnected pairs) with the calculated fill_value
        df_final = df_final.fillna(fill_value)

        return df_final

    except Exception as e:
        logging.error(f"Error reading Pathogen Detection SNP matrix: {e}")
        sys.exit(1)


def read_vcf_matrix(file_path: str) -> pd.DataFrame:
    """
    Parses VCF/VCF.gz files for HAPLOID organisms to create a SNP distance matrix.

    Method:
      1. Robust Text-Based Parsing: Ignores validation errors in non-essential fields (like ABQ).
      2. Biallelic Only: Skips multi-allelic sites.
      3. Haploid Enforcement:
         - Any presence of '1' in GT -> 1 (Variant)
         - Otherwise -> 0 (Reference/Missing)
         - This prevents '1/1' (diploid artifact) from counting as distance 2.
      4. Calculates Manhattan distance (equivalent to SNP count for binary data).
    """
    try:
        # handle gzip or plain text
        if file_path.endswith(".gz"):
            f = gzip.open(file_path, "rt", encoding="utf-8")
        else:
            f = open(file_path, "r", encoding="utf-8")

        samples = []
        dosage_rows = []
        valid_site_count = 0

        with f:
            for line in f:
                # Skip meta-information lines
                if line.startswith("##"):
                    continue

                # Parse Header
                if line.startswith("#CHROM"):
                    parts = line.strip().split("\t")
                    if len(parts) < 10:
                        logging.error("VCF header missing sample columns.")
                        sys.exit(1)
                    samples = parts[9:]
                    logging.info(f"Found {len(samples)} samples: {samples}")
                    continue

                # Parse Data Lines
                if not line.strip():
                    continue

                parts = line.strip().split("\t")

                # Safety check for column count
                if len(parts) < 9 + len(samples):
                    continue

                ref = parts[3]
                alt = parts[4]
                fmt_str = parts[8]

                # Filter: Skip non-biallelic sites (multi-allelic ALTs have commas)
                if "," in alt:
                    continue

                # Filter: Skip Indels (standard for SNP distance)
                if len(ref) > 1 or len(alt) > 1:
                    continue

                # Find GT index
                # FORMAT column looks like "GT:ABQ:DP" or "GT"
                fmt_keys = fmt_str.split(":")
                try:
                    gt_idx = fmt_keys.index("GT")
                except ValueError:
                    continue

                row = np.zeros(len(samples), dtype=np.int8)

                # Extract Genotypes
                sample_data = parts[9:]

                for i, s_str in enumerate(sample_data):
                    # Sample string: "0/1:.:10" or "1/1" or "0"
                    s_fields = s_str.split(":")

                    if len(s_fields) <= gt_idx:
                        dosage = 0  # Malformed/Missing
                    else:
                        gt_val = s_fields[gt_idx]

                        # Handle delimiters / or | (phased)
                        if "|" in gt_val:
                            alleles = gt_val.split("|")
                        else:
                            alleles = gt_val.split("/")

                        # HAPLOID ENFORCEMENT
                        # If the alternate allele '1' is present anywhere, mark as 1.
                        # This handles '1', '1/1', and even '0/1' as "Variant Present".
                        if "1" in alleles:
                            dosage = 1
                        else:
                            dosage = 0  # '0', '0/0', '.', './.'

                    row[i] = dosage

                dosage_rows.append(row)
                valid_site_count += 1

        if not dosage_rows:
            logging.error("No valid biallelic SNP sites found in VCF.")
            sys.exit(1)

        # Convert to numpy array
        genotype_matrix = np.array(dosage_rows, dtype=np.int8)
        genotype_matrix = genotype_matrix.T  # (Samples x Sites)

        logging.info(
            f"Calculating distances for {len(samples)} samples over {valid_site_count} sites..."
        )

        # Manhattan distance
        # For binary vectors (0,1), this sums the mismatches exactly.
        condensed_dist = pdist(genotype_matrix, metric="cityblock")
        square_dist = squareform(condensed_dist)

        df = pd.DataFrame(square_dist, index=samples, columns=samples)

        return df

    except Exception as e:
        logging.error(f"Error reading VCF matrix: {e}")
        sys.exit(1)


def read_melted_matrix(file_path: str) -> pd.DataFrame:
    logging.info("Reading format: Melted (Long-format)")
    try:
        df = pd.read_csv(file_path, sep=None, engine="python", header=None)
        if df.shape[1] < 3:
            logging.error(f"Melted matrix must have at least 3 columns.")
            sys.exit(1)
        logging.info("Converting melted matrix to square format...")
        df = melted_to_pivot(df)
        df = df.fillna(0)
        return df
    except Exception as e:
        logging.error(f"Error reading melted matrix: {e}")
        sys.exit(1)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_regex = r"(snp-dists.*|Consensus_|\.consensus_threshold.*|Unnamed: 0)"
    path_regex = r"(\.fasta|\.fa|\.fna|.*/)"

    for axis in [df.columns, df.index]:
        if hasattr(axis, "str"):
            new_axis = (
                axis.astype(str)
                .str.replace(clean_regex, "", regex=True)
                .str.replace(path_regex, "", regex=True)
            )
            if axis is df.columns:
                df.columns = new_axis
            else:
                df.index = new_axis
    return df


def process_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="all", inplace=True)
    return df


# TODO : Allele distances from MLST, CHEWBACCA, cgMLST

# def read_allel_matrix(file_path: str) -> pd.DataFrame:
#     """
#     Parses Allele Call Matrices (MLST, cgMLST, ChewBBACA).
#     Format: Tabular (Rows=Samples, Cols=Loci, Values=Allele IDs).
#     Metric: Hamming Distance (Count of different alleles).
#     """
#     logging.info("Reading format: Allele Matrix (MLST/cgMLST)")
#     try:

#         # 1. Read the table
#         # Most MLST tools use TSV
#         df = pd.read_csv(file_path, sep="\t", index_col=0, dtype=str)

#         # 2. Transpose if necessary?
#         # Standard format is Rows=Samples. If Cols=Samples, we'd need to detect that.
#         # Heuristic: Usually #Loci >> #Samples. If Shape is (3000, 10), it's correct.
#         # If Shape is (10, 3000), it might be inverted.
#         # For now, assume standard Rows=Samples.

#         logging.info(f"Calculating Allele differences for {len(df)} samples over {df.shape[1]} loci...")

#         # 3. Clean Data
#         # We need to treat "Missing" or "Novel" calls carefully.
#         # Common missing codes: "0", "-", "?", "INF", "LNF", "ASM"
#         # We will treat them as a unique "Missing" string so they don't match anything
#         # (or match each other, depending on philosophy. Usually Missing != Missing).
#         # However, for Hamming, strict string inequality is usually sufficient.

#         # 4. Calculate Hamming Distance
#         # pdist(metric='hamming') returns proportion (0-1).
#         # We want count (integer distance).
#         # Distance = Hamming_Proportion * N_Loci

#         X = df.to_numpy()

#         # Scipy's hamming handles strings/objects nicely
#         condensed_dist = pdist(X, metric='hamming')

#         # Convert proportion to absolute count
#         n_loci = df.shape[1]
#         condensed_dist = condensed_dist * n_loci

#         square_dist = squareform(condensed_dist)

#         return pd.DataFrame(square_dist, index=df.index, columns=df.index)

#     except Exception as e:
#         logging.error(f"Error reading Allele matrix: {e}")
#         sys.exit(1)


def parse_files(input_file, fmt):
    # --- Genomics / ANI ---
    if fmt == "skani":
        df = read_skani_matrix(input_file)
    elif fmt == "fastani":
        df = read_fastani_matrix(input_file)
    elif fmt == "mash":
        df = read_mash_matrix(input_file)
    elif fmt == "sourmash":
        df = read_sourmash_matrix(input_file)
    elif fmt == "poppunk":
        df = read_poppunk_matrix(input_file)
    elif fmt == "ezaai":
        df = read_aai_matrix(input_file)
    elif fmt == "pyani_identity":
        df = read_pyani_identity_matrix(input_file)
    elif fmt == "pyani_errors":
        df = read_pyani_sim_errors_matrix(input_file)

    # --- Phylogeny ---
    elif fmt == "nwk":
        df = read_nwk_matrix(input_file)
    elif fmt == "iqtree":
        df = read_iqtree_mldist_matrix(input_file)

    # --- K-mer / Sketches ---
    elif fmt == "kwip":
        df = read_kwip_matrix(input_file)
    elif fmt == "bindash":
        df = read_bindash_matrix(input_file)
    elif fmt == "dashing":
        df = read_dashing_matrix(input_file)

    # --- SNP / Variants ---
    elif fmt == "ska":
        df = read_ska_matrix(input_file)
    elif fmt == "pathogen_detection":
        df = read_pathogen_detection_snp_matrix(input_file)
    elif fmt == "vcf":
        df = read_vcf_matrix(input_file)

    # --- Gene Content / Alleles ---
    elif fmt == "gene_presence_absence":
        df = read_gene_presence_absence_matrix(input_file)
    # elif fmt == "mlst":
    #    df = read_allel_matrix(input_file)

    # --- Generic / Fallback ---
    elif fmt == "melted":
        df = read_melted_matrix(input_file)
    else:
        df = read_snp_dists_matrix(input_file)

    if not check_diagonal_integrity(df):
        sys.exit(1)

    df = clean_dataframe(df)
    df = process_matrix(df)

    if len(df.columns) < 2:
        logging.error("Matrix must have at least 2 samples.")
        sys.exit(1)

    return df
