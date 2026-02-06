import pytest
import pandas as pd
import numpy as np
from heatcluster.parse_files import read_vcf_matrix

def test_vcf_haploid_enforcement(tmp_path):
    """
    UNIT TEST: Verify that '0/1' (heterozygous) is forced to '1' (variant)
    and multi-allelic sites are skipped.
    """
    # 1. Create a fake VCF with tricky edge cases
    vcf_content = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSampleA\tSampleB",
        # Site 1: Simple SNP (A vs A) -> Dist 0
        "1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0\t0", 
        # Site 2: SampleB is '0/1' (Dipoid). Your logic should count this as '1'.
        # So SampleA(0) vs SampleB(1) -> Dist 1
        "1\t101\t.\tA\tT\t.\tPASS\t.\tGT\t0\t0/1",
        # Site 3: Multi-allelic (ALT=G,T). Your logic should SKIP this entire line.
        "1\t102\t.\tA\tG,T\t.\tPASS\t.\tGT\t0\t1",
    ]
    
    vcf_file = tmp_path / "test.vcf"
    with open(vcf_file, "w") as f:
        f.write("\n".join(vcf_content))

    # 2. Run just the parser (not the whole CLI)
    df = read_vcf_matrix(str(vcf_file))

    # 3. verify the math
    # We expect distance to be 1. 
    # (Site 1 is match, Site 2 is mismatch, Site 3 is skipped).
    dist = df.loc["SampleA", "SampleB"]
    
    assert dist == 1.0, f"Expected distance 1.0 (1 mismatch), got {dist}"