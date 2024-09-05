# HeatCluster SNP Matrix Visualization

Welcome to HeatCluster SNP Matrix Visualization, a Python tool for visualizing SNP (Single Nucleotide Polymorphism) matrices generated via [SNP-dists](https://github.com/tseemann/snp-dists) as heatmaps. 

## Introduction

HeatCluster SNP Matrix Visualization is designed to provide an easy and effective way to visualize SNP matrices. By generating heatmaps from these SNP matrices, it can be easier to identify clusters.

## Getting Started

### Getting the script

Clone the heatcluster SNP Matrix Visualization repository to your local machine:

```bash
git clone https://github.com/DrB-S/heatcluster.git
```

`heatcluster.py` can be found in the heatcluster directory which was just created.

### Getting the dependencies
heatcluster requires
- python3
  - argparse
  - logging
  - pandas 
  - numpy 
  - seaborn
  - matplotlib
  - pathlib

```bash
pip3 install argparse logging pandas numpy seaborn matplotlib pathlib
```

## Running heatcluster.py

```
usage: python heatcluster.py [-h] -i INPUT [-o OUT] [-t TYPE] [-v]

requires: python3.12+

options:
  -h, --help                show this help message and exit
  -i INPUT, --input INPUT   input SNP matrix file name
  -o OUT, --out OUT         final file name (default = 'SNP_matrix')
  -t TYPE, --type TYPE      file extention for final image (default = 'pdf')
  -v, --version             print version and exit
```

Multiple test files are located in [test](./test) which can be used for troubleshooting.

Examples:
```
python heatcluster.py -i test/small_matrix.csv -o small_matrix -t png
```

This will generate a file called 'small_matrix.png'.

```
python heatcluster.py -i test/large_matrix.csv -o large_matrix
```

This will generate a file called 'large_matrix.pdf' from a comma-delimited matrix.

## Limitations

Currently most outputs from snp-dists are supported with the exception of a molten or melted output (created with the `-m` option of snp-dists).
