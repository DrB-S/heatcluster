# HeatCluster SNP Matrix Visualization

Welcome to HeatCluster SNP Matrix Visualization, a Python tool for visualizing SNP (Single Nucleotide Polymorphism) matrices generated via [SNP-dists](https://github.com/tseemann/snp-dists) using heatmaps. 

## Introduction

HeatCluster SNP Matrix Visualization is designed to provide an easy and effective way to visualize SNP matrices. By generating heatmaps from these SNP matrices, it can be easier to identify clusters.

## Getting Started

### Getting the script

Clone the HeatCluster SNP Matrix Visualization repository to your local machine:

```bash
git clone https://github.com/DrB-S/HeatCluster.git
```

`HeatCluster.py` can be found in the HeatCluster directory that was just created.

### Getting the dependencies
HeatCluster requires
- python3
  - pandas 
  - numpy 
  - scipy
  - fastcluster
  - scikit-learn
  - seaborn
  - matplotlib

```bash
pip install pandas numpy scipy fastcluster scikit-learn seaborn matplotlib
```

## Running HeatCluster.py

```
usage: HeatCluster.py [-h] -i INPUT [-o OUT] [-t TYPE] [-v]

options:
  -h, --help                show this help message and exit
  -i INPUT, --input INPUT   input SNP matrix
  -o OUT, --out OUT         final file name (default = 'SNP_matrix')
  -t TYPE, --type TYPE      file extention for final image (default = 'pdf')
  -v, --version             print version and exit
```

There are multiple test files located in [test](./test) that can be used for trouble shooting.

Example:
```
./HeatCluster.py -i test/snp_matrix.txt -o test
```

This will generate a file called 'test.pdf'.

Examples:
```
./HeatCluster.py -i test/snp_matrix.txt -o test -t png
```

![alt text](assets/test.png)

```
./HeatCluster.py -i test/small_matrix.csv -o small_test -t png
```
![alt text](assets/small_test.png)

```
./HeatCluster.py -i test/med_matrix.txt -o med_test -t png
```
![alt text](assets/med_test.png)

## Limitations

Right now most outputs from snp-dists are supported with the exception of getting a molten or melted output (created with the `-m` option of snp-dists).
