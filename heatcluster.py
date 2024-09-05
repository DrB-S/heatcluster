##########################################
# heatcluster-v1.2.5.20240904.py          #
# - equivalent to heatcluster-v1.0.2g.py  #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heatmaps                    #
# from SNP matrices                       #
# Outputs sorted csv SNP matrix           #
###########################################

import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input SNP matrix file name', default='snp-dists.txt')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extension for final image', default='pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '1.2.5.20240904')
args = parser.parse_args()

def main(args):
    SNPmatrix = args.input
    logging.info(f'Creating figure for {SNPmatrix}')

    df = read_snp_matrix(SNPmatrix)

    if len(df.index) > len(df.columns):
        logging.debug('This matrix has been melted. Sorry for your loss!')
        exit(0)

    df = clean_and_read_df(df)
    df, fontSize, labelSize, figsize, labels = determine_heatmap_size(df, SNPmatrix)
    create_heatmap(df, fontSize, labelSize, figsize, labels, args)

    logging.info("Done")

def read_snp_matrix(file):
    try:
        df = pd.read_csv(file, sep=None, engine='python')
        logging.debug(f'Read SNP matrix with shape {df.shape}')
    except Exception as e:
        logging.error(f'Error reading SNP matrix: {e}')
        exit(1)
    return df

def clean_and_read_df(df):
    df = df.iloc[:, 1:]
    df.columns = df.columns.str.replace(r'(snp-dists 0.8.2|\.consensus_threshold_0.6_quality_20|Consensus_|Unnamed: 0)', '', regex=True)
    df = df.set_index(df.columns)
    return df

def determine_heatmap_size(df, SNPmatrix):
    numSamples = len(df.columns)
    logging.info(f'Found {numSamples} samples in {SNPmatrix}')

    if numSamples <= 3:
        logging.fatal('This matrix must have 4+ samples. Sorry!')
        exit(0)

    fontSize, labelSize, figsize = determine_font_and_size(numSamples)

    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df.sum(axis=1).sort_values().index]

    if df.shape[0] == df.shape[1]:
        df = df.reindex(columns=df.index)

    labels = df.where(df != 10000, '10K+')
    labels.to_csv('sorted_matrix.csv', index=True, encoding='utf-8')

    return df, fontSize, labelSize, figsize, labels

def determine_font_and_size(numSamples):
    if numSamples >= 120:
        return 1, 2, (18, 15)
    elif numSamples >= 80:
        return 2, 3, (18, 15)
    elif numSamples >= 40:
        return 3, 4, (11, 8)
    elif numSamples >= 30:
        return 4, 6, (11, 8)
    else:
        return 6, 7, (11, 8)

def create_heatmap(df, fontSize, labelSize, figsize, labels, args):
    fig, ax = plt.subplots(figsize=figsize)
    logging.debug('Creating heatmap')

    df_display = df.map(lambda x: '10K+' if x == 10000 else x)

    sns.heatmap(df, annot=df_display, fmt='', cbar_kws={'fraction': 0.01}, cmap='Reds_r', linewidths=0.001, ax=ax)

    plt.title('SNP Matrix Heatmap', fontsize='x-large')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=labelSize)
    plt.setp(ax.get_yticklabels(), rotation='horizontal', ha='right', fontsize=labelSize)

    outfile = f"{args.out}.{args.type}"
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main(args)
