#!/usr/bin/python3

###########################################
# heatcluster-1.2.0.20240515_a            #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heatmaps                    #
# from SNP matrices                       #
# Outputs sorted csv SNP matrix           #
# Uses Polars instead of Pandas           #
###########################################

import argparse
import logging
import polars as pl
import numpy as np
import seaborn_polars as snl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input SNP matrix file name', default='snp-dists.txt')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extension for final image', default='pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '1.0.2c')
args = parser.parse_args()

def main(args):
    SNPmatrix = args.input
    logging.info(f'Creating figure for {SNPmatrix}')

    df = read_snp_matrix(SNPmatrix)
    logging.debug('The input SNP matrix:')
    logging.debug(df)

    #if len(df.columns) > len(df.rows):
    #    print('This matrix has been melted. Sorry for your loss!')
    #    exit(0)
        
    df = clean_and_read_df(df)
    logging.debug('The clean SNP matrix:')
    logging.debug(df)

    (df, fontSize, labelSize, figsize, labels) = determine_heatmap_size(df, SNPmatrix)

    create_heatmap(df, fontSize, labelSize, figsize, labels)
    logging.info("Done")

def read_snp_matrix(file):
    logging.debug('Determining if file is comma or tab delimited')
    tabs   = pl.scan_csv(file, n_rows=1, separator='\t')
    commas = pl.scan_csv(file, n_rows=1, separator=',')
    if len(tabs.columns) > len(commas.columns):
        logging.debug('The file is tab-delimited')
        df = pl.read_csv(file, separator='\t')
    else:
        logging.debug('The file is comma-delimited')
        df = pl.read_csv(file, separator=',')
    return df

def clean_and_read_df(df):
    logging.debug('Dropping the first column')
    df = df.drop("0")

    #df = df.with_columns(df.columns().map(str))
    
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_', 'Unnamed: 0']
    
    #df = df.select([pl.col(col).replace(consensus_patterns, '') for col in df.columns])
    
    #df = df.set_index("0")

    return df

def determine_heatmap_size(df, SNPmatrix):
    numSamples = df.shape[1]
    logging.info(f'Found {numSamples} samples in {SNPmatrix}')
    if numSamples <= 3:
        logging.fatal('This matrix must have 4+ samples. Sorry!')
        exit(0)

    if numSamples >= 120:
        fontSize = 1
        labelSize = 2
        figsize = (18, 15)
    elif numSamples >= 80:
        fontSize = 2
        labelSize = 3    
        figsize = (18, 15)
    elif numSamples >= 40:
        fontSize = 3
        labelSize = 4    
        figsize = (11, 8)
    elif numSamples >= 30:
        fontSize = 4 
        labelSize = 6    
        figsize = (11, 8)
    else:
        fontSize = 6    
        labelSize = 7    
        figsize = (11, 8)
    
    logging.debug(f'The fontSize will be {fontSize}')
    
    logging.debug('Sorting dataframe and removing empty rows/columns')
    #df = df.sort("0").dropna()
    ##df = df.sort("0").collect()
    ##df = df.drop_nulls().collect()

    # Apply the function to each column in the DataFrame
    for col in df.columns:
        labels = df.with_columns(df[col].apply(replace_large_values).alias(col))

    """
    Save sorted csv SNP matrix
    """

    labels.to_csv('sorted_matrix.csv', index=True, encoding='utf-8')

    return (df, fontSize, labelSize, figsize, labels)

def create_heatmap(df, fontSize, labelSize, figsize, labels):
    fig,ax = plt.subplots(figsize=figsize)
    logging.debug('Creating heatmap')
    
    heatmap = snl.heatmap(
        df,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=60,
        center=25,
        annot=False,
        cbar_kws={'fraction': 0.01},
        cmap='Reds_r',
        linecolor="white",
        linewidths=0.001
    )
    
    colors = heatmap.get_children()[0].get_array()

    for i in range(len(df.rows)):
        for j in range(len(df.columns)):
            val = labels.iloc[i, j]
            color_intensity = heatmap.get_children()[0].get_facecolor()[i * len(df.columns) + j].ravel() 
            font_color = 'white' if np.mean(color_intensity) < 0.7 else 'black'
            ax.text(j + 0.5, i + 0.5, val, ha='center', va='center', fontsize=fontSize, color=font_color)

    plt.setp(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=labelSize)
    plt.setp(heatmap.get_yticklabels(), rotation='horizontal', ha='right', fontsize=labelSize)

    plt.title('SNP Matrix Heatmap', fontsize='x-large')

    outfile = (args.out + "." + args.type)
    print("\tOutput file is ", outfile)
    plt.savefig(outfile, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main(args)
