#!/usr/bin/python3

###########################################
# heatcluster-1.1.0.20240131              #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heatmaps                    #
# from SNP matrices                       #
# Outputs sorted csv SNP matrix           #
# Uses Polars instead of Pandas           #
###########################################

import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import polars as pl
import seaborn_polars as snl
import pyarrow


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)

"""
Parse command-line arguments or use defaults
    
    Image types: eps, pdf (default), png, svg, tiff
"""        
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input SNP matrix file name', default='snp-dists.txt')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extension for final image', default='pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '1.0.2c')
args = parser.parse_args()

def main(args):
    """
    Creates image for SNP matrix
    """

    SNPmatrix = args.input
    logging.info(f'Creating figure for {SNPmatrix}')

    df = read_snp_matrix(SNPmatrix)
    logging.debug('The input SNP matrix:')
    #logging.debug(df.to_string())

    #if len(df.index) > len(df.columns):
    #    print('This matrix has been melted. Sorry for your loss!')
    #    exit(0)
        
    df = clean_and_read_df(df)
    logging.debug('Cleaning SNP matrix')
    #logging.debug(df.to_string())

    (df, fontSize, labelSize, figsize, labels) = determine_heatmap_size(df, SNPmatrix)

    create_heatmap(df, fontSize, labelSize, figsize, labels)
    logging.info("Done")

def read_snp_matrix(file):
    """
    Reads the SNP matrix into a pandas dataframe.
    
    Args:
        file (str): SNP dist output file that should be converted to pandas dataframe
        
    Returns:
        df (DataFrame): Polars dataframe of SNP matrix.
    """
    logging.debug('Determining if file is comma or tab delimited')
    tabs   = pl.scan_csv(file, n_rows=1, separator='\t')
    commas = pl.scan_csv(file, n_rows=1, separator=',')
    if len(tabs.columns) > len(commas.columns):
        logging.debug('The file is tab-delimited: tabs')
        df = pl.scan_csv(file, separator='\t', row_index_name=None, has_header=True)
    else:
        logging.debug('The file is comma-delimited: commas')
        df = pl.scan_csv(file, separator=',', row_index_name=None, has_header=True)
    return df

def clean_and_read_df(df):
    """
    Clean and read DataFrame from lines.
    
    Args:
        lines (list): List of strings representing lines of data.
        
    Returns:
        df (DataFrame): Cleaned DataFrame.
    """
    #logging.debug('Dropping the first column') # taken care of in lines 79 and 82
    #df = df[:, 1:] # changed from df.iloc[:, 1:]
    

    """
    Convert column names to strings
    """
    #df.columns = df.columns.map(str) #df.columns.map doesn't work in Polars, so see it line 112 will work without it
    
    """
    Define consensus patterns
    """
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_', 'Unnamed: 0']
    
    """
    Replace consensus patterns in column names
    """
    #df.columns = df.columns.str.replace('|'.join(consensus_patterns), '', regex=True)
    #df.columns = list(map(lambda x: x.replace('|'.join(consensus_patterns), '', regex=True), df.columns)) 

    """
    Setting the index
    """
    df = df.set_index(df.columns)

    return df

def determine_heatmap_size(df, SNPmatrix):
    """
    Determine size of heatmap
    """
    numSamples = len(df.columns)
    logging.info(f'Found {numSamples} samples in {SNPmatrix}')
    if numSamples <= 3:
        logging.fatal('This matrix must have 4+ samples. Sorry!')
        exit(0)

    """
    Set output figure size tuple based on number of samples
    """
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
    
    """
    Sort dataframe and remove empty rows/columns
    """
    logging.debug('Sorting dataframe and removing empty rows/columns')
    df = df.loc[df.sum(axis=1).sort_values(ascending=True).index]
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna()

    """
    Reindex columns
    """
    df = df.reindex(columns=df.index)

    """
    Replace 10000+ with 10K
    """

    labels = df.map(lambda v: '10K+' if v > 10000 else v)
    """
    Save sorted csv SNP matrix
    """

    labels.to_csv('sorted_matrix.csv', index=True, encoding='utf-8')

    return (df, fontSize, labelSize, figsize, labels)

def create_heatmap(df, fontSize, labelSize, figsize, labels):
    """
    Create heatmap
    """
    fig,ax = plt.subplots(figsize=figsize)
    logging.debug('Creating heatmap')
    
    #heatmap = sns.heatmap(
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
    ''' Access the color information from the heatmap'''
    colors = heatmap.get_children()[0].get_array()

    ''' Manually set annotations using text parameter with font color'''
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = labels.iloc[i, j]
            '''Assuming cmap is grayscale'''
            color_intensity = heatmap.get_children()[0].get_facecolor()[i * len(df.columns) + j].ravel() 
            font_color = 'white' if np.mean(color_intensity) < 0.7 else 'black'
            ax.text(j + 0.5, i + 0.5, val, ha='center', va='center', fontsize=fontSize, color=font_color)

    ''' Set orientation and size of axes labels'''
    plt.setp(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=labelSize)
    plt.setp(heatmap.get_yticklabels(), rotation='horizontal', ha='right', fontsize=labelSize)

    plt.title('SNP Matrix Heatmap', fontsize='x-large')

    SNP_matrix = args.out
    outfile = (args.out + "." + args.type)
    print("\tOutput file is ", outfile)
    plt.savefig(outfile, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main(args)
