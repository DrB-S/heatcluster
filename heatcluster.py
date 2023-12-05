#!/usr/bin/python3

###########################################
# heatcluster-1.0.0                      #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heat/cluster maps           #
# from SNP matrices                       #
# - adjusts font, label and figure sizes  #
# based on matrix size                    #
###########################################

import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)

"""
Parse command-line arguments or use defaults
    
    Image types: eps, pdf (default), png, svg, tiff
"""        
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input SNP matrix file name', default='snp-dists.txt')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extension for final image', default = 'pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '1.0.0')
args = parser.parse_args()

def main(args):
    """
    Creates image for SNP matrix
    """

    SNPmatrix = args.input
    logging.info('Creating figure for ' + SNPmatrix)

    df = read_snp_matrix(SNPmatrix)
    logging.debug('The input SNP matrix:')
    logging.debug(df.to_string())

    if len(df.index) > len(df.columns):
        print('This matrix has been melted. Sorry for your loss!')
        exit(0)
        
    df = clean_and_read_df(df)
    logging.debug('The clean SNP matrix:')
    logging.debug(df.to_string())

    (df, fontSize, labelSize, figsize, labels) = determine_heatmap_size(df, SNPmatrix)

    create_heatmap(df, fontSize, labelSize, figsize, labels)
    logging.info("Done")
def read_snp_matrix(file):
    """
    Reads the SNP matrix into a pandas dataframe.
    
    Args:
        file (str): SNP dist output file that should be converted to pandas dataframe
        
    Returns:
        df (DataFrame): Pandas dataframe of SNP matrix.
    """
    logging.debug('Determining if file is comma or tab delimited')
    tabs   = pd.read_csv(file, nrows=1, sep='\t').shape[1]
    commas = pd.read_csv(file, nrows=1, sep=',').shape[1]
    if tabs > commas:
        logging.debug('The file is probably tab-delimited')
        df = pd.read_csv(file, sep='\t', index_col= False)
    else:
        logging.debug('The file is probably comma-delimited')
        df = pd.read_csv(file, sep=',', index_col= False)
        
    return df

def clean_and_read_df(df):
    """
    Clean and read DataFrame from lines.
    
    Args:
        lines (list): List of strings representing lines of data.
        
    Returns:
        df (DataFrame): Cleaned DataFrame.
    """
    logging.debug('Dropping the first column')
    df = df.iloc[: , 1:]

    """
    Convert column names to strings
    """
    df.columns = df.columns.map(str)
    
    """
    Define consensus patterns
    """
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_', 'Unnamed: 0']
    
    """
    Replace consensus patterns in column names
    """
    df.columns = df.columns.str.replace('|'.join(consensus_patterns), '', regex=True)

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
    logging.info('Found ' + str(numSamples) + ' samples in ' + SNPmatrix)
    if numSamples <= 3:
        logging.fatal('This matrix must have 4+ samples. Sorry!')
        exit(0)

    """
    Set output figure size tuple based on number of samples
    """
    if (numSamples) >=120:
        fontSize = 1
        labelSize = 2
        figsize = (18, 15)
    elif (numSamples) >=80:
        fontSize = 2
        labelSize = 3    
        figsize = (18, 15)
    elif (numSamples) >=40:
        fontSize = 3
        labelSize = 4    
        figsize = (11, 8)
    elif (numSamples) >=30:
        fontSize = 4 
        labelSize = 6    
        figsize = (11, 8)
    else:
        fontSize= 6    
        labelSize = 7    
        figsize = (11, 8)
    
    logging.debug('The fontSize will be ' + str(fontSize))
    
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
    Replace 1000's with K
    """

    labels = df.applymap(lambda v: v/1000." K" if v > 1000 else v)
    
    return (df, fontSize, labelSize, figsize, labels)

def create_heatmap(df, fontSize, labelSize, figsize, labels):
    """
    Create heatmap
    """
    fig, ax = plt.subplots(figsize=figsize)
    logging.debug('Creating heatmap')
    heatmap = sns.heatmap(
        df,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=60,
        center=25,
        #annot=True,
        annot=labels,
        annot_kws={'size': fontSize},
        cbar_kws={'fraction': 0.01},
        cmap='Reds_r',
        linecolor="white",
        linewidth=0.1,
        fmt='d'
    )
    
    """
    Set orientation and size of axes labels
    """
    plt.setp(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=labelSize)
    plt.setp(heatmap.get_yticklabels(), rotation='horizontal', ha='right', fontsize=labelSize)

    plt.title('SNP Matrix Heatmap', fontsize='x-large')

    SNP_matrix = args.out
    outfile = (args.out + "." + args.type)
    print("\tOutput file is ", outfile)
    plt.savefig(outfile)

    plt.show()

if __name__ == "__main__":
    main(args)
