#!/usr/bin/python3

###########################################
# HeatCluster-0.4.10                      #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heat/cluster maps           #
# from SNP matrices                       #
###########################################

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required = True, type = str, help ='input SNP matrix')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extension for final image', default = 'pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '0.4.10')
args = parser.parse_args()

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

    # Convert column names to strings
    df.columns = df.columns.map(str)
    
    # Define consensus patterns
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_', 'Unnamed: 0']
    
    # Replace consensus patterns in column names
    df.columns = df.columns.str.replace('|'.join(consensus_patterns), '', regex=True)
    
    # Setting the index
    df = df.set_index(df.columns)

    return df

def main():
    """
    Creates image for SNP matrix.
    """

    SNPmatrix=args.input
    logging.info('Creating figure for ' + SNPmatrix)

    df = read_snp_matrix(SNPmatrix)
    logging.debug('The input SNP matrix:')
    logging.debug(df)

    if len(df.index) > len(df.columns):
        logging.fatal('This matrix has been melted. Sorry!')
        exit(0)

    df = clean_and_read_df(df)
    logging.debug('The clean SNP matrix:')
    logging.debug(df)

    numSamples = len(df.columns)
    logging.info('Found ' + str(numSamples) + ' samples in ' + SNPmatrix)

    if numSamples <= 3:
        logging.fatal('This matrix has too few samples. Sorry!')
        exit(0)

    # Set output figure size tuple based on number of samples
    if (numSamples) >= 140:
        fontSize = 2
    elif (numSamples) >=100:
        fontSize = 4
    elif (numSamples) >=60:
        fontSize = 6
    else:
        fontSize=8  

    logging.debug('The fontSize will be ' + str(fontSize))
    
    df = df.loc[df.sum(axis=1).sort_values(ascending=True).index]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()


    heatmap = sns.clustermap(
        df,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=80,
        center=20,
        annot=True,
        annot_kws={'size': fontSize},
        cbar_kws={"orientation": "vertical", "pad": 0.5},
        cmap='Reds_r',
        linecolor="white",
        linewidths=.1,
        fmt='d',
        col_cluster=False, 
        row_cluster=False
    )
    plt.setp(heatmap.ax_heatmap.get_xticklabels(), rotation=45, ha='right',fontsize=fontSize)
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation='horizontal', fontsize=fontSize)
    
    plt.title('SNP matrix visualized via HeatCluster')
        
    heatmap.ax_row_dendrogram.set_visible(False)
    heatmap.ax_col_dendrogram.set_visible(False)

    heatmap.savefig('SNP_matrix.pdf')

    plt.show()
    print("Done")

if __name__ == "__main__":
    main()
