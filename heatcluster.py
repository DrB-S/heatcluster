#!/usr/bin/python3

###########################################
# heatcluster-0.4.12                      #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heat/cluster maps           #
# from SNP matrices                       #
# - removed pathlib                       #
###########################################

import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input SNP matrix file name', default='snp-dists.txt')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extension for final image', default = 'pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '0.4.12')
args = parser.parse_args()

def main(args):
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

    (df, fontSize) = determine_heatmap_size(df, SNPmatrix)
    
    create_heatmap(df, fontSize)
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

    # Convert column names to strings
#    df.columns = df.row(0, named=True)
    df.columns = df.columns.map(str)
    
         # Define consensus patterns
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_', 'Unnamed: 0']
    
    # Replace consensus patterns in column names
    df.columns = df.columns.str.replace('|'.join(consensus_patterns), '', regex=True)

    # Setting the index
    df = df.set_index(df.columns)

    return df

def determine_heatmap_size(df, SNPmatrix):
    numSamples = len(df.columns)
    logging.info('Found ' + str(numSamples) + ' samples in ' + SNPmatrix)

    if numSamples <= 3:
        logging.fatal('This matrix must have 4+ samples. Sorry!')
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
    
    logging.debug('Sorting dataframe and removing empty rows/columns')
    df = df.loc[df.sum(axis=1).sort_values(ascending=True).index]
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna()

    df = df.reindex(columns=df.index)
    
    return (df, fontSize)

def create_heatmap(df, fontSize):
    logging.debug('Creating heatmap')
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
    
# Set orientation of axes labels
    plt.setp(heatmap.ax_heatmap.get_xticklabels(), rotation=45, ha='right',fontsize=fontSize)
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation='horizontal', fontsize=fontSize)

    plt.title('SNP matrix visualized via heatcluster')

    heatmap.ax_row_dendrogram.set_visible(False)
    heatmap.ax_col_dendrogram.set_visible(False)

    SNP_matrix = args.out
    outfile = (args.out + "." + args.type)
    print("\tOutput file is ", outfile)
    heatmap.savefig(outfile)

    plt.show()

if __name__ == "__main__":
    main(args)
