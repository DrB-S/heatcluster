#!/usr/bin/python3

##########################################
# HeatCluster-0.4.9                       #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heat/cluster maps           #
# from SNP matrices                       #
###########################################

import argparse
import fastcluster as sch 
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%b-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required = True, type = str, help ='input SNP matrix')
parser.add_argument('-o', '--out', type=str, help='final file name', default='SNP_matrix')
parser.add_argument('-t', '--type', type=str, help='file extention for final image', default = 'pdf')
parser.add_argument('-v', '--version', help='print version and exit', action='version', version='%(prog)s ' + '0.4.9')
args = parser.parse_args()

def read_snp_matrix(file):
    logging.debug('Determining if file is comma or tab delimited')
    tabs   = pd.read_csv(file, nrows=1, sep='\t').shape[1]
    commas = pd.read_csv(file, nrows=1, sep=',').shape[1]
    if tabs > commas:
        logging.debug('The file is probaby tab-delimited')
        df = pd.read_csv(file, sep='\t', index_col= False)
    else:
        logging.debug('The file is probaby comma-delimited')
        df = pd.read_csv(file, sep=',', index_col= False)

    return df

def clean_and_read_df(df):
    """
    Clean and read DataFrame from lines.
    
    Returns:
        df (DataFrame): Cleaned DataFrame.
    """
    logging.debug('Dropping the first column')
    df = df.iloc[: , 1:]

    # Convert column names to strings
    df.columns = df.columns.map(str)
    
    # Define consensus patterns
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_']
    
    # Replace consensus patterns in column names
    df.columns = df.columns.str.replace('|'.join(consensus_patterns), '', regex=True)
    
    # Setting the index
    df = df.set_index(df.columns)

    return df

def compute_clusters(data, method='complete'):
    clusters = sch.linkage(data.values, method=method, metric='euclidean')
    logging.debug('The clusters are ' + str(clusters))
    return clusters

def calculate_silhouette_scores(data, upper_range):
    silhouette_scores = []
    for n_clusters in range(2, upper_range):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(data.values)
        silhouette_avg = silhouette_score(data.values, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    logging.debug('The silhouette scores are ' + str(silhouette_scores))
    return silhouette_scores

def main():
    SNPmatrix=args.input
    logging.info('Creating figure for ' + SNPmatrix)

    df = read_snp_matrix(SNPmatrix)
    logging.debug('The input SNP matrix:')
    logging.debug(df)

    df = clean_and_read_df(df)
    logging.debug('The clean SNP matrix:')
    logging.debug(df)

    numSamples = len(df.columns)
    logging.info('Found ' + str(numSamples) + ' samples in ' + SNPmatrix)

    if numSamples <= 3:
        logging.fatal('This matrix has too few samples or has been melted. Sorry!')
        exit(0)

    # Set output figure size tuple based on number of samples
    if numSamples <= 20:
        figureSize = (10, 8)
    elif numSamples <= 40:
        figureSize = (20, 16)
    elif numSamples <= 60:
        figureSize = (30, 24)
    else:
        figureSize = (40, 32)

    logging.debug('The figure size will be ' + str(figureSize))
    
    df = df.loc[df.sum(axis=1).sort_values(ascending=True).index]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    #
    # what is this used for? The clusters variable isn't used anywhere
    clusters = compute_clusters(df)
    #

    clustergrid = sns.clustermap(
        df,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=50,
        center=20,
        annot=True,
        annot_kws={'size': 6},
        cbar_kws={'pad': 0.5},
        cmap='Reds_r',
        linecolor='white',
        linewidths=.2,
        fmt='d',
        dendrogram_ratio=0.1,
        col_cluster=True,
        row_cluster=True,
        figsize=figureSize
    )
    plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation='horizontal')
    clustergrid.ax_row_dendrogram.set_visible(False)
    clustergrid.ax_col_dendrogram.set_visible(False)

    row_order = clustergrid.dendrogram_row.reordered_ind
    col_order = clustergrid.dendrogram_col.reordered_ind

    logging.debug('Use the reordered indices to sort the DataFrame')
    sorted_df = df.iloc[row_order, col_order]

    df = sorted_df.apply(lambda row: row[row < 500].sum(), axis=1)
    sorted_df['Within_Cluster_SNPs'] = df.values

    silhouette_scores = calculate_silhouette_scores(sorted_df, min(numSamples, 11))
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    kmeans = KMeans(n_clusters=optimal_num_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(sorted_df.values)
    sorted_df['Cluster'] = cluster_labels

    sorted_df = sorted_df.sort_values(by=['Cluster', 'Within_Cluster_SNPs'], ascending=[True, True], kind='mergesort')
    sorted_df = sorted_df.drop(['Cluster', 'Within_Cluster_SNPs'], axis='columns')
    sorted_df = sorted_df.reindex(columns=sorted_df.index)

    heatmap = sns.clustermap(
        sorted_df, 
        xticklabels=True, 
        yticklabels=True, 
        vmin=0, 
        vmax=50, 
        center=20,
        annot=True, 
        annot_kws={'size': 6}, 
        cbar_kws={'orientation': 'vertical', 'pad': 0.5},
        cmap='Reds_r', 
        linecolor='white', 
        linewidths=.1, 
        fmt='d', 
        dendrogram_ratio=0.1,
        col_cluster=True, 
        row_cluster=True,
    )

    logging.debug('Set orientation of axes labels')
    plt.setp(heatmap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation='horizontal')

    plt.title('SNP matrix visualized via HeatCluster')

    heatmap.ax_row_dendrogram.set_visible(False)
    heatmap.ax_col_dendrogram.set_visible(False)

    heatmap.savefig(args.out + '.' + args.type)

    plt.show()
    logging.info('Done')

if __name__ == '__main__':
    main()
