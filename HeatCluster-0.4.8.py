#!/usr/bin/python3

##########################################
# HeatCluster-0.4.8                       #
# written by Stephen Beckstrom-Sternberg  #
# Creates SNP heat/cluster maps           #
# from SNP matrices                       #
###########################################

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import fastcluster as sch 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import StringIO
from pathlib import Path

def read_snp_matrix(path):
    try:
        path.resolve(strict=True)
    except FileNotFoundError:
        path = Path('./snp_matrix.txt')
    print("Using file path:", path)

    with open(path, "r") as infile:
        lines = infile.readlines()
    
    return lines

def clean_and_read_df(lines):
    """
    Clean and read DataFrame from lines.
    
    Args:
        lines (list): List of strings representing lines of data.
        
    Returns:
        df (DataFrame): Cleaned DataFrame.
    """
    # Join lines into a single string
    snp_matrix_string = "".join(lines)
    
    # Read DataFrame from string
    df = pd.read_csv(StringIO(snp_matrix_string), sep=',')
    
    # Convert column names to strings
    df.columns = df.columns.map(str)
    
    # Define consensus patterns
    consensus_patterns = ['snp-dists 0.8.2', '.consensus_threshold_0.6_quality_20', 'Consensus_']
    
    # Replace consensus patterns in column names
    df.columns = df.columns.str.replace('|'.join(consensus_patterns), '', regex=True)
    # Replace consensus patterns in entire dataframe to change row names
    df = df.replace(consensus_patterns, '', regex=True) 

    # Keep only numeric columns
    df = df.set_index(df.columns[0])
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)  
    return df

def compute_clusters(data, method='complete'):
    clusters = sch.linkage(data.values, method=method, metric='euclidean')
    return clusters

def calculate_silhouette_scores(data, upper_range):
    silhouette_scores = []
    for n_clusters in range(2, upper_range):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(data.values)
        silhouette_avg = silhouette_score(data.values, cluster_labels)
        silhouette_scores.append(silhouette_avg)
#        print(silhouette_scores,"\n") 
    return silhouette_scores

def main():
    path = Path('./snp-dists.txt')
    lines = read_snp_matrix(path)
    numSamples = len(lines) - 1

    df = clean_and_read_df(lines)

    # Set output figure size tuple based on number of samples
    if numSamples <= 20:
        figureSize = (10, 8)
    elif numSamples <= 40:
        figureSize = (20, 16)
    elif numSamples <= 60:
        figureSize = (30, 24)
    else:
        figureSize = (40, 32)
    
    df = df.loc[df.sum(axis=1).sort_values(ascending=True).index]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    clusters = compute_clusters(df)
    clustergrid = sns.clustermap(
        df,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=50,
        center=20,
        annot=True,
        annot_kws={'size': 6},
        cbar_kws={"pad": 0.5},
        cmap='Reds_r',
        linecolor="white",
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
    clustergrid.ax_col_dendrogram.set_visible(True)

    row_order = clustergrid.dendrogram_row.reordered_ind
    col_order = clustergrid.dendrogram_col.reordered_ind

# Use the reordered indices to sort the DataFrame
    sorted_df = df.iloc[row_order, col_order]

    df = sorted_df.apply(lambda row: row[row < 500].sum(), axis=1)
    sorted_df['Within_Cluster_SNPs'] = df.values

    silhouette_scores = calculate_silhouette_scores(sorted_df, min(numSamples, 11))
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    kmeans = KMeans(n_clusters=optimal_num_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(sorted_df.values)
    sorted_df['Cluster'] = cluster_labels

    sorted_df = sorted_df.sort_values(by=['Cluster', 'Within_Cluster_SNPs'], ascending=[True, True], kind="mergesort")
    print("sorted_df\n",sorted_df,"\n\n")
    sorted_df.drop(['Cluster', 'Within_Cluster_SNPs'], axis='columns', inplace=True)
    sorted_df = sorted_df.reindex(columns=sorted_df.index)

    heatmap = sns.clustermap(
        sorted_df, xticklabels=True, yticklabels=True, vmin=0, vmax=50, center=20,
        annot=True, annot_kws={'size': 6}, cbar_kws={"orientation": "vertical", "pad": 0.5},
        cmap='Reds_r', linecolor="white", linewidths=.1, fmt='d', dendrogram_ratio=0.1,
        col_cluster=True, row_cluster=True,
    )
# Set orientation of axes labels
    plt.setp(heatmap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation='horizontal')
    
    heatmap.ax_row_dendrogram.set_visible(False)
    heatmap.ax_col_dendrogram.set_visible(True)

    heatmap.savefig('SNP_matrix.pdf')
    heatmap.savefig('SNP_matrix.png')

    plt.show()
    print("Done")

if __name__ == "__main__":
    main()
