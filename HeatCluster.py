#!/usr/bin/python3

##########################################
# HeatCluster-0.4.4                       #
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

# Read the SNP matrix file and determine the delimiter
path = Path('./snp-dists.txt')
try:
    path.resolve(strict=True)
except FileNotFoundError:
    path = Path('./snp_matrix.txt')

print("Using file path:", path)

# Read the SNP matrix file into a DataFrame
with open(path, "r") as infile:
    lines = infile.readlines()  # Do NOT Skip the header line

numSamples = len(lines) - 1  # counts data lines

# Combine the cleaned lines into a single string instead of a file
snp_matrix_string = "\n".join(line for line in lines)

print("\n\n", snp_matrix_string, "\n\n")

# Read the tab-delimited string into a DataFrame
df = pd.read_csv(StringIO(snp_matrix_string), sep=',')


# Remove 'snp-dists 0.8.2,' from the first column name
# Define the regular expression patterns for replacements
consensus_pattern = r'snp-dists 0\.8\.2,|\.consensus_threshold_0\.6_quality_20|Consensus_'

# Replace the header row using the pattern
df.columns = df.columns.str.replace(consensus_pattern, '', regex=True)

# Replace the data rows using the same pattern
df = df.replace(consensus_pattern, '', regex=True)

df = df.set_index(df.columns[0])

print("/n/n", df, "\n\n")
# Define colormap for heatmap
cmap = 'Reds_r'
print("Colormap set to '", cmap, "'")

# Vectorized operation to add 'Total_SNPs' column
df['Total_SNPs'] = df.sum(axis=1)

# Vectorized operation to sort based on 'Total_SNPs'
df = df.sort_values(by='Total_SNPs', kind='mergesort', axis=0)

# Reorder the columns to mirror the row order
sorted_cluster_matrix = df.reindex(columns=df.index)

# Check for Missing Values
if sorted_cluster_matrix.isnull().sum().sum() > 0:
    sorted_cluster_matrix = sorted_cluster_matrix.dropna()

# Check for Non-Finite Values
if ~np.isfinite(sorted_cluster_matrix).all().all():
    sorted_cluster_matrix = sorted_cluster_matrix.replace([np.inf, -np.inf], np.nan).dropna()

# Change output figure size tuple based on number of samples
if numSamples <= 20:
    figureSize = (10, 8)
elif numSamples <= 40:
    figureSize = (20, 16)
elif numSamples <= 60:
    figureSize = (30, 24)
else:
    figureSize = (40, 32)
print("Number of samples:", numSamples, "\nFigure size:", figureSize)

# Compute clusters
# Compute clusters using fastcluster
#clusters = fastcluster.linkage(sorted_cluster_matrix.values, method='complete', metric='euclidean')
clusters = sch.linkage(sorted_cluster_matrix.values, method='complete', metric='euclidean')

# Create clustermap to get the order of rows and columns based on clustering
clustergrid = sns.clustermap(
    sorted_cluster_matrix,
    xticklabels=True,
    vmin=0,
    vmax=50,
    center=20,
    annot=True,
    annot_kws={'size': 6},
    cbar_kws={"pad": 0.5},
    cmap=cmap,
    linecolor="white",
    linewidths=.2,
    fmt='d',
    dendrogram_ratio=0.1,
    col_cluster=True,
    row_cluster=True,
    figsize=figureSize
)
plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

# Suppress printing of dendrogram along the y-axis
clustergrid.ax_row_dendrogram.set_visible(False)
clustergrid.ax_col_dendrogram.set_visible(True)

row_order = clustergrid.dendrogram_row.reordered_ind
col_order = row_order

# Sort the DataFrame based on the cluster order
sorted_df = sorted_cluster_matrix.iloc[row_order, col_order]

# Vectorized operation to calculate 'within_cluster_snps'
within_cluster_snps = (df[df < 500]).sum(axis=1)

# Computing the number of SNPs within the cluster per row
within_cluster_snps = sorted_df.apply(lambda row: row[row < 500].sum(), axis=1)

# Add 'Within_Cluster_SNPs' column to the sorted DataFrame
sorted_df['Within_Cluster_SNPs'] = within_cluster_snps.values

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
upper_range = min(numSamples, 11)
for n_clusters in range(2, upper_range):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(sorted_df.values)
    silhouette_avg = silhouette_score(sorted_df.values, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters with the highest silhouette score
optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Use the optimal number of clusters to assign cluster labels and sort the DataFrame
kmeans = KMeans(n_clusters=optimal_num_clusters, n_init=10)
cluster_labels = kmeans.fit_predict(sorted_df.values)
sorted_df['Cluster'] = cluster_labels

# Sort the DataFrame first by 'Cluster' and then by 'Within_Cluster_SNPs'
sorted_df = sorted_df.sort_values(by=['Cluster', 'Within_Cluster_SNPs'], ascending=[True, True], kind="mergesort")

# Drop 'Cluster' and 'Within_Cluster_SNPs' columns
sorted_df = sorted_df.drop(['Cluster', 'Within_Cluster_SNPs'], axis='columns')
sorted_df = sorted_df.reindex(columns=sorted_df.index)

# Save the finally sorted tab-delimited SNP matrix
sorted_df.to_csv('Final_snp_matrix.tsv', sep='\t', header=True, index=True)

# Create the reordered heatmap with correct values
heatmap = sns.clustermap(
    sorted_df, xticklabels=True, yticklabels=True, vmin=0, vmax=50, center=20,
    annot=True, annot_kws={'size': 6}, cbar_kws={"orientation": "vertical", "pad": 0.5},
    cmap=cmap, linecolor="white", linewidths=.1, fmt='d', dendrogram_ratio=0.1,
    col_cluster=True, row_cluster=True, figsize=figureSize
)
plt.setp(heatmap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

heatmap.ax_row_dendrogram.set_visible(False)

# Save the reordered heatmap as PDF and PNG
heatmap.savefig('SNP_matrix.pdf')
heatmap.savefig('SNP_matrix.png')

plt.show()
plt.close()
print("Done")
