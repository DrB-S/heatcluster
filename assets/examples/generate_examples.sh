heatcluster -i test/med_matrix.txt -o assets/examples/basic.png
heatcluster -i test/med_matrix.txt -o assets/examples/basic_vmax2000.png --vmax 200  --title "Setting a vmax of 200"
heatcluster -i test/med_matrix.txt -o assets/examples/basic_10.png --cluster-t 10 --title "Setting a threshold of 10"
heatcluster -i test/large_matrix.csv -o assets/examples/basic_auto_k.png --auto-k --title "Using --auto-k"
heatcluster -i test/large_matrix.csv -o assets/examples/basic_3_k.png --cluster-k 3 --title "Using --cluster-k 3"
heatcluster -i test/med_matrix.txt --pca --pca-out assets/examples/basic_pca.png
heatcluster -i test/poppunk_db.dists.npy --format poppunk --cmap magma --no-annot --title "PopPUNK Core Distances" -o assets/examples/poppunk_viz.png 
heatcluster -i test/example.nwk --format nwk --cmap YlGnBu --title "Phylogenetic Tree Distances (Patristic)" -o assets/examples/phylo_tree.png
heatcluster -i test/large_matrix.csv --auto-k --title "Automated Lineage Discovery (Auto-K)" -o assets/examples/ml_analysis.png
heatcluster -i test/skani_example.txt --format skani --cmap viridis --no-annot --title "Skani Average Nucleotide Identity" -o assets/examples/ani_heatmap.png
heatcluster -i test/snp-dists.txt --format snp-dists --cmap RdYlBu_r --title "SNP Distance Clustering" -o assets/examples/snp_heatmap.png
