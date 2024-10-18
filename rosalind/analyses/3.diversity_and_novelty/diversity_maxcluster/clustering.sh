

#!/bin/bash

#https://github.com/jasonkyuyim/se3_diffusion/issues/35
# Check if directory path is provided as argument
if [ $# -ne 2 ]; then
    echo "Usage: $0 <directory_path> <prefix>"
    exit 1
fi

# Directory path
directory="$1"
# Check if directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory not found"
    exit 1
fi

prefix="$2"

cd /scratch/alexb2/generative_protein_models/analyses/3.diversity_and_novelty/diversity_maxcluster
mkdir -p "$prefix"
# Find all .pdb files in the directory and subdirectories and write their paths to the list file
#find "$directory" -maxdepth 1 -type f -name "$prefix*.pdb" > "./$prefix/$prefix.list"
find "$directory" -maxdepth 1 -type f -name "$prefix*.pdb" > "./$prefix/$prefix.list"
#find "$directory" -maxdepth 1 -type f -name "*.pdb" > "./$prefix/$prefix.list"
/scratch/alexb2/generative_protein_models/tools/maxcluster/maxcluster64bit -l "./$prefix/$prefix.list" -C 2 -in -Rl ./$prefix/tm_results.txt -Tm 0.5 >  ./$prefix/output.txt

#extract clusters from output.txt
awk '/INFO  : Item     Cluster/, /INFO  : ======================================/ {print}' ./$prefix/output.txt | awk 'NR>1 && NR!=1{lines[NR]=$0} END{for(i=2;i<=NR-1;i++) print lines[i]}' > "./$prefix/clusters.tsv"
awk '{print $3 "\t" $5 "\t" $6 }' "./$prefix/clusters.tsv" > ./$prefix/temp.tsv && mv ./$prefix/temp.tsv ./$prefix/clusters.tsv
echo -e "item\tcluster\tpath" > temp_file
cat ./$prefix/clusters.tsv >> temp_file
mv temp_file "./$prefix/clusters.tsv"

#extract centroids from output.txt
awk '/INFO  : Cluster  Centroid  Size        Spread/, /INFO  : ======================================/ {print}' ./$prefix/output.txt | awk 'NR>1 && NR!=1{lines[NR]=$0} END{for(i=2;i<=NR-1;i++) print lines[i]}' > "./$prefix/centroids.tsv"
awk '{print $3 "\t" $5 "\t" $6 "\t" $7 "\t" $8 }' "./$prefix/centroids.tsv" > ./$prefix/temp.tsv && mv ./$prefix/temp.tsv ./$prefix/centroids.tsv
echo -e "cluster\tcentroid\tsize\tspread\tpath" > temp_file
cat ./$prefix/centroids.tsv >> temp_file
mv temp_file "./$prefix/centroids.tsv"

#extract heirachical tree from output.txt
awk '/INFO  : Node     Item 1   Item 2      Distance/, /INFO  : ======================================/ {print}' ./$prefix/output.txt | awk 'NR>1 && NR!=1{lines[NR]=$0} END{for(i=2;i<=NR-1;i++) print lines[i]}' > "./$prefix/tree.tsv"
awk '{print $3 "\t" $5 "\t" $6 "\t" $7 "\t" $8 }' "./$prefix/tree.tsv" > ./$prefix/temp.tsv && mv ./$prefix/temp.tsv ./$prefix/tree.tsv
echo -e "node\titem_1\titem_2\tdistance\tpath" > temp_file
cat ./$prefix/tree.tsv >> temp_file
mv temp_file "./$prefix/tree.tsv"






