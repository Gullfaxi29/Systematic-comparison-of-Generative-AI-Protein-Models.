directories=("RITA" "RFdiffusion" "Protpardelle" "ProtGPT2" "ProteinSGM" "Protein-Generator" "ProGen2" "Genie" "FrameDiff" "FoldingDiff" "EvoDiff" "ESM-Design" "PISCES" "Chroma")

output_file="combined_clusters.tsv"
cd /scratch/alexb2/generative_protein_models/analyses/3.diversity_and_novelty/diversity_maxcluster

if [ -f "$output_file" ]; then
    rm "$output_file"
fi

for dir in "${directories[@]}"; do
    # Find all clusters.tsv files in the directory
    for file in $(find "$dir" -name "clusters.tsv" 2>/dev/null); do
        echo $file
        # Append the content of each file to the output file
        if [ -f "$output_file" ]; then
            # Skip header from subsequent files
            tail -n +2 "$file" >> "$output_file"
        else
            # Copy the header from the first file
            cat "$file" > "$output_file"
        fi
    done
done