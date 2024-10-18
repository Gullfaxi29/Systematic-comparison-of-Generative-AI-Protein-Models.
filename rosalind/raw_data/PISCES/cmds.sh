awk '{print substr($1, 1, 4)}' your_file.tsv | paste -sd "," -
awk 'NR>1 {print substr($1, 1, 4)}' your_file.tsv | paste -sd "," -


./pdb_batch_download.sh -f pisces_sample_list.csv -p 