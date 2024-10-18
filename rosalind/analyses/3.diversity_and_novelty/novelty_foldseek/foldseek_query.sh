
check_status() {
    local id=$1
    local log_file=$2
    local output_file=$3
    local response
    # Loop until status is one of the terminal states
    while true; do
        # Extract the value of 'status' using string manipulation
        response=$(curl https://search.foldseek.com/api/ticket/$id) >> "$log_file" 2>&1
        status=$(echo "$response" | grep -o '"status":"[^"]*' | cut -d':' -f2 | tr -d '"')
        # Check if status is one of the terminal states
        case $status in
            "PENDING")
                echo "$id Status: PENDING" >> "$log_file"
                sleep 5
                ;;
            "RUNNING")
                echo "$id Status: RUNNING" >> "$log_file"
                sleep 5
                ;;
            "COMPLETE")
                echo "$id Status: COMPLETE" >> "$log_file"
                result=$(curl -X GET https://search.foldseek.com/api/result/$id/0)
                echo -n "$result" > $output_file
                echo "Result saved to $output_file ">> "$log_file"
                echo "..." >> "$log_file"
                echo "..." >> "$log_file"
                break
                ;;
            "ERROR")
                echo "$id Status: ERROR" >> "$log_file"
                break
                ;;
            "UNKNOWN")
                echo "$id Status: UNKNOWN" >> "$log_file"
                break
                ;;
            *)
                echo "$id Unknown status: $status" >> "$log_file"
                #This is often due to making to many queries so we'll add a longish pause here 
                sleep 45
                break
                ;;
        esac
    done
}

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_folder_path>"
    exit 1
fi



#input_dir="$1"

#model=$(basename $input_dir)
#timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
#log_file="logfile_${model}_${timestamp}.log"

input_dir="/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-150/whole_structures"
for length in {21..50}
do
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    for file in $input_dir/*len${length}_*.pdb
    do
        bn=$(basename "$file" .pdb)
        log_file="/scratch/alexb2/generative_protein_models/analyses/novelty/logs/${timestamp}_len${length}.log"
        echo $bn
        echo "Querying $file" >> "$log_file"
        output_file="/scratch/alexb2/generative_protein_models/analyses/novelty/results/${bn}_${timestamp}.json"
        #mkdir -p $(dirname "$output_file")
        if [ -f $output_file ]; then
            echo "Output file ($output_file) already exits for this structure. Skipping..." >> "$log_file" 
        else
            echo "Command: curl -X POST -F q=@$file -F 'mode=tmalign'  https://search.foldseek.com/api/ticket" >> "$log_file"
            ticket_post=$(curl -X POST -F q=@$file -F 'mode=tmalign' -F 'database[]=afdb-swissprot' -F 'database[]=mgnify_esm30' -F 'database[]=pdb100' https://search.foldseek.com/api/ticket) >> "$log_file" 2>&1
            echo $ticket_post >> "$log_file"
            id=$(echo $ticket_post | grep -o '"id":"[^"]*' | cut -d':' -f2 | tr -d '"')
            sleep 2
            check_status "$id" "$log_file" "$output_file"
        fi
    done
done

#-F 'database[]=afdb50' -F 'database[]=afdb-swissprot' -F 'database[]=afdb-proteome' -F 'database[]=cath50' -F 'database[]=mgnify_esm30' -F 'database[]=pdb100' -F 'database[]=gmgcl_id'

#filename="/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/all_atom_structures/folded/backbone/rfdiffusion/RFdiffusion_150it_len100_ff575c66-8d41-40e8-895a-b8b59cd81f41_design5.pdb"
#curl -X POST -F q=@$filename -F 'mode=tmalign' -F 'database[]=afdb50' -F 'database[]=afdb-swissprot' -F 'database[]=afdb-proteome' -F 'database[]=cath50' -F 'database[]=mgnify_esm30' -F 'database[]=pdb100' -F 'database[]=gmgcl_id' https://search.foldseek.com/api/ticket

#curl https://search.foldseek.com/api/ticket/V7yV9HcMoM2JSWxjzSfGhTDbyohFligQy7UmRQ

#result=$(curl -X GET https://search.foldseek.com/api/result/V7yV9HcMoM2JSWxjzSfGhTDbyohFligQy7UmRQ/0)

#echo $result >> check_seq.json