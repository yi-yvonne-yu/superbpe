#!/bin/bash
#SBATCH --job-name=tokenizer
#SBATCH --output=/fs/ess/PAS2836/yu4063/decoder/local_model/SuperBPE_10G_1M_finetune_data/superbpe_tok.log
#SBATCH --error=/fs/ess/PAS2836/yu4063/decoder/local_model/SuperBPE_10G_1M_finetune_data/superbpe_tok.err

#SBATCH --partition=hugemem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1800G        
#SBATCH --time=2-00:00:00
#SBATCH --account=PAS2836 

source /users/PAS2834/yu4063/miniconda3/etc/profile.d/conda.sh
conda activate superbpe

orig_tokenizer_dir=/fs/ess/PAS2836/yu4063/decoder/local_model/allenai/OLMo-2-0425-1B
vocab_size=1000000
corpus_dir=/fs/scratch/PAS2836/yu4063/olmo-mix-1124-subset-p99/train  # a directory containing txt files for tokenizer training
num_bytes=$((10**10))

# convert num_bytes to something like 10G or 100M, depending on the value
if [ $num_bytes -ge $((10**9)) ]; then
    num_bytes_str=$(($num_bytes / 10**9))G
elif [ $num_bytes -ge $((10**6)) ]; then
    num_bytes_str=$(($num_bytes / 10**6))M
elif [ $num_bytes -ge $((10**3)) ]; then
    num_bytes_str=$(($num_bytes / 10**3))K
else
    num_bytes_str=${num_bytes}
fi


# convert vocab_size to something like 100K, depending on the value
if [ $vocab_size -ge $((10**6)) ]; then
    vocab_size_str=$(($vocab_size / 10**6))M
elif [ $vocab_size -ge $((10**3)) ]; then
    vocab_size_str=$(($vocab_size / 10**3))K
else
    vocab_size_str=${vocab_size}
fi

# output_dir=/fs/ess/PAS2836/yu4063/decoder/local_model/SuperBPE_10G_1M-tulu_all
output_dir=/fs/ess/PAS2836/yu4063/decoder/local_model/SuperBPE_${num_bytes_str}_${vocab_size_str}_finetune_data_assistant
mkdir $output_dir
echo "output_dir: $output_dir"

# mkdir -p $output_dir
cp $orig_tokenizer_dir/merges.txt $output_dir/merges.txt
# cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --do_whitespace_pretokenization false \
    --corpus_dir $corpus_dir \
    --num_bytes $num_bytes 
