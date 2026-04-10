#!/bin/bash
#SBATCH --job-name=tokenizer
#SBATCH --output=/fs/scratch/PAS2836/yu4063/logs/superbpe/tok_%j.log
#SBATCH --error=/fs/scratch/PAS2836/yu4063/logs/superbpe/tok_%j.err

#SBATCH --partition=hugemem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1800G        
#SBATCH --time=2-00:00:00
#SBATCH --account=PAS2836 

source /users/PAS2834/yu4063/miniconda3/etc/profile.d/conda.sh
conda activate superbpe

orig_tokenizer_dir=/fs/scratch/PAS2836/yu4063/local_model/allenai/OLMo-2-1124-7B
vocab_size=1000000
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

# output_dir=/fs/scratch/PAS2836/yu4063/local_model/SuperBPE_10G_1M-tulu_all
output_model_name="OLMo-2-7B-tulu"
output_dir=/fs/scratch/PAS2836/yu4063/local_model/SuperBPE_${output_model_name}_${num_bytes_str}_${vocab_size_str}_tokenizer
mkdir -p $output_dir
echo "output_dir: $output_dir"

# Ensure origin merges are seeded
cp $orig_tokenizer_dir/merges.txt $output_dir/merges.txt

cd superbpe

# Run main training using HF dataset
python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --do_whitespace_pretokenization false \
    --num_bytes $num_bytes 
