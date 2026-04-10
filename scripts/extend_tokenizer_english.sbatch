#!/bin/bash
#SBATCH --job-name=tokenizer_english
#SBATCH --output=/fs/scratch/PAS2836/yu4063/logs/superbpe/tok_eng_%j.log
#SBATCH --error=/fs/scratch/PAS2836/yu4063/logs/superbpe/tok_eng_%j.err

#SBATCH --partition=hugemem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1800G        
#SBATCH --time=2-00:00:00
#SBATCH --account=PAS2836 

# =============================================================================
# 資料準備指令 (建議在 Login Node 執行一次即可)
# =============================================================================
# 1. 抓取語料 (Aya English):
# python -c "from datasets import load_dataset; ds = load_dataset('CohereLabs/aya_collection_language_split', 'english', split='train', streaming=True); f = open('/fs/scratch/PAS2836/yu4063/local_model/datasets/english/aya_english.txt', 'w', encoding='utf-8'); [f.write(ex['targets'] + '\n') for ex in ds]; f.close()"
# 
# 2. 萃取基礎詞表 (針對有 tokenizer.json 但無 .txt 檔案的模型):
# python -c "import json; d = json.load(open('$orig_tokenizer_dir/tokenizer.json')); f = open('$orig_tokenizer_dir/merges.txt', 'w'); f.write('#version: 0.2\n' + '\n'.join(d['model']['merges'])); f.close()"
# python -c "import json; d = json.load(open('$orig_tokenizer_dir/tokenizer.json')); json.dump(d['model']['vocab'], open('$orig_tokenizer_dir/vocab.json', 'w'));"
# =============================================================================

source /users/PAS2834/yu4063/miniconda3/etc/profile.d/conda.sh
conda activate superbpe

# Scratch Environment setup
export HF_HOME="/fs/scratch/PAS2836/yu4063/.cache/huggingface"
export TRITON_CACHE_DIR="/tmp/$USER/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

orig_tokenizer_dir=/fs/scratch/PAS2836/yu4063/local_model/meta-llama/Llama-3.2-1B
corpus_dir=/fs/scratch/PAS2836/yu4063/local_model/datasets/english
vocab_size=1000000
num_bytes=$((10**10)) # 10G

# convert num_bytes to something like 10G or 100M, depending on the value
if [ $num_bytes -ge $((10**9)) ]; then
    num_bytes_str=$(($num_bytes / 10**9))G
elif [ $num_bytes -ge $((10**6)) ]; then
    num_bytes_str=$(($num_bytes / 10**6))M
else
    num_bytes_str=${num_bytes}
fi

# convert vocab_size to something like 1M or 100K, depending on the value
if [ $vocab_size -ge $((10**6)) ]; then
    vocab_size_str=$(($vocab_size / 10**6))M
elif [ $vocab_size -ge $((10**3)) ]; then
    vocab_size_str=$(($vocab_size / 10**3))K
else
    vocab_size_str=${vocab_size}
fi

output_model_name="Llama-3.2-1B-aya-english"
output_dir=/fs/scratch/PAS2836/yu4063/local_model/SuperBPE_${output_model_name}_${num_bytes_str}_${vocab_size_str}_tokenizer
mkdir -p $output_dir
echo "output_dir: $output_dir"

# Ensure origin merges and vocab are seeded
if [ -f $orig_tokenizer_dir/merges.txt ]; then
    cp $orig_tokenizer_dir/merges.txt $output_dir/merges.txt
fi
if [ -f $orig_tokenizer_dir/vocab.json ]; then
    cp $orig_tokenizer_dir/vocab.json $output_dir/vocab.json
fi

cd superbpe

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --do_whitespace_pretokenization false \
    --num_bytes $num_bytes \
    --corpus_dir $corpus_dir
