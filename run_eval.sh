#!/bin/bash
#SBATCH --job-name=RAG_Eval_v2
#SBATCH --output=logs/eval_rag/eval_%j.out
#SBATCH --error=logs/eval_rag/eval_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=ailab-l4-01

# --- CONFIGURATION ---
PROJECT_ROOT=$(pwd)
SIF_IMAGE="/ceph/container/vllm-openai_latest.sif"
PASSWD_FILE="$PROJECT_ROOT/my_passwd" 

# Environment Paths
VENV_HOST="$PROJECT_ROOT/my_rag_env"
VENV_CONTAINER="/scratch/my_rag_env"

export HF_HOME="$PROJECT_ROOT/cache"

echo "Evaluation Job running on $(hostname)"

# --- STEP 1: USER ID FIX ---
getent passwd $USER > "$PASSWD_FILE"

# --- EXECUTION ---
singularity exec --nv \
    -B "$PROJECT_ROOT:$PROJECT_ROOT" \
    -B "$VENV_HOST:$VENV_CONTAINER" \
    "$SIF_IMAGE" \
    bash -c "
     
        export TMPDIR=/scratch/singularity/tmp && \
        export TOKENIZERS_PARALLELISM=false && \
        export TORCH_CUDA_ARCH_LIST='8.9' && \
        export VLLM_WORKER_MULTIPROC_METHOD=spawn && \
        source $VENV_CONTAINER/bin/activate && \
        pip install --no-cache-dir \
            scikit-learn \
            rouge-score \
            sentence-transformers \
            vllm && \
        
        # Running the V2 Evaluation Script
        python3 $PROJECT_ROOT/evaluate_rag_v2.py \
        --results_file $PROJECT_ROOT/results.json \
        --golden_file $PROJECT_ROOT/data/golden_dataset_V2.json \
        --embedding_model BAAI/bge-m3 \
        --llm_model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
        --num_gpus 1
    "

    rm "$PASSWD_FILE"