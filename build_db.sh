#!/bin/bash
#SBATCH --job-name=Build_FAISS
#SBATCH --output=logs/build_db/build_%j.out
#SBATCH --error=logs/build_db/build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00 

# --- CONFIGURATION ---
PROJECT_ROOT=$(pwd)
SIF_IMAGE="/ceph/container/vllm-openai_latest.sif"

# Paths
DATA_FOLDER="$PROJECT_ROOT/data/json_chunks"
OUTPUT_FOLDER="$PROJECT_ROOT/data/faiss_index_folder"
VENV_HOST="$PROJECT_ROOT/my_rag_env"
VENV_CONTAINER="/scratch/my_rag_env"
PASSWD_FILE="$PROJECT_ROOT/my_passwd"  # <--- NEW: Temp file for user info

# Logs & Cache
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/cache"
export HF_HOME="$PROJECT_ROOT/cache"

# Define Python Log File
LOG_FILE="$PROJECT_ROOT/logs/build_db_${SLURM_JOB_ID}.log"

echo "Building FAISS DB on $(hostname)"
echo "Python Logs: $LOG_FILE"

# --- STEP 1: GENERATE USER INFO FILE ---
# This grabs your user ID info so the container knows who you are.
# It does NOT contain your password.
getent passwd $USER > "$PASSWD_FILE"

# --- EXECUTION ---
singularity exec --nv \
    -B "$PROJECT_ROOT:$PROJECT_ROOT" \
    -B "$VENV_HOST:$VENV_CONTAINER" \
    -B "$PASSWD_FILE:/etc/passwd" \
    "$SIF_IMAGE" \
    bash -c "
        source $VENV_CONTAINER/bin/activate && \
        pip install --no-cache-dir einops && \
        python3 $PROJECT_ROOT/build_db.py \
        --data_folder $DATA_FOLDER \
        --output_folder $OUTPUT_FOLDER \
        --log_file $LOG_FILE \
        --embedding_model BAAI/bge-m3 
    "
   

# --- CLEANUP ---
# Remove the temporary password file to keep folder clean
rm "$PASSWD_FILE"