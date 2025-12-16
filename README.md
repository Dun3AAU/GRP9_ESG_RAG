# GRP9 ESG RAG Pipeline

> **Note**: This project is designed to run on the **AAU AI LAB (HPC)** cluster with SLURM job scheduling and Singularity containers.

A Retrieval-Augmented Generation (RAG) system for analyzing Environmental, Social, and Governance (ESG) reports from major companies. This pipeline extracts and answers questions about carbon emissions, sustainability targets, and climate initiatives using Large Language Models (LLMs) and vector databases.

## Overview

This project implements a complete RAG pipeline that:
- Processes ESG report chunks from 24+ major companies
- Builds a FAISS vector database for efficient retrieval
- Answers standardized questions about carbon emissions and sustainability goals
- Evaluates answer quality using multiple metrics (semantic similarity, ROUGE-L, LLM judge)

The system focuses on three key ESG questions for each company:
1. **Scope 1 GHG Emissions**: Total direct greenhouse gas emissions reported
2. **Net Zero Target**: Year the company aims to achieve net zero emissions
3. **Carbon Reduction Initiatives**: Specific initiatives to reach carbon reduction goals

## Repository Structure

```
.
├── build_db.py              # Builds FAISS vector database from JSON chunks
├── build_db.sh              # SLURM job script for database building
├── evaluate_rag.py          # Evaluates RAG outputs with multiple metrics
├── run_eval.sh              # SLURM job script for evaluation
├── rag_batchv1.py           # RAG pipeline version 1 (basic)
├── rag_batchv2.py           # RAG pipeline version 2 (improved filtering)
├── rag_batchv3.py           # RAG pipeline version 3 (enhanced retrieval)
├── rag_batchv4.py           # RAG pipeline version 4 (latest, with strict filtering)
├── golden_dataset_V2.json   # Ground truth dataset with 24 companies
├── json_chunks/             # Pre-chunked ESG report text for each company
│   ├── chunks_NVIDIA.json
│   ├── chunks_Microsoft.json
│   ├── chunks_Amazon.json
│   └── ...
├── results_golden_data_v2_faiss_v2.json           # RAG output results
└── results_golden_data_v2_faiss_v2_evaluated.json # Evaluated results with scores
```

## Companies Covered

The dataset includes ESG reports from 24 major companies across various sectors:

- **Technology**: NVIDIA, Microsoft, Amazon, Meta, Apple, Google, Tesla, Netflix, Salesforce, Qualcomm, Intuit, Micron, Analog Devices, ASML
- **Retail**: Walmart, IKEA, LEGO Group
- **Energy**: Chevron, Shell
- **Healthcare**: Pfizer, CVS Health
- **Finance**: Morgan Stanley
- **Consumer Goods**: Coca-Cola, Disney

## Prerequisites

### Environment
This project is designed for the **AAU AI LAB (HPC)** cluster environment and requires:
- Access to AAU AI LAB HPC cluster with SLURM job scheduler
- Singularity container runtime (available on AAU AI LAB)
- Python 3.8+
- CUDA-capable GPU (provided by AAU AI LAB cluster nodes)
- Singularity image: `/ceph/container/vllm-openai_latest.sif` (available on AAU AI LAB)

### Key Dependencies
- `langchain-community` - Vector store and document handling
- `langchain-huggingface` - HuggingFace embeddings integration
- `vllm` - Fast LLM inference
- `sentence-transformers` - Semantic similarity evaluation
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `rouge-score` - ROUGE metric calculation

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Dun3AAU/GRP9_ESG_RAG.git
cd GRP9_ESG_RAG
```

2. **Set up Python virtual environment**:
```bash
python3 -m venv my_rag_env
source my_rag_env/bin/activate
```

3. **Install dependencies**:
```bash
pip install langchain-community langchain-huggingface vllm sentence-transformers faiss-cpu rouge-score scikit-learn torch
```

## Usage

### 1. Build FAISS Vector Database

Build the vector database from pre-chunked ESG reports:

```bash
python build_db.py \
  --data_folder ./json_chunks \
  --output_folder ./faiss_index \
  --embedding_model BAAI/bge-m3 \
  --log_file build_db.log
```

**Parameters**:
- `--data_folder`: Path to directory containing JSON chunk files
- `--output_folder`: Where to save the FAISS index
- `--embedding_model`: HuggingFace embedding model (default: BAAI/bge-m3)
- `--log_file`: Log file path

**SLURM Cluster**: Use `build_db.sh` for cluster deployment with GPU support.

### 2. Run RAG Pipeline

Execute the RAG pipeline to answer ESG questions:

```bash
python rag_batchv4.py \
  --input_file golden_dataset_V2.json \
  --faiss_path ./faiss_index \
  --output_file results.json \
  --embedding_model BAAI/bge-m3 \
  --llm_model mistralai/Mistral-7B-Instruct-v0.2 \
  --top_k 3 \
  --num_gpus 1
```

**Parameters**:
- `--input_file`: Golden dataset with ground truth answers
- `--faiss_path`: Path to FAISS index created by build_db.py
- `--output_file`: Where to save RAG results
- `--embedding_model`: Embedding model for retrieval (should match build step)
- `--llm_model`: LLM for answer generation
- `--top_k`: Number of context documents to retrieve per question
- `--num_gpus`: Number of GPUs for LLM inference
- `--log_file`: Log file path (default: rag_execution.log)

### 3. Evaluate Results

Evaluate the quality of generated answers:

```bash
python evaluate_rag.py \
  --results_file results.json \
  --golden_file golden_dataset_V2.json \
  --embedding_model BAAI/bge-m3 \
  --llm_model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --num_gpus 1
```

**Parameters**:
- `--results_file`: RAG output file to evaluate
- `--golden_file`: Ground truth dataset
- `--embedding_model`: Model for semantic similarity
- `--llm_model`: LLM for judging answer quality
- `--num_gpus`: Number of GPUs for evaluation

**SLURM Cluster**: Use `run_eval.sh` for cluster deployment.

## Pipeline Workflow

```
┌─────────────────┐
│  JSON Chunks    │ (Pre-chunked ESG reports)
│  (24 companies) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  build_db.py    │ → Creates FAISS vector index
└────────┬────────┘
         │
         v
┌─────────────────┐
│  FAISS Index    │ (Vector embeddings of all chunks)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ rag_batchv4.py  │ → Retrieves context & generates answers
│                 │   (Question → Retrieve → Generate)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Results JSON    │ (Generated answers + retrieved context)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ evaluate_rag.py │ → Calculates quality metrics
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Evaluated JSON  │ (Results + scores)
└─────────────────┘
```

## Output Format

### RAG Results (`results.json`)
```json
[
  {
    "ticker": "NVDA",
    "company_name": "NVIDIA",
    "file_key": "NVIDIA",
    "question": "What is the total amount of Scope 1 (Direct) GHG emissions...",
    "ground_truth": "14,390 metric tons CO2e",
    "ground_truth_source": "In 2025 [FY24], the total Scope 1 emissions...",
    "generated_answer": "According to the context provided...",
    "retrieved_context": [
      {
        "content": "Text chunk from ESG report...",
        "source": "chunks_NVIDIA.json"
      }
    ]
  }
]
```

### Evaluated Results (`results_evaluated.json`)
Includes additional `scores` field:
```json
{
  "scores": {
    "recall": 1.0,
    "similarity": 0.8542,
    "judge": 4
  }
}
```

## Evaluation Metrics

The pipeline uses four metrics to assess answer quality:

1. **Context Recall** (0-1): Whether the ground truth evidence was retrieved
2. **Semantic Similarity** (0-1): Cosine similarity between generated and ground truth answers
3. **ROUGE-L** (0-1): Longest common subsequence overlap
4. **LLM Judge Score** (1-5): AI-based quality assessment

**Target Performance**:
- Context Recall: >90%
- Semantic Similarity: >0.70
- ROUGE-L: >0.40
- LLM Judge Score: >4.0/5

## Configuration

### Company Configuration (rag_batchv4.py)

The `COMPANY_CONFIG` dictionary maps stock tickers to company names and file keys:

```python
COMPANY_CONFIG = {
    "NVDA": {"name": "NVIDIA", "file_key": "NVIDIA"},
    "MSFT": {"name": "Microsoft", "file_key": "Microsoft"},
    # ... more companies
}
```

Add new companies by:
1. Adding chunks to `json_chunks/chunks_CompanyName.json`
2. Adding entry to `COMPANY_CONFIG`
3. Adding ground truth to `golden_dataset_V2.json`

### Embedding Models

Default: `BAAI/bge-m3` (multilingual, high quality)

Alternatives:
- `sentence-transformers/all-mpnet-base-v2`
- `intfloat/e5-large-v2`
- `BAAI/bge-large-en-v1.5`

### LLM Models

Default: `mistralai/Mistral-7B-Instruct-v0.2`

Alternatives:
- `meta-llama/Llama-2-7b-chat-hf`
- `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` (quantized, faster)
- Any Mistral/Llama-compatible model

## RAG Pipeline Versions

- **v1**: Basic retrieval without filtering
- **v2**: Added company-specific filtering
- **v3**: Enhanced context formatting
- **v4**: Strict filtering with hardcoded questions (recommended)

Each version improves retrieval accuracy and answer quality. Use **v4** for production.

## Golden Dataset Structure

The `golden_dataset_V2.json` contains ground truth for evaluation:

```json
[
  {
    "ticker_name": "NVDA",
    "company_name": "NVIDIA",
    "question1": "What is the total amount of Scope 1...",
    "answer1": "14,390 metric tons CO2e",
    "source_answer1": "In 2025 [FY24], the total Scope 1...",
    "question2": "By which year does the company aim...",
    "answer2": "Not found",
    "source_answer2": "NA (Report focuses on...)",
    "question3": "What are the initiatives...",
    "answer3": "Energy efficiency in GPUs...",
    "source_answer3": "Initiatives include: transitioning to 100%..."
  }
]
```

## Troubleshooting

### Out of Memory Errors
- Reduce `gpu_memory_utilization` in LLM initialization
- Decrease `batch_size` in embedding encode_kwargs
- Use quantized models (AWQ/GPTQ)

### Poor Retrieval Quality
- Increase `top_k` parameter
- Adjust `FETCH_K` multiplier in rag_batchv4.py
- Verify company `file_key` matches chunk filename

### Low Evaluation Scores
- Check if ground truth sources are in the chunks
- Verify embedding model consistency between build and query
- Review prompt formatting in `format_prompt()`

## SLURM Cluster Deployment

The repository includes SLURM scripts for HPC cluster deployment:

- **build_db.sh**: Builds database with GPU acceleration
- **run_eval.sh**: Runs evaluation with resource management

Key SLURM parameters:
- `--gres=gpu:1`: Request GPU
- `--mem=32G`: Memory allocation
- `--time=02:00:00`: Time limit

## License

This project is for academic/research purposes. Please cite appropriately if used in publications.

## Contributing

When adding new companies:
1. Add chunked report to `json_chunks/`
2. Update `COMPANY_CONFIG` in RAG scripts
3. Add ground truth to golden dataset
4. Rebuild FAISS index
5. Run evaluation to verify

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

## Acknowledgments

- LangChain for RAG framework
- HuggingFace for models and embeddings
- vLLM for efficient LLM inference
- FAISS for vector similarity search
