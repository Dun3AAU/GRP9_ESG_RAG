import json
import argparse
import logging
import sys
import os
from typing import List, Dict
from vllm import LLM, SamplingParams

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
# "name": Used in the question text (e.g., "What is the target for NVIDIA?")
# "file_key": Used to match the filename (e.g., matches "chunks_NVIDIA.json")
COMPANY_CONFIG = {
    "NVDA": {"name": "NVIDIA", "file_key": "NVIDIA"},
    "MSFT": {"name": "Microsoft", "file_key": "Microsoft"},
    "AMZN": {"name": "Amazon", "file_key": "Amazon"},
    "META": {"name": "Meta", "file_key": "Meta"},
    "TSLA": {"name": "Tesla", "file_key": "Tesla"},
    "AAPL": {"name": "Apple", "file_key": "Apple"},
    "GOOG": {"name": "Google", "file_key": "Google"},
    "WMT": {"name": "Walmart", "file_key": "Walmart"},
    "QCOM": {"name": "Qualcomm", "file_key": "Qualcomm"},
    "NFLX": {"name": "Netflix", "file_key": "Netflix"},
    "CHEVRON": {"name": "Chevron", "file_key": "Chevron"},
    "COCA COLA": {"name": "Coca-Cola", "file_key": "Coca-Cola"},
    "SALESFORCE": {"name": "Salesforce", "file_key": "Salesforce"},
    "MICRON": {"name": "Micron", "file_key": "Micron"},
    "MORGAN STANLEY": {"name": "Morgan Stanley", "file_key": "Morgan Stanley"},
    "INTUIT": {"name": "Intuit", "file_key": "Intuit"},
    "WALT DISNEY": {"name": "The Walt Disney Company", "file_key": "Disney"}, # Fixed File Key
    "PFIZER": {"name": "Pfizer", "file_key": "Pfizer"},
    "ANALOG DEVICES": {"name": "Analog Devices", "file_key": "Analog Devices"},
    "CVS HEALTH": {"name": "CVS Health", "file_key": "CVS"}, # Fixed File Key
    "LEGO": {"name": "LEGO Group", "file_key": "LEGO"}, 
    "IKEA": {"name": "IKEA", "file_key": "IKEA"},
    "ASML": {"name": "ASML", "file_key": "ASML"},
    "SHELL": {"name": "Shell", "file_key": "Shell"},
    "CARGILL": {"name": "Cargill", "file_key": "Cargill"}
}

def setup_logging(output_log):
    logging.basicConfig(
        filename=output_log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    logging.info("Logging initialized.")

def load_data(json_path: str) -> Dict:
    with open(json_path, 'r') as f:
        return json.load(f)

def format_prompt(question: str, docs: List) -> str:
    # We add a small instruction to ignore irrelevant context if any slipped through
    context_text = "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(docs)])
    prompt = f"""[INST] You are a helpful AI assistant. Answer the user's question based strictly on the context provided below.

### Context:
{context_text}

### Question:
{question}

### Answer:
[/INST]"""
    return prompt

def main(args):
    setup_logging(args.log_file)
    logging.info(f"Starting RAG Batch Job with Strict Filtering...")

    # 1. Load Dataset & Prepare Questions
    try:
        raw_data = load_data(args.input_file)
        test_cases = []
        for ticker, info in raw_data.items():
            # Get config or fallback
            config = COMPANY_CONFIG.get(ticker, {"name": ticker, "file_key": ticker})
            
            # Helper to add a test case
            def add_case(q_text, answer_key):
                if answer_key in info:
                    test_cases.append({
                        "ticker": ticker,
                        "company_name": config["name"],
                        "file_key": config["file_key"],
                        "question": f"{q_text} {config['name']}?",
                        "ground_truth": info[answer_key]
                    })

            add_case("What is the reduction target for", "reduction_target")
            add_case("What is the current progress for", "current_progress")
        
        logging.info(f"Generated {len(test_cases)} questions.")
        
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 2. Load Embeddings & FAISS
    logging.info(f"Loading FAISS Index from {args.faiss_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}
    )
    vector_db = FAISS.load_local(
        folder_path=args.faiss_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Retrieve with FILTERING
    logging.info("Retrieving and filtering contexts...")
    prompts = []
    final_records = []
    
    # We fetch more docs initially (top_k * 5) to ensure we have enough after filtering
    FETCH_K = args.top_k * 5 

    for i, case in enumerate(test_cases):
        # 1. Broad Search
        raw_docs = vector_db.similarity_search(case["question"], k=FETCH_K)
        
        # 2. Strict Filtering
        filtered_docs = []
        target_key = case["file_key"].lower() # Normalize to lowercase for matching
        
        for doc in raw_docs:
            source = doc.metadata.get("source", "").lower()
            # Check if the company file key appears in the source path
            # e.g. "nvidia" in ".../chunks_nvidia.json"
            if target_key in source:
                filtered_docs.append(doc)
        
        # 3. Take Top K from filtered list
        final_docs = filtered_docs[:args.top_k]
        
        # Fallback: If filtering removed ALL docs (rare), use raw docs but warn
        if not final_docs:
            logging.warning(f"No strict matches for {case['company_name']} (key: {target_key}). Using raw docs.")
            final_docs = raw_docs[:args.top_k]

        prompts.append(format_prompt(case["question"], final_docs))
        
        # Store metadata for final JSON
        case["retrieved_context"] = [
            {"content": d.page_content, "source": d.metadata.get("source", "unknown")} 
            for d in final_docs
        ]
        final_records.append(case)

    # 4. Initialize vLLM
    logging.info(f"Initializing vLLM ({args.llm_model})...")
    llm = LLM(
        model=args.llm_model,
        tensor_parallel_size=args.num_gpus,
        dtype="auto",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True 
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # 5. Generate
    logging.info("Generating answers...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Save Results
    results = []
    for i, output in enumerate(outputs):
        record = final_records[i]
        record["generated_answer"] = output.outputs[0].text
        results.append(record)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Job completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--faiss_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="rag_results.json")
    parser.add_argument("--log_file", type=str, default="rag_execution.log")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--llm_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    main(args)