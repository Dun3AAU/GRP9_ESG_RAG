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
# Keys match the "ticker_name" from golden_dataset_V2.json
# "file_key" matches the filename in the vector store (e.g. "chunks_NVIDIA.json")
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
    "CVX":  {"name": "Chevron", "file_key": "Chevron"},
    "KO":   {"name": "Coca-Cola", "file_key": "Coca-Cola"},
    "CRM":  {"name": "Salesforce", "file_key": "Salesforce"},
    "MU":   {"name": "Micron", "file_key": "Micron"},
    "MS":   {"name": "Morgan Stanley", "file_key": "Morgan Stanley"},
    "INTU": {"name": "Intuit", "file_key": "Intuit"},
    "DIS":  {"name": "The Walt Disney Company", "file_key": "Disney"},
    "PFIZER": {"name": "Pfizer", "file_key": "Pfizer"},
    "ADI":  {"name": "Analog Devices", "file_key": "Analog Devices"},
    "CVS":  {"name": "CVS Health", "file_key": "CVS"},
    "LEGO": {"name": "LEGO Group", "file_key": "LEGO"}, 
    "IKEA": {"name": "IKEA", "file_key": "IKEA"},
    "ASML": {"name": "ASML", "file_key": "ASML"},
    "SHEL": {"name": "Shell", "file_key": "Shell"},
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

def load_data(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

def format_prompt(question: str, docs: List) -> str:
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
    logging.info(f"Starting RAG Batch Job with Hardcoded Questions...")

    # 1. Load Dataset & Prepare Questions
    try:
        raw_data = load_data(args.input_file)
        test_cases = []
        
        for entry in raw_data:
            # Note: JSON uses "ticker_name"
            ticker = entry.get("ticker_name")
            
            # Get clean company name from config, or fallback to JSON value
            config = COMPANY_CONFIG.get(ticker, {
                "name": entry.get("company_name", ticker), 
                "file_key": entry.get("company_name", ticker)
            })
            
            company_clean_name = config["name"]
            
            # --- UPDATED: Hardcoded Questions Map ---
            # We ignore the question text in the JSON and use these templates instead.
            questions_map = {
                1: f"What is the total amount of Scope 1 (Direct) GHG emissions reported on last fiscal year for {company_clean_name}?",
                2: f"By which year does the {company_clean_name} aim to achieve Net Zero emissions?",
                3: f"What are the initiatives by {company_clean_name} to reach the carbon reduction goal?"
            }

            # Iterate through indices 1, 2, 3
            for i in range(1, 4):
                a_key = f"answer{i}"
                src_key = f"source_answer{i}"
                
                # We add the test case using the hardcoded question map
                # We still extract the ground truth from the JSON
                test_cases.append({
                    "ticker": ticker,
                    "company_name": company_clean_name,
                    "file_key": config["file_key"],
                    "question": questions_map[i], 
                    "ground_truth": entry.get(a_key, "Not found"),
                    "ground_truth_source": entry.get(src_key, "")
                })
        
        logging.info(f"Generated {len(test_cases)} questions from {len(raw_data)} companies.")
        
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

    # 3. Retrieve with STRICT FILTERING
    logging.info("Retrieving and filtering contexts...")
    prompts = []
    final_records = []
    
    FETCH_K = args.top_k * 5 

    for i, case in enumerate(test_cases):
        # 3a. Broad Search with the HARDCODED question
        raw_docs = vector_db.similarity_search(case["question"], k=FETCH_K)
        
        # 3b. Strict Filtering by File Key
        filtered_docs = []
        target_key = case["file_key"].lower() 
        
        for doc in raw_docs:
            source = doc.metadata.get("source", "").lower()
            if target_key in source:
                filtered_docs.append(doc)
        
        # 3c. Take Top K
        final_docs = filtered_docs[:args.top_k]
        
        if not final_docs:
            logging.warning(f"No strict matches for {case['company_name']} (key: {target_key}). Using raw docs.")
            final_docs = raw_docs[:args.top_k]

        prompts.append(format_prompt(case["question"], final_docs))
        
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