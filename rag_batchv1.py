import json
import argparse
import logging
import sys
from typing import List, Dict
from vllm import LLM, SamplingParams

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    # 0. Setup Logging
    setup_logging(args.log_file)
    logging.info(f"Starting RAG Batch Job with args: {args}")

    # 1. Load the Golden Dataset & Flatten to List
    logging.info(f"Loading dataset from {args.input_file}...")
    try:
        raw_data = load_data(args.input_file)
        
        # --- FIX: Convert Dictionary to List of Questions ---
        test_cases = []
        for company, info in raw_data.items():
            # Generate Question 1: Reduction Target
            if "reduction_target" in info:
                test_cases.append({
                    "question": f"What is the reduction target for {company}?",
                    "answer": info["reduction_target"]
                })
            # Generate Question 2: Current Progress
            if "current_progress" in info:
                test_cases.append({
                    "question": f"What is the current progress for {company}?",
                    "answer": info["current_progress"]
                })
        
        questions = [item['question'] for item in test_cases]
        logging.info(f"Successfully loaded and generated {len(questions)} questions from dataset.")
        
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 2. Load Embeddings & FAISS Index
    logging.info(f"Loading Embeddings ({args.embedding_model})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}
    )
    
    logging.info(f"Loading FAISS Index from {args.faiss_path}...")
    try:
        vector_db = FAISS.load_local(
            folder_path=args.faiss_path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        logging.info("FAISS index loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")
        sys.exit(1)

    # 3. Retrieve Contexts
    logging.info("Retrieving contexts for all questions...")
    prompts = []
    all_retrieved_docs = []
    
    for i, q in enumerate(questions):
        docs = vector_db.similarity_search(q, k=args.top_k)
        all_retrieved_docs.append(docs)
        prompts.append(format_prompt(q, docs))
        if (i + 1) % 10 == 0:
            logging.info(f"Processed retrieval for {i + 1}/{len(questions)} questions.")

    # 4. Initialize vLLM
    logging.info(f"Initializing vLLM with model: {args.llm_model}...")
    llm = LLM(
        model=args.llm_model,
        tensor_parallel_size=args.num_gpus,
        dtype="auto",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # 5. Generate Answers
    logging.info(f"Generating answers for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Compile & Save Results
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        retrieved_metadata = [
            {"content": d.page_content, "source": d.metadata.get("source", "unknown")} 
            for d in all_retrieved_docs[i]
        ]
        
        # --- FIX: Use the flattened test_cases list for ground truth ---
        results.append({
            "question": questions[i],
            "ground_truth": test_cases[i].get("answer", ""), 
            "generated_answer": generated_text,
            "retrieved_context": retrieved_metadata
        })

    logging.info(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Job completed successfully.")

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