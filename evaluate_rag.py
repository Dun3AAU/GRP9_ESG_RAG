import json
import argparse
import logging
import re
import sys
import numpy as np
from typing import List, Dict
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# --- LLM JUDGE PROMPT ---
def format_judge_prompt(question, ground_truth, generated_answer):
    return f"""[INST] You are an impartial judge evaluating the quality of an AI assistant's answer.
    
    1. Compare the Generated Answer to the Ground Truth.
    2. Assign a score from 1 to 5 based on accuracy and completeness.
    3. Return ONLY the integer score.

    Scale:
    1 - Completely incorrect or irrelevant.
    2 - Mostly incorrect, misses key points.
    3 - Partially correct, misses some details.
    4 - Mostly correct, minor details missing.
    5 - Completely correct and accurate.

    ### Question:
    {question}

    ### Ground Truth:
    {ground_truth}

    ### Generated Answer:
    {generated_answer}

    ### Score (1-5):
    [/INST]"""

def main(args):
    # 1. Load Data
    logging.info(f"Loading results from {args.results_file}...")
    results = load_json(args.results_file)
    
    # We load the golden file for reference, though our V5 RAG script 
    # already injected the ground truth into the results file.
    logging.info(f"Loading golden dataset from {args.golden_file}...")
    golden_data = load_json(args.golden_file)

    # 2. Initialize Standard Metrics
    logging.info("Initializing Sentence Transformer for similarity...")
    sim_model = SentenceTransformer(args.embedding_model, device='cpu')
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 3. Initialize vLLM for Judging
    logging.info(f"Initializing Judge LLM: {args.llm_model}...")
    llm = LLM(
        model=args.llm_model,
        quantization="awq",
        tensor_parallel_size=args.num_gpus,
        dtype="auto",
        gpu_memory_utilization=0.60,
        trust_remote_code=True,
        enforce_eager=True
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # 4. Prepare Evaluation Loops
    judge_prompts = []
    metrics = {
        "context_recall": [],
        "semantic_similarity": [],
        "rouge_l": [],
        "llm_judge_score": []
    }

    logging.info("Calculating Standard Metrics & Preparing Judge Prompts...")
    
    for item in results:
        # Extract fields directly from the V3/V5 RAG result structure
        q = item.get('question', '')
        gen = item.get('generated_answer', '')
        truth = item.get('ground_truth', 'Not found')
        truth_source = item.get('ground_truth_source', '') # The specific evidence text
        
        # 'retrieved_context' is a list of dicts: [{'content': '...', 'source': '...'}]
        retrieved_texts = [d.get('content', '') for d in item.get('retrieved_context', [])]

        # --- A. Context Recall (Evidence Hit Rate) ---
        # We check if the 'ground_truth_source' (from the golden dataset) 
        # appears in any of the chunks we retrieved.
        hit = 0.0
        
        # If the truth is "Not found", we technically don't need evidence, 
        # but usually, we check if the source text was retrieved.
        if not truth_source or truth_source == "NA":
            # If there is no specific source text to look for, we skip strictly penalizing.
            # Alternatively, set to 1.0 if we consider "no source needed" as success.
            hit = 1.0 
        else:
            # Normalize strings for comparison
            truth_snippet = truth_source.lower()
            # We check if a significant portion of the truth source is in the retrieved chunks
            # Since exact matching is hard with long text, we check if at least one chunk
            # contains a reasonable substring or the whole thing.
            for ctx in retrieved_texts:
                # 1. Simple inclusion check
                if truth_snippet in ctx.lower():
                    hit = 1.0
                    break
                
                # 2. Fallback: Semantic overlap or keyword matching could go here
                # For now, we stick to a strict substring check or simple overlap
                # (You can implement more fuzzy matching if needed)
                
        metrics['context_recall'].append(hit)

        # --- B. Semantic Similarity ---
        # Encode Generated vs Ground Truth
        emb1 = sim_model.encode(gen, convert_to_tensor=True)
        emb2 = sim_model.encode(truth, convert_to_tensor=True)
        metrics['semantic_similarity'].append(util.cos_sim(emb1, emb2).item())

        # --- C. ROUGE-L ---
        metrics['rouge_l'].append(rouge.score(truth, gen)['rougeL'].fmeasure)

        # --- D. Prepare Judge Prompt ---
        judge_prompts.append(format_judge_prompt(q, truth, gen))

    # 5. Run LLM Judge Batch
    logging.info(f"Running LLM Judge on {len(judge_prompts)} items...")
    if judge_prompts:
        judge_outputs = llm.generate(judge_prompts, sampling_params)

        for output in judge_outputs:
            try:
                score_text = output.outputs[0].text.strip()
                # Extract first digit found in response
                match = re.search(r'\d+', score_text)
                if match:
                    score = int(match.group())
                    score = max(1, min(5, score)) # Clamp
                else:
                    score = 1
            except:
                score = 1 
            metrics['llm_judge_score'].append(score)
    else:
        logging.warning("No judge prompts generated.")

    # 6. Final Report
    avg_recall = np.mean(metrics['context_recall']) if metrics['context_recall'] else 0
    avg_sim = np.mean(metrics['semantic_similarity']) if metrics['semantic_similarity'] else 0
    avg_rouge = np.mean(metrics['rouge_l']) if metrics['rouge_l'] else 0
    avg_judge = np.mean(metrics['llm_judge_score']) if metrics['llm_judge_score'] else 0

    print("\n" + "="*40)
    print("       RAG PIPELINE EVALUATION REPORT       ")
    print("="*40)
    print(f"Total Samples:       {len(results)}")
    print(f"Context Recall:      {avg_recall:.2%}  (Evidence Found?)")
    print(f"Semantic Similarity: {avg_sim:.4f}  (Target: >0.70)")
    print(f"ROUGE-L Score:       {avg_rouge:.4f}  (Target: >0.40)")
    print(f"LLM Judge Score:     {avg_judge:.2f}/5 (Target: >4.0)")
    print("="*40)

    # Save detailed metrics
    output_path = args.results_file.replace(".json", "_evaluated.json")
    with open(output_path, "w") as f:
        # Add scores back to the result items for inspection
        for i, item in enumerate(results):
            item['scores'] = {
                "recall": metrics['context_recall'][i] if i < len(metrics['context_recall']) else 0,
                "similarity": metrics['semantic_similarity'][i] if i < len(metrics['semantic_similarity']) else 0,
                "judge": metrics['llm_judge_score'][i] if i < len(metrics['llm_judge_score']) else 0
            }
        json.dump(results, f, indent=2)
    logging.info(f"Detailed evaluation saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--golden_file", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--llm_model", type=str, default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ") # Use AWQ for faster eval
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    main(args)