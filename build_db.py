import json
import os
import glob
import argparse
import logging
import sys
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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

def load_chunks_from_folder(folder_path: str) -> List[Document]:
    documents = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    logging.info(f"Found {len(json_files)} JSON files in {folder_path}")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Check structure
                if isinstance(data, list) and isinstance(data[0], str):
                    for text in data:
                        documents.append(Document(page_content=text, metadata={"source": file_path}))
                
                elif isinstance(data, list) and isinstance(data[0], dict):
                    for item in data:
                        content = item.get("text") or item.get("content") or item.get("chunk")
                        if content:
                            documents.append(Document(page_content=content, metadata={"source": file_path, **item}))
        except Exception as e:
            logging.warning(f"Skipping file {file_path} due to error: {e}")

    return documents

def main(args):
    setup_logging(args.log_file)
    logging.info(f"Starting FAISS Build with args: {args}")

    # 1. Load Data
    logging.info(f"Loading data from {args.data_folder}...")
    docs = load_chunks_from_folder(args.data_folder)
    logging.info(f"Total chunks loaded: {len(docs)}")

    if not docs:
        logging.error("No documents found. Exiting.")
        sys.exit(1)

    # 2. Initialize Embedding Model
    logging.info(f"Loading Embedding Model: {args.embedding_model}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={'device': 'cuda', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32} # Adjust batch size as needed
        )
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        sys.exit(1)

        # --- SAFETY BLOCK: --- Clear GPU Cache ---
    import torch
    import gc
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ------------------------------

    # 3. Create FAISS Index
    logging.info("Creating FAISS index (this may take a while)...")
    try:
        db = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        logging.error(f"Failed to create FAISS index: {e}")
        sys.exit(1)

    # 4. Save to Disk
    logging.info(f"Saving index to {args.output_folder}...")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    db.save_local(args.output_folder)
    logging.info("Database build finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--log_file", type=str, default="build_db.log") # New Argument
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()
    main(args)