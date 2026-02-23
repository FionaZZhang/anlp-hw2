#!/usr/bin/env python3
"""
Quick script to run the RAG pipeline.
"""
import os
import sys

# Disable parallelism to avoid segfaults on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import process_documents
from src.retriever import HybridRetriever
from src.generator import SimpleGenerator
import json
from pathlib import Path
from tqdm import tqdm


def main():
    # Configuration
    DATA_DIRS = ["baseline_data", "additional_data"]
    DOCUMENTS_PATH = "data/processed/documents.json"
    QUERIES_PATH = "leaderboard_queries.json"
    OUTPUT_PATH = "system_outputs/system_output_2.json"
    REPROCESS_DOCS = False

    # Fixed Andrew ID
    ANDREW_ID = "Magmar" 

    # Model options:
    # - "microsoft/Phi-3-mini-4k-instruct" (3.8B, good balance) 
    # - "mistralai/Mistral-7B-Instruct-v0.2"
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    TOP_K = 5 

    # Step 1: Process documents if needed
    if REPROCESS_DOCS or not Path(DOCUMENTS_PATH).exists():
        print("=" * 50)
        print("Step 1: Processing documents...")
        print("=" * 50)
        Path(DOCUMENTS_PATH).parent.mkdir(parents=True, exist_ok=True)
        documents = process_documents(DATA_DIRS, DOCUMENTS_PATH)
    else:
        print("Documents already processed, loading...")
        with open(DOCUMENTS_PATH, 'r') as f:
            documents = json.load(f)

    print(f"Loaded {len(documents)} document chunks")

    # Step 2: Initialize retriever
    print("\n" + "=" * 50)
    print("Step 2: Initializing hybrid retriever...")
    print("=" * 50)
    retriever = HybridRetriever()
    retriever.index_documents(documents)

    # Step 3: Initialize generator
    print("\n" + "=" * 50)
    print("Step 3: Loading LLM for answer generation...")
    print("=" * 50)
    generator = SimpleGenerator(MODEL_NAME)

    # Step 4: Process queries
    print("\n" + "=" * 50)
    print("Step 4: Processing queries...")
    print("=" * 50)

    with open(QUERIES_PATH, 'r') as f:
        queries = json.load(f)

    results = {}
    if ANDREW_ID:
        results["andrewid"] = ANDREW_ID

    for query in tqdm(queries, desc="Answering questions"):
        qid = query['id']
        question = query['question']

        # Retrieve documents
        retrieved = retriever.retrieve(question, top_k=TOP_K)

        # Generate answer
        answer = generator.generate_answer(question, retrieved)
        results[qid] = answer

    # Save results
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Done! Results saved to {OUTPUT_PATH}")
    print(f"{'=' * 50}")

    # Show sample results
    print("\nSample results:")
    for qid in list(results.keys())[:3]:
        if qid != "andrewid":
            q = next(q for q in queries if q['id'] == qid)
            print(f"\nQ{qid}: {q['question']}")
            print(f"A: {results[qid]}")


if __name__ == "__main__":
    main()
