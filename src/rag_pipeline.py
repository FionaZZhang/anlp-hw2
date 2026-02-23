"""
Main RAG pipeline for question answering.
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

from document_processor import process_documents
from retriever import HybridRetriever, DenseRetriever, SparseRetriever
from generator import AnswerGenerator, SimpleGenerator

# Fixed Andrew ID
ANDREW_ID = "Magmar"


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    def __init__(
        self,
        documents_path: str,
        retriever_type: str = "hybrid",
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.

        Args:
            documents_path: Path to processed documents JSON
            retriever_type: Type of retriever ("dense", "sparse", "hybrid")
            model_name: HuggingFace model name for generation
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        self.retriever_type = retriever_type

        # Load documents
        print(f"Loading documents from {documents_path}")
        with open(documents_path, 'r') as f:
            self.documents = json.load(f)

        # Initialize retriever
        print(f"Initializing {retriever_type} retriever...")
        if retriever_type == "dense":
            self.retriever = DenseRetriever()
        elif retriever_type == "sparse":
            self.retriever = SparseRetriever()
        else:
            self.retriever = HybridRetriever()

        self.retriever.index_documents(self.documents)

        # Initialize generator
        print(f"Initializing generator with {model_name}...")
        try:
            self.generator = SimpleGenerator(model_name)
        except Exception as e:
            print(f"Failed to load SimpleGenerator: {e}")
            print("Falling back to AnswerGenerator...")
            self.generator = AnswerGenerator(model_name)

    def answer_question(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The question to answer
            top_k: Override default top_k

        Returns:
            Dict with answer and retrieved documents
        """
        k = top_k or self.top_k

        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(question, top_k=k)

        # Generate answer
        answer = self.generator.generate_answer(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": [
                {"id": doc['id'], "title": doc['title'], "score": score}
                for doc, score in retrieved
            ]
        }

    def process_queries(self, queries_path: str, output_path: str):
        """
        Process a list of queries and save results.

        Args:
            queries_path: Path to queries JSON file
            output_path: Path to save results
        """
        with open(queries_path, 'r') as f:
            queries = json.load(f)

        results = {"andrewid": ANDREW_ID}

        print(f"Processing {len(queries)} queries...")
        for query in tqdm(queries):
            qid = query['id']
            question = query['question']

            result = self.answer_question(question)
            results[qid] = result['answer']

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Pittsburgh/CMU QA")

    parser.add_argument('--data_dir', default='baseline_data', help='Directory with raw HTML files')
    parser.add_argument('--documents_path', default='data/processed/documents.json', help='Path to processed documents')
    parser.add_argument('--queries_path', default='leaderboard_queries.json', help='Path to queries JSON')
    parser.add_argument('--output_path', default='system_outputs/system_output_1.json', help='Path for output')
    parser.add_argument('--retriever', default='hybrid', choices=['dense', 'sparse', 'hybrid'], help='Retriever type')
    parser.add_argument('--model', default='microsoft/Phi-3-mini-4k-instruct', help='LLM model name')
    parser.add_argument('--top_k', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--process_docs', action='store_true', help='Process documents before running')

    args = parser.parse_args()

    if args.process_docs or not Path(args.documents_path).exists():
        print("Processing documents...")
        Path(args.documents_path).parent.mkdir(parents=True, exist_ok=True)
        process_documents(args.data_dir, args.documents_path)

    # Initialize pipeline
    pipeline = RAGPipeline(
        documents_path=args.documents_path,
        retriever_type=args.retriever,
        model_name=args.model,
        top_k=args.top_k
    )

    # Process queries
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.process_queries(args.queries_path, args.output_path)


if __name__ == "__main__":
    main()
