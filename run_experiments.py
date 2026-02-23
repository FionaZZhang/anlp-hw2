#!/usr/bin/env python3
"""
Experiment runner for RAG system variations.
Generates results and visualizations for the report.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import process_documents
from src.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from src.generator import SimpleGenerator
from tqdm import tqdm

ANDREW_ID = "Magmar"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    data_dirs: List[str]
    retriever_type: str  # "dense", "sparse", "hybrid"
    embedding_model: str
    top_k: int
    fusion_method: str  # "rrf" or "weighted" (for hybrid only)
    dense_weight: float  # for weighted fusion
    chunk_size: int
    chunk_overlap: int


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config_name: str
    num_documents: int
    num_chunks: int
    indexing_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    total_time: float
    output_file: str


def get_experiment_configs() -> List[ExperimentConfig]:
    """Define all experiment configurations."""
    configs = []

    # Base configurations
    base_embedding = "all-MiniLM-L6-v2"

    # Experiment 1: Baseline data only with different retrievers
    configs.append(ExperimentConfig(
        name="baseline_dense",
        data_dirs=["baseline_data"],
        retriever_type="dense",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    configs.append(ExperimentConfig(
        name="baseline_sparse",
        data_dirs=["baseline_data"],
        retriever_type="sparse",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    configs.append(ExperimentConfig(
        name="baseline_hybrid_rrf",
        data_dirs=["baseline_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    # Experiment 2: Baseline + Additional data
    configs.append(ExperimentConfig(
        name="full_dense",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="dense",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    configs.append(ExperimentConfig(
        name="full_sparse",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="sparse",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    configs.append(ExperimentConfig(
        name="full_hybrid_rrf",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    # Experiment 3: Different fusion methods (hybrid only)
    configs.append(ExperimentConfig(
        name="full_hybrid_weighted_05",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="weighted",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    configs.append(ExperimentConfig(
        name="full_hybrid_weighted_07",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="weighted",
        dense_weight=0.7,
        chunk_size=500,
        chunk_overlap=100
    ))

    # Experiment 4: Different top_k values
    configs.append(ExperimentConfig(
        name="full_hybrid_topk3",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=3,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    configs.append(ExperimentConfig(
        name="full_hybrid_topk7",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=7,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=500,
        chunk_overlap=100
    ))

    # Experiment 5: Different chunk sizes
    configs.append(ExperimentConfig(
        name="full_hybrid_chunk300",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=300,
        chunk_overlap=50
    ))

    configs.append(ExperimentConfig(
        name="full_hybrid_chunk700",
        data_dirs=["baseline_data", "additional_data"],
        retriever_type="hybrid",
        embedding_model=base_embedding,
        top_k=5,
        fusion_method="rrf",
        dense_weight=0.5,
        chunk_size=700,
        chunk_overlap=150
    ))

    return configs


def process_docs_for_config(config: ExperimentConfig, cache_dir: str = "data/cache") -> List[Dict]:
    """Process documents with specific chunking parameters."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create cache key based on config
    data_key = "_".join(sorted([Path(d).name for d in config.data_dirs]))
    cache_file = cache_path / f"docs_{data_key}_c{config.chunk_size}_o{config.chunk_overlap}.json"

    if cache_file.exists():
        print(f"  Loading cached documents from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Process documents with custom chunking
    from src.document_processor import extract_text_from_html, chunk_text, clean_text
    import re

    documents = []
    html_files = []

    for data_dir in config.data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            html_files.extend(list(dir_path.glob('*.htm')))
            html_files.extend(list(dir_path.glob('*.html')))

    for html_path in html_files:
        try:
            doc = extract_text_from_html(str(html_path))
            if len(doc['text']) < 100:
                continue

            # Custom chunking with config params
            text = doc['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > config.chunk_size and current_chunk:
                    chunk_text_str = ' '.join(current_chunk)
                    chunks.append(chunk_text_str)
                    overlap_words = config.chunk_size - config.chunk_overlap
                    words = chunk_text_str.split()
                    if len(words) > overlap_words:
                        current_chunk = [' '.join(words[-config.chunk_overlap:])]
                        current_length = config.chunk_overlap
                    else:
                        current_chunk = []
                        current_length = 0
                current_chunk.append(sentence)
                current_length += sentence_length

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f"{Path(html_path).stem}_{i}",
                    'title': doc['title'],
                    'text': chunk,
                    'source': str(html_path)
                })
        except Exception as e:
            print(f"  Error processing {html_path}: {e}")

    # Cache the processed documents
    with open(cache_file, 'w') as f:
        json.dump(documents, f)

    return documents


def create_retriever(config: ExperimentConfig, documents: List[Dict]):
    """Create retriever based on config."""
    if config.retriever_type == "dense":
        retriever = DenseRetriever(config.embedding_model)
    elif config.retriever_type == "sparse":
        retriever = SparseRetriever()
    else:
        retriever = HybridRetriever(config.embedding_model)

    retriever.index_documents(documents)
    return retriever


def run_experiment(
    config: ExperimentConfig,
    queries: List[Dict],
    generator: SimpleGenerator,
    output_dir: str = "experiment_outputs"
) -> ExperimentResult:
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name}")
    print(f"{'='*60}")
    print(f"  Data: {config.data_dirs}")
    print(f"  Retriever: {config.retriever_type}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Fusion: {config.fusion_method} (weight={config.dense_weight})")
    print(f"  Chunks: size={config.chunk_size}, overlap={config.chunk_overlap}")

    # Process documents
    start_time = time.time()
    documents = process_docs_for_config(config)
    num_chunks = len(documents)
    print(f"  Loaded {num_chunks} document chunks")

    # Create retriever
    retriever = create_retriever(config, documents)
    indexing_time = time.time() - start_time
    print(f"  Indexing time: {indexing_time:.2f}s")

    # Process queries
    results = {"andrewid": ANDREW_ID}
    retrieval_times = []
    generation_times = []

    for query in tqdm(queries, desc="  Processing queries"):
        qid = query['id']
        question = query['question']

        # Retrieve
        ret_start = time.time()
        if config.retriever_type == "hybrid":
            retrieved = retriever.retrieve(
                question,
                top_k=config.top_k,
                method=config.fusion_method,
                dense_weight=config.dense_weight
            )
        else:
            retrieved = retriever.retrieve(question, top_k=config.top_k)
        retrieval_times.append(time.time() - ret_start)

        # Generate
        gen_start = time.time()
        answer = generator.generate_answer(question, retrieved)
        generation_times.append(time.time() - gen_start)

        results[qid] = answer

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{config.name}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    return ExperimentResult(
        config_name=config.name,
        num_documents=len(set(d['source'] for d in documents)),
        num_chunks=num_chunks,
        indexing_time=indexing_time,
        avg_retrieval_time=np.mean(retrieval_times),
        avg_generation_time=np.mean(generation_times),
        total_time=total_time,
        output_file=str(output_file)
    )


def generate_report_charts(results: List[ExperimentResult], output_dir: str = "report_figures"):
    """Generate charts for the report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    names = [r.config_name for r in results]
    indexing_times = [r.indexing_time for r in results]
    retrieval_times = [r.avg_retrieval_time * 1000 for r in results]  # Convert to ms
    generation_times = [r.avg_generation_time for r in results]
    num_chunks = [r.num_chunks for r in results]

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # Chart 1: Number of chunks per configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), num_chunks, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Number of Document Chunks')
    ax.set_title('Document Chunks per Configuration')
    for bar, val in zip(bars, num_chunks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'chunks_per_config.png', dpi=150)
    plt.close()

    # Chart 2: Retrieval time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), retrieval_times, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average Retrieval Time (ms)')
    ax.set_title('Retrieval Time per Configuration')
    for bar, val in zip(bars, retrieval_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'retrieval_times.png', dpi=150)
    plt.close()

    # Chart 3: Generation time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), generation_times, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average Generation Time (s)')
    ax.set_title('Generation Time per Configuration')
    for bar, val in zip(bars, generation_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'generation_times.png', dpi=150)
    plt.close()

    # Chart 4: Retriever type comparison (grouped)
    retriever_types = ['dense', 'sparse', 'hybrid']
    data_types = ['baseline', 'full']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(retriever_types))
    width = 0.35

    baseline_times = []
    full_times = []
    for rt in retriever_types:
        baseline_match = [r for r in results if f'baseline_{rt}' in r.config_name]
        full_match = [r for r in results if f'full_{rt}' in r.config_name and 'rrf' in r.config_name]
        baseline_times.append(baseline_match[0].avg_retrieval_time * 1000 if baseline_match else 0)
        full_times.append(full_match[0].avg_retrieval_time * 1000 if full_match else 0)

    bars1 = ax.bar(x - width/2, baseline_times, width, label='Baseline Data', color='steelblue')
    bars2 = ax.bar(x + width/2, full_times, width, label='Full Data', color='coral')

    ax.set_ylabel('Average Retrieval Time (ms)')
    ax.set_title('Retrieval Time: Baseline vs Full Data')
    ax.set_xticks(x)
    ax.set_xticklabels(['Dense', 'Sparse', 'Hybrid (RRF)'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'baseline_vs_full.png', dpi=150)
    plt.close()

    # Chart 5: Fusion method comparison
    fusion_results = [r for r in results if 'hybrid' in r.config_name and 'full' in r.config_name]
    if len(fusion_results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        fusion_names = [r.config_name.replace('full_hybrid_', '') for r in fusion_results]
        fusion_times = [r.avg_retrieval_time * 1000 for r in fusion_results]

        bars = ax.bar(range(len(fusion_names)), fusion_times, color=plt.cm.Paired(np.linspace(0, 1, len(fusion_names))))
        ax.set_xticks(range(len(fusion_names)))
        ax.set_xticklabels(fusion_names, rotation=45, ha='right')
        ax.set_ylabel('Average Retrieval Time (ms)')
        ax.set_title('Hybrid Retrieval: Fusion Method Comparison')
        plt.tight_layout()
        plt.savefig(output_path / 'fusion_comparison.png', dpi=150)
        plt.close()

    print(f"\nCharts saved to {output_path}/")


def generate_stats_table(results: List[ExperimentResult], output_dir: str = "report_figures"):
    """Generate statistics table for the report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create markdown table
    md_content = """# Experiment Results Summary

## Configuration Comparison

| Configuration | Documents | Chunks | Index Time (s) | Avg Retrieval (ms) | Avg Generation (s) | Total Time (s) |
|--------------|-----------|--------|----------------|--------------------|--------------------|----------------|
"""

    for r in results:
        md_content += f"| {r.config_name} | {r.num_documents} | {r.num_chunks} | {r.indexing_time:.2f} | {r.avg_retrieval_time*1000:.2f} | {r.avg_generation_time:.2f} | {r.total_time:.2f} |\n"

    # Add leaderboard scores section with placeholders
    md_content += """
## Leaderboard Scores (Fill in after submission)

| Configuration | Submitted? | Recall | F1 | ROUGE-L | LLM-Judge | Notes |
|--------------|------------|--------|-----|---------|-----------|-------|
"""
    for r in results:
        md_content += f"| {r.config_name} | [ ] | ___ | ___ | ___ | ___ | |\n"

    md_content += """
*Instructions: Mark [x] when submitted, fill in scores from leaderboard. Max 10 submissions allowed.*

"""

    # Add analysis sections
    md_content += """
## Key Findings

### 1. Data Source Impact
"""
    baseline_results = [r for r in results if 'baseline' in r.config_name]
    full_results = [r for r in results if 'full' in r.config_name]

    if baseline_results and full_results:
        avg_baseline_chunks = np.mean([r.num_chunks for r in baseline_results])
        avg_full_chunks = np.mean([r.num_chunks for r in full_results])
        md_content += f"- Baseline data average chunks: {avg_baseline_chunks:.0f}\n"
        md_content += f"- Full data average chunks: {avg_full_chunks:.0f}\n"
        md_content += f"- Chunk increase: {((avg_full_chunks/avg_baseline_chunks)-1)*100:.1f}%\n"

    md_content += """
### 2. Retriever Type Comparison
"""
    for rt in ['dense', 'sparse', 'hybrid']:
        rt_results = [r for r in results if rt in r.config_name]
        if rt_results:
            avg_time = np.mean([r.avg_retrieval_time * 1000 for r in rt_results])
            md_content += f"- {rt.capitalize()} retriever avg time: {avg_time:.2f}ms\n"

    md_content += """
### 3. Top-K Impact
"""
    topk_results = [r for r in results if 'topk' in r.config_name]
    for r in topk_results:
        md_content += f"- {r.config_name}: {r.avg_retrieval_time*1000:.2f}ms retrieval\n"

    # Save markdown
    with open(output_path / 'experiment_stats.md', 'w') as f:
        f.write(md_content)

    # Save JSON for programmatic access
    stats_json = {
        "results": [asdict(r) for r in results],
        "summary": {
            "total_experiments": len(results),
            "avg_retrieval_time_ms": np.mean([r.avg_retrieval_time * 1000 for r in results]),
            "avg_generation_time_s": np.mean([r.avg_generation_time for r in results]),
        }
    }

    with open(output_path / 'experiment_stats.json', 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"Stats saved to {output_path}/experiment_stats.md")
    print(f"JSON saved to {output_path}/experiment_stats.json")


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific configs to run (default: all)')
    parser.add_argument('--queries', default='leaderboard_queries.json',
                        help='Path to queries file')
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='LLM model for generation')
    parser.add_argument('--output-dir', default='experiment_outputs',
                        help='Directory for experiment outputs')
    parser.add_argument('--figures-dir', default='report_figures',
                        help='Directory for report figures')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip answer generation (retrieval only)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick subset of experiments')
    args = parser.parse_args()

    # Load queries
    with open(args.queries, 'r') as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries")

    # Get experiment configs
    all_configs = get_experiment_configs()

    if args.quick:
        # Run only key experiments for quick testing
        quick_names = ['baseline_hybrid_rrf', 'full_hybrid_rrf', 'full_dense', 'full_sparse']
        all_configs = [c for c in all_configs if c.name in quick_names]

    if args.configs:
        all_configs = [c for c in all_configs if c.name in args.configs]

    print(f"Running {len(all_configs)} experiments")

    # Initialize generator (shared across experiments)
    if not args.skip_generation:
        print(f"\nLoading generator model: {args.model}")
        generator = SimpleGenerator(args.model)
    else:
        generator = None

    # Run experiments
    results = []
    for config in all_configs:
        try:
            result = run_experiment(
                config,
                queries,
                generator,
                args.output_dir
            )
            results.append(result)
            print(f"  Completed: {result.output_file}")
        except Exception as e:
            print(f"  ERROR in {config.name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate reports
    if results:
        print("\n" + "="*60)
        print("Generating report materials...")
        print("="*60)

        generate_report_charts(results, args.figures_dir)
        generate_stats_table(results, args.figures_dir)

        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for r in results:
            print(f"\n{r.config_name}:")
            print(f"  Chunks: {r.num_chunks}")
            print(f"  Indexing: {r.indexing_time:.2f}s")
            print(f"  Avg Retrieval: {r.avg_retrieval_time*1000:.2f}ms")
            print(f"  Avg Generation: {r.avg_generation_time:.2f}s")
            print(f"  Output: {r.output_file}")


if __name__ == "__main__":
    main()
