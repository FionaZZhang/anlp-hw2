"""
Document processor for loading and chunking HTML documents.
"""
import os
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict


def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[edit\]', '', text)
    return text.strip()


def extract_text_from_html(html_path: str) -> Dict:
    """Extract text content from an HTML file."""
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'lxml')

    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()

    title = soup.title.string if soup.title else Path(html_path).stem
    title = clean_text(title) if title else Path(html_path).stem

    content_div = soup.find('div', {'id': 'mw-content-text'}) or \
                  soup.find('div', {'class': 'mw-parser-output'}) or \
                  soup.find('main') or \
                  soup.find('article') or \
                  soup.body

    if content_div:
        text = content_div.get_text(separator=' ')
    else:
        text = soup.get_text(separator=' ')

    text = clean_text(text)

    return {
        'title': title,
        'text': text,
        'source': html_path
    }


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    Uses sentence-aware chunking to avoid breaking mid-sentence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            overlap_words = chunk_size - overlap
            words = chunk_text.split()
            if len(words) > overlap_words:
                current_chunk = [' '.join(words[-overlap:])]
                current_length = overlap
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_documents(data_dirs, output_path: str) -> List[Dict]:
    """
    Process all HTML documents in one or more directories and save chunked documents.

    Args:
        data_dirs: Single directory path (str) or list of directory paths
        output_path: Path to save processed documents
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    documents = []
    html_files = []

    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            html_files.extend(list(dir_path.glob('*.htm')))
            html_files.extend(list(dir_path.glob('*.html')))

    print(f"Found {len(html_files)} HTML files from {len(data_dirs)} directories")

    for html_path in html_files:
        try:
            doc = extract_text_from_html(str(html_path))

            if len(doc['text']) < 100:
                continue

            chunks = chunk_text(doc['text'])

            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f"{Path(html_path).stem}_{i}",
                    'title': doc['title'],
                    'text': chunk,
                    'source': str(html_path)
                })
        except Exception as e:
            print(f"Error processing {html_path}: {e}")

    print(f"Created {len(documents)} document chunks")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)

    return documents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='baseline_data', help='Directory with HTML files')
    parser.add_argument('--output', default='data/processed/documents.json', help='Output path')
    args = parser.parse_args()

    process_documents(args.data_dir, args.output)
