"""
Answer generator using LLM.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Tuple, Optional


class AnswerGenerator:
    """Generate answers using a language model."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None
    ):
        """
        Initialize the answer generator.

        Args:
            model_name: HuggingFace model name
                - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, fast)
                - microsoft/Phi-3-mini-4k-instruct (3.8B, efficient)
                - mistralai/Mistral-7B-Instruct-v0.2 (7B, strong)
        """
        # Force CPU to avoid MPS issues
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_context(self, retrieved_docs: List[Tuple[Dict, float]], max_context_length: int = 2000) -> str:
        """Format retrieved documents as context."""
        context_parts = []
        current_length = 0

        for doc, score in retrieved_docs:
            text = doc['text']
            if current_length + len(text) > max_context_length:
                remaining = max_context_length - current_length
                if remaining > 100:
                    text = text[:remaining]
                else:
                    break

            context_parts.append(text)
            current_length += len(text)

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Tuple[Dict, float]],
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate an answer given a question and retrieved documents.

        Args:
            question: The question to answer
            retrieved_docs: List of (document, score) tuples
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated answer string
        """
        context = self.format_context(retrieved_docs)

        prompt = f"""You are a helpful assistant that answers questions about Pittsburgh and Carnegie Mellon University based on the provided context. Give concise, factual answers.

Context:
{context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)

        answer = answer.strip()
        for stop in ['\n\n', '\nQuestion:', '\nContext:']:
            if stop in answer:
                answer = answer.split(stop)[0]

        return answer.strip()


class SimpleGenerator:
    """Simple generator using transformers pipeline (easier setup)."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading model: {model_name}")

        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float32,
                device="cpu"
            )
        except Exception as e:
            print(f"Error loading with pipeline, trying direct model load: {e}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            ).to(self.device)
            self.pipe = None
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_context(self, retrieved_docs: List[Tuple[Dict, float]], max_context_length: int = 2000) -> str:
        """Format retrieved documents as context."""
        context_parts = []
        current_length = 0

        for doc, score in retrieved_docs:
            text = doc['text']
            if current_length + len(text) > max_context_length:
                remaining = max_context_length - current_length
                if remaining > 100:
                    text = text[:remaining]
                else:
                    break
            context_parts.append(text)
            current_length += len(text)

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Tuple[Dict, float]],
        max_new_tokens: int = 100
    ) -> str:
        """Generate an answer given a question and retrieved documents."""
        context = self.format_context(retrieved_docs)

        if self.pipe is not None:
            # Use pipeline
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions about Pittsburgh and Carnegie Mellon University. Give concise, factual answers based only on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a brief, direct answer:"}
            ]

            try:
                outputs = self.pipe(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False
                )

                answer = outputs[0]['generated_text']
                if isinstance(answer, list):
                    answer = answer[0].get('content', str(answer))
            except Exception as e:
                prompt = f"<|system|>You are a helpful assistant.</s><|user|>Context:\n{context}\n\nQuestion: {question}\n\nProvide a brief answer:</s><|assistant|>"
                outputs = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
                answer = outputs[0]['generated_text']
        else:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        answer = answer.strip()
        for stop in ['\n\n', '\nQuestion:', '\nContext:']:
            if stop in answer:
                answer = answer.split(stop)[0].strip()

        return answer


if __name__ == "__main__":
    generator = SimpleGenerator()

    test_docs = [
        ({"text": "Carnegie Mellon University was founded in 1900 by Andrew Carnegie as the Carnegie Technical Schools."}, 0.9),
        ({"text": "The university is located in Pittsburgh, Pennsylvania."}, 0.8)
    ]

    answer = generator.generate_answer("When was Carnegie Mellon University founded?", test_docs)
    print(f"Answer: {answer}")
