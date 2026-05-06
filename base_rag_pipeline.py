#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_primary_gpu_id(raw_gpu_id: Optional[Any]) -> int:
    if raw_gpu_id is None:
        return 0
    if isinstance(raw_gpu_id, int):
        return raw_gpu_id

    text = str(raw_gpu_id).strip()
    if not text:
        return 0
    first_token = text.split(",")[0].strip()
    if not first_token:
        return 0
    return int(first_token)


def synchronize_cuda_if_needed(device_index: Optional[int] = None) -> None:
    if not torch.cuda.is_available():
        return
    if device_index is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device=device_index)


def pooling(
    pooler_output: Optional[torch.Tensor],
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_method: str,
) -> torch.Tensor:
    if pooling_method == "mean":
        masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        if pooler_output is None:
            raise ValueError("pooler_output is None, but pooling_method='pooler'.")
        return pooler_output
    raise ValueError(f"Unsupported pooling method: {pooling_method}")


def extract_question(record: Dict[str, Any], preferred_field: str) -> Optional[str]:
    if preferred_field in record:
        value = record.get(preferred_field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for field in ("question", "query", "question_text", "text"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_field in ("text", "question"):
                nested_value = value.get(nested_field)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return None


def extract_doc_text(doc: Dict[str, Any]) -> str:
    if "contents" in doc and doc["contents"] is not None:
        return str(doc["contents"])
    if "text" in doc and doc["text"] is not None:
        if "title" in doc and doc["title"]:
            return f"{doc['title']}\n{doc['text']}"
        return str(doc["text"])
    if "title" in doc and doc["title"] is not None:
        return str(doc["title"])
    return json.dumps(doc, ensure_ascii=False)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_queries(args: argparse.Namespace) -> List[str]:
    if args.queries_file is not None:
        query_path = Path(args.queries_file).expanduser().resolve()
        if not query_path.is_file():
            raise FileNotFoundError(f"queries_file not found: {query_path}")
        suffix = query_path.suffix.lower()

        if suffix == ".txt":
            with query_path.open("r", encoding="utf-8") as handle:
                queries = [line.strip() for line in handle if line.strip()]
        elif suffix in (".jsonl", ".json"):
            if suffix == ".jsonl":
                records = load_jsonl(query_path)
            else:
                content = json.loads(query_path.read_text(encoding="utf-8"))
                if isinstance(content, list):
                    records = content
                elif isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                    records = content["data"]
                else:
                    raise ValueError("JSON query file must be a list or a dict with a 'data' list.")

            queries = []
            for record in records:
                if isinstance(record, str):
                    query = record.strip()
                elif isinstance(record, dict):
                    query = extract_question(record, args.query_field)
                else:
                    query = None
                if query:
                    queries.append(query)
        else:
            raise ValueError("queries_file must be .txt, .jsonl, or .json")
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        queries = []
        for row in dataset:
            if not isinstance(row, dict):
                continue
            query = extract_question(row, args.query_field)
            if query:
                queries.append(query)

    if not queries:
        raise ValueError("No queries loaded.")

    if args.sample_queries is not None and args.sample_queries > 0 and args.sample_queries < len(queries):
        rng = np.random.RandomState(args.seed)
        sampled = rng.choice(len(queries), size=args.sample_queries, replace=False)
        sampled = sorted(sampled.tolist())
        queries = [queries[i] for i in sampled]

    return queries


def load_corpus(corpus_path: str) -> Any:
    path = Path(os.path.expandvars(os.path.expanduser(corpus_path)))
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            return load_jsonl(path)
        if path.suffix.lower() == ".json":
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                return content
            if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                return content["data"]
            raise ValueError("JSON corpus file must be a list or a dict with a 'data' list.")
        return load_dataset("json", data_files=str(path), split="train")

    dataset = load_dataset(corpus_path)
    if "train" in dataset:
        return dataset["train"]
    first_split = next(iter(dataset.keys()))
    return dataset[first_split]


class QueryEmbeddingStage:
    def __init__(
        self,
        model_path: str,
        pooling_method: str,
        max_length: int,
        backend: str,
        use_fp16: bool,
        gpu_id: int,
    ) -> None:
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.backend = backend
        self.cuda_device_index: Optional[int] = None

        if backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("xE=1 requires CUDA, but CUDA is not available.")
            self.device = torch.device("cuda")
            self.cuda_device_index = gpu_id
        elif backend == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        if self.device.type == "cuda" and use_fp16:
            self.model = self.model.half()
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, queries: Sequence[str]) -> Tuple[np.ndarray, float]:
        synchronize_cuda_if_needed(self.cuda_device_index)
        start = time.perf_counter()

        inputs = self.tokenizer(
            list(queries),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs, return_dict=True)
        embeddings = pooling(
            pooler_output=getattr(outputs, "pooler_output", None),
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=inputs["attention_mask"],
            pooling_method=self.pooling_method,
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32, order="C")

        synchronize_cuda_if_needed(self.cuda_device_index)
        return embeddings_np, time.perf_counter() - start


class RetrievalStage:
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        topk: int,
        nprobe: int,
        backend: str,
        gpu_id: int,
    ) -> None:
        import faiss

        self.faiss = faiss
        self.topk = int(topk)
        self.backend = backend
        self.cuda_device_index: Optional[int] = None
        self.gpu_resources = None

        cpu_index = self.faiss.read_index(index_path)
        if hasattr(cpu_index, "nprobe"):
            cpu_index.nprobe = int(nprobe)

        if backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("xR=1 requires CUDA, but CUDA is not available.")
            if not hasattr(self.faiss, "StandardGpuResources"):
                raise RuntimeError("FAISS GPU APIs unavailable. Install faiss-gpu for xR=1.")
            self.cuda_device_index = gpu_id
            self.gpu_resources = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, cpu_index)
        elif backend == "cpu":
            self.index = cpu_index
        else:
            raise ValueError(f"Unknown retrieval backend: {backend}")

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = int(nprobe)

        self.corpus = load_corpus(corpus_path)

    def __call__(self, embeddings: np.ndarray) -> Tuple[List[List[str]], float]:
        synchronize_cuda_if_needed(self.cuda_device_index)
        start = time.perf_counter()
        _, indices = self.index.search(embeddings, self.topk)

        retrieved_docs: List[List[str]] = []
        corpus_size = len(self.corpus)
        for row in indices:
            docs: List[str] = []
            for idx in row:
                if idx < 0 or idx >= corpus_size:
                    continue
                doc = self.corpus[int(idx)]
                docs.append(extract_doc_text(doc))
            retrieved_docs.append(docs)

        synchronize_cuda_if_needed(self.cuda_device_index)
        return retrieved_docs, time.perf_counter() - start


class GenerationStage:
    def __init__(
        self,
        model_path: str,
        prompt_template: str,
        max_output_len: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        max_model_len: Optional[int],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("xG is fixed to GPU in this pipeline, but CUDA is not available.")

        from vllm import LLM, SamplingParams

        self.prompt_template = prompt_template
        self.max_output_len = max_output_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.SamplingParams = SamplingParams
        self._fallback_tokenizer = None

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": bool(enforce_eager),
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)

        self.llm = LLM(**llm_kwargs)
        self._fallback_tokenizer = self.llm.get_tokenizer()

    def __call__(
        self,
        queries: Sequence[str],
        retrieved_docs: Sequence[Sequence[str]],
    ) -> Tuple[List[str], float, List[int]]:
        prompts = []
        for query, docs in zip(queries, retrieved_docs):
            context = "\n".join(docs)
            prompts.append(self.prompt_template.format(query=query, context=context))

        sampling_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_len,
        }
        if self.top_k is not None:
            sampling_kwargs["top_k"] = int(self.top_k)
        sampling_params = self.SamplingParams(**sampling_kwargs)

        synchronize_cuda_if_needed()
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        answers: List[str] = []
        generated_tokens: List[int] = []

        for output in outputs:
            if not output.outputs:
                answers.append("")
                generated_tokens.append(0)
                continue
            first_output = output.outputs[0]
            answers.append(first_output.text)
            token_ids = getattr(first_output, "token_ids", None)
            if token_ids is not None:
                generated_tokens.append(len(token_ids))
            else:
                generated_tokens.append(len(self._fallback_tokenizer.encode(first_output.text)))

        synchronize_cuda_if_needed()
        return answers, time.perf_counter() - start, generated_tokens


@dataclass
class BatchStats:
    batch_index: int
    batch_size: int
    embedding_sec: float
    retrieval_sec: float
    generation_sec: float
    generated_tokens: int


class StandaloneRAGPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = logging.getLogger("standalone_rag_pipeline")
        self.gpu_id = parse_primary_gpu_id(args.gpu_id)

        self.embedding_backend = self._map_binary_backend(args.xE, "xE")
        self.retrieval_backend = self._map_binary_backend(args.xR, "xR")

        self.embedding_stage = QueryEmbeddingStage(
            model_path=args.embedding_model,
            pooling_method=args.pooling_method,
            max_length=args.embedding_max_length,
            backend=self.embedding_backend,
            use_fp16=args.embedding_use_fp16,
            gpu_id=self.gpu_id,
        )
        self.retrieval_stage = RetrievalStage(
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            topk=args.topk,
            nprobe=args.nprobe,
            backend=self.retrieval_backend,
            gpu_id=self.gpu_id,
        )
        self.generation_stage = GenerationStage(
            model_path=args.generator_model,
            prompt_template=args.prompt_template,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager,
            max_model_len=args.max_model_len,
        )
        self.queries = load_queries(args)
        self.logger.info(
            "Loaded %d queries | b=%d xE=%d(%s) xR=%d(%s) nprobe=%d topk=%d",
            len(self.queries),
            args.b,
            args.xE,
            self.embedding_backend,
            args.xR,
            self.retrieval_backend,
            args.nprobe,
            args.topk,
        )

    @staticmethod
    def _map_binary_backend(x: int, label: str) -> str:
        if x not in (0, 1):
            raise ValueError(f"{label} must be 0 or 1.")
        if x == 1:
            if not torch.cuda.is_available():
                raise RuntimeError(f"{label}=1 requires CUDA, but CUDA is not available.")
            return "gpu"
        return "cpu"

    def run(self) -> Dict[str, Any]:
        total_embedding_sec = 0.0
        total_retrieval_sec = 0.0
        total_generation_sec = 0.0
        total_generated_tokens = 0
        records: List[BatchStats] = []
        samples: List[Dict[str, str]] = []

        batch_size = int(self.args.b)
        if batch_size <= 0:
            raise ValueError("b must be positive.")

        for batch_index, start in enumerate(range(0, len(self.queries), batch_size), start=1):
            batch_queries = self.queries[start : start + batch_size]
            embeddings, embedding_sec = self.embedding_stage(batch_queries)
            retrieved_docs, retrieval_sec = self.retrieval_stage(embeddings)
            answers, generation_sec, token_counts = self.generation_stage(batch_queries, retrieved_docs)

            total_embedding_sec += embedding_sec
            total_retrieval_sec += retrieval_sec
            total_generation_sec += generation_sec
            batch_tokens = int(sum(token_counts))
            total_generated_tokens += batch_tokens

            records.append(
                BatchStats(
                    batch_index=batch_index,
                    batch_size=len(batch_queries),
                    embedding_sec=embedding_sec,
                    retrieval_sec=retrieval_sec,
                    generation_sec=generation_sec,
                    generated_tokens=batch_tokens,
                )
            )

            if len(samples) < self.args.show_samples:
                for query, docs, answer in zip(batch_queries, retrieved_docs, answers):
                    samples.append(
                        {
                            "query": query,
                            "top_doc_snippet": (docs[0][:300] if docs else ""),
                            "answer": answer,
                        }
                    )
                    if len(samples) >= self.args.show_samples:
                        break

            if batch_index % self.args.log_interval == 0:
                self.logger.info(
                    "Processed batch %d, queries=%d",
                    batch_index,
                    min(start + batch_size, len(self.queries)),
                )

        total_sec = total_embedding_sec + total_retrieval_sec + total_generation_sec
        total_queries = len(self.queries)
        summary = {
            "config": {
                "b": self.args.b,
                "xE": self.args.xE,
                "xR": self.args.xR,
                "xG": 1,
                "nprobe": self.args.nprobe,
                "topk": self.args.topk,
                "embedding_model": self.args.embedding_model,
                "generator_model": self.args.generator_model,
            },
            "num_queries": total_queries,
            "avg_embedding_ms": (total_embedding_sec * 1000.0 / total_queries) if total_queries else 0.0,
            "avg_retrieval_ms": (total_retrieval_sec * 1000.0 / total_queries) if total_queries else 0.0,
            "avg_generation_ms": (total_generation_sec * 1000.0 / total_queries) if total_queries else 0.0,
            "total_ms": total_sec * 1000.0,
            "throughput_qps": (total_queries / total_sec) if total_sec > 0 else float("inf"),
            "total_generated_tokens": total_generated_tokens,
            "generation_ms_per_token": (
                (total_generation_sec * 1000.0 / total_generated_tokens)
                if total_generated_tokens > 0
                else float("inf")
            ),
            "samples": samples,
            "per_batch": [record.__dict__ for record in records],
        }
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone RAG pipeline with manual controls: b (batch), xE (embedding device), "
            "xR (retrieval device), nprobe (retrieval search depth). "
            "No dependence on HedraRAG internals."
        )
    )

    parser.add_argument("--index-path", type=str, required=True, help="FAISS index path.")
    parser.add_argument("--corpus-path", type=str, required=True, help="Corpus path or HuggingFace dataset id.")
    parser.add_argument("--generator-model", type=str, required=True, help="Generator model path/id for vLLM.")

    parser.add_argument("--b", type=int, required=True, help="Manual query batch size.")
    parser.add_argument("--xE", type=int, choices=[0, 1], required=True, help="Embedding device: 0=CPU, 1=GPU.")
    parser.add_argument("--xR", type=int, choices=[0, 1], required=True, help="Retrieval device: 0=CPU, 1=GPU.")
    parser.add_argument("--nprobe", type=int, default=128, help="Retrieval search depth (default: 128).")
    parser.add_argument("--topk", type=int, default=1, help="Number of retrieved docs per query.")

    parser.add_argument("--embedding-model", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--pooling-method", type=str, default="mean", choices=["mean", "cls", "pooler"])
    parser.add_argument("--embedding-use-fp16", action="store_true")

    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Question: {query}\nContext: {context}\nAnswer:",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--gpu-id", type=str, default="0")

    parser.add_argument("--queries-file", type=str, default=None, help="Optional .txt/.jsonl/.json query file.")
    parser.add_argument("--query-field", type=str, default="question")
    parser.add_argument("--dataset-name", type=str, default="natural_questions")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--sample-queries", type=int, default=256)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_random_seed(args.seed)
    pipeline = StandaloneRAGPipeline(args)
    summary = pipeline.run()

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Saved summary to %s", output_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
