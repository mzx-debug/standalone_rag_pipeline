# Standalone RAG Pipeline (Manual `b/xE/xR`)

This is an independent RAG pipeline extracted from the experiment design, without importing `HedraRAG` internals.

It supports manual controls:
- `b`: query batch size
- `xE`: embedding device (`0=CPU`, `1=GPU`)
- `xR`: retrieval device (`0=CPU`, `1=GPU`)
- `nprobe`: FAISS retrieval search depth (default `128`)

`xG` (generation device) is fixed to GPU in this implementation.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Minimal Run

```bash
python standalone_rag_pipeline.py \
  --index-path /path/to/ivf.index \
  --corpus-path Tevatron/msmarco-passage-corpus \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --b 64 \
  --xE 1 \
  --xR 0 \
  --nprobe 128 \
  --topk 1 \
  --output-json ./output/run_summary.json
```

## 3. Query Sources

Two modes:
1. Local file via `--queries-file` (`.txt/.jsonl/.json`)
2. HuggingFace dataset via `--dataset-name/--dataset-split` (default `natural_questions/validation`)

If `--queries-file` is not provided, the pipeline loads queries from HuggingFace.

## 4. Key Notes

- No query compression or preprocessing is included.
- Retrieval stage reads FAISS index directly and sets `nprobe`.
- Corpus can be:
  - local JSON/JSONL file
  - HuggingFace dataset id
- Generation uses `vLLM`.

## 5. Output

The script prints a JSON summary and can optionally write it via `--output-json`.

Summary includes:
- per-query average stage latency
- total latency
- throughput
- generated token stats
- per-batch timing records
- a few sample outputs
"# standalone_rag_pipeline" 
