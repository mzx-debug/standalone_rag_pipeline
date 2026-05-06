# Standalone RAG Pipeline

## 概览

两个脚本：
1. `build_index.py` — 对语料库编码并构建 FAISS 索引
2. `standalone_rag_pipeline.py` — 运行 RAG 推理（embedding → 检索 → 生成）

推理阶段的控制参数：
- `b`：query batch size
- `xE`：embedding 设备（`0=CPU`，`1=GPU`）
- `xR`：检索设备（`0=CPU`，`1=GPU`）
- `nprobe`：FAISS 检索搜索深度（默认 `128`）
- `xG`（生成设备）固定为 GPU

## 安装

```bash
pip install -r requirements.txt
```

## 第一步：构建索引

```bash
python build_index.py \
  --corpus-path /path/to/corpus.jsonl \
  --output-dir ./indexes/ \
  --model-path intfloat/e5-large-v2 \
  --batch-size 256 \
  --max-length 512 \
  --pooling-method mean \
  --use-fp16 \
  --faiss-type Flat
```

主要参数：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--corpus-path` | 语料库文件路径（JSONL/JSON）或 HuggingFace dataset id | （必填） |
| `--output-dir` | 索引保存目录，输出 `faiss.index` | （必填） |
| `--model-path` | Embedding 模型 | `intfloat/e5-large-v2` |
| `--batch-size` | 编码 batch size | `256` |
| `--max-length` | 最大 token 长度 | `512` |
| `--pooling-method` | `mean` 或 `cls` | `mean` |
| `--use-fp16` | 使用 FP16 编码 | 关闭 |
| `--faiss-type` | FAISS 索引类型（如 `Flat`、`IVF4096,Flat`） | `Flat` |
| `--faiss-gpu` | 使用 GPU 构建索引 | 关闭 |
| `--save-embeddings` | 同时保存原始 embedding 为 `.npy` 文件 | 关闭 |
| `--device` | 编码设备（`cuda`/`cpu`），不指定则自动检测 | 自动 |

语料库格式：JSONL，每行包含 `contents` 字段，或 `title` + `text` 字段。

## 第二步：运行 RAG 推理

```bash
python standalone_rag_pipeline.py \
  --index-path ./indexes/faiss.index \
  --corpus-path /path/to/corpus.jsonl \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --queries-file ./queries.jsonl \ 
  --b 64 \
  --xE 1 \
  --xR 0 \
  --output-json ./output/run_summary.json
```

参数参考：
```bash
python standalone_rag_pipeline.py \
  --index-path ./indexes/faiss.index \       # FAISS 索引路径（必填）
  --corpus-path /path/to/corpus.jsonl \      # 语料库路径（必填）
  --generator-model meta-llama/Llama-3.1-8B-Instruct \  # 生成模型（必填）
  --b 64 \                                   # batch size（必填）
  --xE 1 \                                   # embedding 设备 0=CPU 1=GPU（必填）
  --xR 0 \                                   # 检索设备 0=CPU 1=GPU（必填）
  --nprobe 128 \                             # FAISS 搜索深度（默认 128）
  --topk 1 \                                 # 每个 query 检索文档数（默认 1）
  --embedding-model intfloat/e5-large-v2 \   # embedding 模型（默认 e5-large-v2）
  --embedding-max-length 512 \               # embedding 最大长度（默认 512）
  --pooling-method mean \                    # pooling 方式（默认 mean）
  --embedding-use-fp16 \                     # embedding 使用 FP16
  --max-output-len 128 \                     # 生成最大 token 数（默认 128）
  --temperature 0.0 \                        # 生成温度（默认 0.0）
  --prompt-template "Question: {query}\nContext: {context}\nAnswer:" \
  --tensor-parallel-size 1 \                 # vLLM tensor 并行数（默认 1）
  --gpu-memory-utilization 0.8 \             # vLLM 显存利用率（默认 0.8）
  --gpu-id 0 \                               # GPU 编号（默认 0）
  --queries-file ./queries.jsonl \           # query 文件（可选）
  --query-field question \                   # query 字段名（默认 question）
  --sample-queries 256 \                     # 采样 query 数（默认 256）
  --output-json ./output/run_summary.json    # 输出 JSON 路径（可选）
```

## 输出

推理脚本输出 JSON 摘要（可通过 `--output-json` 写入文件），包含：
- 各阶段平均延迟（embedding / 检索 / 生成）
- 总延迟和吞吐量（QPS）
- 生成 token 统计
- 每个 batch 的计时记录
- 示例输出
