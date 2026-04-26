# Standalone RAG Pipeline 使用说明

本文档说明如何使用 `standalone_rag_pipeline.py` 进行最小测试与日常运行。

## 1. 适用范围

- 三个阶段：
  - Query Embedding
  - Retrieval (FAISS)
  - Generation (vLLM)
- 设备控制参数：
  - `xE`：Embedding 设备（`0=CPU`, `1=GPU`）
  - `xR`：Retrieval 设备（`0=CPU`, `1=GPU`）
  - `xG`：固定为 GPU（脚本内部固定）

## 2. 依赖与环境

## 2.1 Python 依赖

`requirements.txt`：

- `torch>=2.5.0`
- `transformers>=4.40.0`
- `datasets>=3.0.0`
- `faiss-cpu`
- `vllm>=0.6.0`
- `numpy>=1.24.0`

## 2.2 运行资源

你需要准备：

- FAISS 索引文件：`--index-path`
- 语料文件或语料数据集：`--corpus-path`
- 生成模型：`--generator-model`
- 查询文件（可选）：`--queries-file`

注意：生成阶段使用 vLLM，必须有可用 CUDA。

## 3. 安装步骤（Linux）

```bash
cd /目标文件夹
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

退出环境：

```bash
deactivate
```

下次进入：

```bash
cd /目标文件夹
source .venv/bin/activate
```

## 4. 最小冒烟测试

## 4.1 创建最小 query 集

在当前目录创建 `queries.txt`：

```bash
cat > queries.txt << 'EOF'
What is photosynthesis?
EOF
```

## 4.2 运行命令（你当前这套路径）

```bash
python standalone_rag_pipeline.py \
  --index-path /data/home/mazhenxiang/Hedra-RAG-EXP/HedraRAG/data/index_0319/msacro/IVF4096/ivf.index \
  --corpus-path /data/home/mazhenxiang/Hedra-RAG-EXP/data/msmarco_2k.jsonl \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --queries-file ./queries.txt \
  --b 1 \
  --xE 1 \
  --xR 0 \
  --topk 1 \
  --max-output-len 16 \
  --output-json ./rag_smoke_min.json
```

查看输出：

```bash
ls -lh ./rag_smoke_min.json
cat ./rag_smoke_min.json
```

## 5. Query 导入方式

支持三种 `--queries-file`：

- `.txt`：每行一条 query
- `.jsonl`：每行一个 JSON 记录
- `.json`：可以是数组，或 `{ "data": [...] }`

默认字段优先读取 `question`。如果字段名不同，使用：

```bash
--query-field query
```

如果不传 `--queries-file`，脚本会从 HuggingFace 数据集加载（默认 `natural_questions/validation`）。

## 6. 常用参数

- `--b`：query batch size
- `--xE`：embedding 设备（`0/1`）
- `--xR`：retrieval 设备（`0/1`）
- `--nprobe`：FAISS 检索深度，默认 `128`
- `--topk`：每条 query 取回文档数，默认 `1`
- `--max-output-len`：生成最大 token 数
- `--output-json`：保存结果摘要路径

## 7. 常见问题

## 7.1 403 Gated Repo

报错示例：`Cannot access gated repo ... meta-llama/Llama-3.1-8B-Instruct`

原因：当前 HuggingFace 账号无权限访问 gated 模型。

处理方式：

- 申请并通过该模型访问权限，然后 `huggingface-cli login`
- 或先换非 gated 模型做测试，例如 `Qwen/Qwen2.5-1.5B-Instruct`

## 7.2 `top_doc_snippet` 为空

常见原因：`index` 与 `corpus` 不是同一套构建数据（文档 ID 不对齐）。

建议：使用构建该索引时的原始 corpus 做对齐测试。

## 7.3 NCCL 退出警告

`destroy_process_group() was not called ...` 常见于进程退出阶段，冒烟测试可先忽略；若长期运行再考虑在框架层补优雅退出。

## 8. 结果说明（输出 JSON）

关键字段：

- `num_queries`：本次处理 query 数
- `avg_embedding_ms` / `avg_retrieval_ms` / `avg_generation_ms`：三阶段平均耗时
- `throughput_qps`：吞吐（query/s）
- `samples`：样例 query、检索片段和答案
- `per_batch`：每批次详细耗时

