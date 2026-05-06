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

需要准备：

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

## 4.2 运行命令

```bash
python standalone_rag_pipeline.py \
  --index-path /索引路径
  --corpus-path /语料路径
  --generator-model /大模型名
  --queries-file /问题路径
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



