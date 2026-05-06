<<<<<<< HEAD
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
=======
# BaseRAGpipeline



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin http://202.120.38.140:14022/rag-pipeline/baseragpipeline.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](http://202.120.38.140:14022/rag-pipeline/baseragpipeline/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
>>>>>>> ddd9ac231f123dc28b597910d7757075fc307f85
