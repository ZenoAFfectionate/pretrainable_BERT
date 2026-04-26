# BERT & DeBERTa 从零实现 · 预训练 · SQuAD 微调

本仓库从零实现了 **BERT** 和 **DeBERTa**，共享统一的数据管道、训练器和日志系统，并在 **SQuAD v1.1** 抽取式问答任务上完成了下游微调与评估。三个阶段（预训练 / 微调 / 评估）由 [`main.py`](main.py) 统一调度。

> - **BERT**: Pre-training of Deep Bidirectional Transformers for Language Understanding — https://arxiv.org/abs/1810.04805
> - **DeBERTa**: Decoding-enhanced BERT with Disentangled Attention — https://arxiv.org/abs/2006.03654

---

## 目录

1. [项目概览](#项目概览)
2. [模型介绍：BERT vs DeBERTa](#模型介绍bert-vs-deberta)
3. [项目结构](#项目结构)
4. [环境安装](#环境安装)
5. [完整流程](#完整流程)
    1. [Step 1 · 构建预训练语料](#step-1--构建预训练语料)
    2. [Step 2 · 训练 BPE 分词器](#step-2--训练-bpe-分词器)
    3. [Step 3 · 预训练 BERT / DeBERTa](#step-3--预训练-bert--deberta)
    4. [Step 4 · 下载并微调 SQuAD v1.1](#step-4--下载并微调-squad-v11)
    5. [Step 5 · 评估微调后的模型](#step-5--评估微调后的模型)
6. [使用 YAML 配置 · 单卡复现论文](#使用-yaml-配置--单卡复现论文)
7. [Shell 快捷脚本](#shell-快捷脚本)
8. [模型超参数配置](#模型超参数配置)
9. [数据集介绍](#数据集介绍)
10. [实验结果](#实验结果)
11. [输出目录结构](#输出目录结构)
12. [常见问题](#常见问题)

---

## 项目概览

| 阶段 | 入口 | 作用 |
| --- | --- | --- |
| Corpus build | [`dataset/build_corpus.py`](dataset/build_corpus.py) | 从 HF Wikipedia + BookCorpus 流式拉取，生成 `sentence1 \t sentence2` 对 |
| Tokenizer train | [`dataset/build_tokenizer.py`](dataset/build_tokenizer.py) | 在语料上训练 Byte-Level BPE，得到 `tokenizer.json` |
| Pre-train | [`pretrain.py`](pretrain.py) | BERT / DeBERTa 预训练（MLM + NSP） |
| Fine-tune | [`finetune.py`](finetune.py) | 在 SQuAD v1.1 上微调 QA head |
| Evaluate | [`evaluate.py`](evaluate.py) | 加载微调 checkpoint，报告 EM / F1 |
| Unified CLI | [`main.py`](main.py) | `python main.py {pretrain,finetune,evaluate} ...` |

共享基础设施：

- **分词**：HuggingFace Byte-Level BPE，特殊 token 固定在 `0:<pad>` `1:<unk>` `2:<eos>` `3:<sos>` `4:<mask>`
- **输入格式**：`[SOS] 句1 [EOS] 句2 [EOS] <pad>...`，`segment_ids` 分别为 1 / 2
- **优化**：AdamW + 线性 warmup-then-decay，可选 FP16 混合精度
- **日志**：统一由 `utils.TrainingLogger` 写 `config.json` / `training.log` / `metrics.jsonl` / `checkpoints/`

---

## 模型介绍：BERT vs DeBERTa

两种模型共用同一个 `BERTEmbedding`（token + absolute position + segment）和同一个预训练目标（MLM + NSP），区别集中在 **Transformer 块内的注意力** 与 **MLM 解码头**：

| 模块 | BERT | DeBERTa |
| --- | --- | --- |
| 自注意力 | 标准多头自注意力 | **Disentangled Self-Attention**：`content-to-content` + `content-to-position` + `position-to-content` 三项相加，缩放因子 `1/√(3·d_head)` |
| 位置编码 | 仅 absolute（加在 embedding 上） | absolute + **relative**（`pos_q_proj` / `pos_k_proj` + 共享的相对位置 embedding `[2k+1, H]`） |
| MLM 解码头 | 单层 `Linear(hidden → vocab)` | **Enhanced Mask Decoder (EMD)**：`LayerNorm(ctx + abs_pos) → Dense+GELU → Linear(vocab)`，把绝对位置重新注入最后一层 |
| NSP 头 | 两类分类头（共享） | 同左 |
| 参数开销 | 基准 | 每层多 2 组位置投影 + 相对位置表，约 +3–5% 参数 |

代码位置：[model/attention.py](model/attention.py) 定义两种注意力，[model/model.py](model/model.py) 组装 `BERT` / `DeBERTa` encoder 与 `BERTLM` / `DeBERTaLM` 预训练包装。

---

## 项目结构

```
BERT/
├── main.py                      # 统一 CLI（pretrain/finetune/evaluate）
├── pretrain.py                  # BERT/DeBERTa 预训练（选 --model）
├── finetune.py                  # SQuAD v1.1 微调
├── evaluate.py                  # 微调后模型的 EM/F1 评估
│
├── model/
│   ├── model.py                 # BERT / DeBERTa encoder, BERTLM / DeBERTaLM
│   ├── attention.py             # MultiHeadAttention + DisentangledSelfAttention
│   ├── utils.py                 # SublayerConnection / FFN / BERTEmbedding / ...
│   └── qa.py                    # QAModel（encoder + 2-way 起止头）+ ckpt 载入
│
├── dataset/
│   ├── build_corpus.py          # 从 HF 拉 Wikipedia + BookCorpus 流式生成语料
│   ├── build_tokenizer.py       # 训练 BPE tokenizer
│   ├── dataset.py               # BERTDataset（NSP+MLM 采样）
│   ├── vocab.py                 # BPEVocab（封装 HF tokenizer）
│   ├── squad.py                 # SQuAD 加载 / 特征化 / EM+F1 官方打分
│   └── data/
│       ├── corpus/              # train.txt / test.txt
│       └── tokenizer/           # tokenizer.json
│
├── utils/
│   ├── trainer.py               # BERTTrainer / DeBERTaTrainer（共享基类）
│   ├── logger.py                # TrainingLogger（日志/指标/配置/checkpoint）
│   ├── scheduler.py             # warmup + linear decay
│   └── common.py                # 共用小工具
│
├── dataset/tune/squad/          # SQuAD v1.1 原始 JSON（见 Step 4）
├── result/<run>/                # 每次运行一个目录（详见「输出目录结构」）
└── requirements.txt
```

---

## 环境安装

```bash
pip install torch torchvision torchaudio
pip install tokenizers transformers einops tqdm datasets nltk numpy
```

硬件建议：≥ 1 张 24 GB GPU（如 RTX 3090/4090）。预训练 Base 规模需要多卡或长时间训练。

---

## 完整流程

以下示例全部假设在仓库根目录下执行。每条命令都等价于 `python main.py <子命令> ...`，也可以直接 `python pretrain.py` / `finetune.py` / `evaluate.py` 独立调用。

### Step 1 · 构建预训练语料

从 HuggingFace 拉取 [`wikimedia/wikipedia` (20231101.en)](https://huggingface.co/datasets/wikimedia/wikipedia) 与 [`lucadiliello/bookcorpusopen`](https://huggingface.co/datasets/lucadiliello/bookcorpusopen)，切成相邻句对，输出到 `dataset/data/corpus/{train,test}.txt`（格式：`sentence1 \t sentence2` 每行一对）。

```bash
# 全量（约 20 GB、数小时）
python -m dataset.build_corpus

# 快速试跑：只拉 1 万篇 Wiki + 500 本 book
python -m dataset.build_corpus --max_articles 10000 --max_books 500

# 只要 Wiki、不要 BookCorpus
python -m dataset.build_corpus --skip_books --max_articles 50000
```

主要参数：

- `--max_articles / --max_books`：上限，`-1` 代表全量
- `--test_ratio`：切分到 `test.txt` 的比例，默认 `0.005`
- `--skip_wiki / --skip_books`：跳过某一路数据源

### Step 2 · 训练 BPE 分词器

在上一步的语料上训练 Byte-Level BPE，保存到 `dataset/data/tokenizer/tokenizer.json`。特殊 token 会被强制钉在 ids `0–4` 上，确保后续模型的 `padding_idx=0` / `ignore_index=0` 约定不变。

```bash
python -m dataset.build_tokenizer \
    --corpus dataset/data/corpus/train.txt \
    --out dataset/data/tokenizer/tokenizer.json \
    --vocab_size 32000 \
    --min_frequency 2
```

可选 `--no_lowercase` 保留大小写（默认小写）。

### Step 3 · 预训练 BERT / DeBERTa

两种架构共享同一入口，用 `--model` 选择：

```bash
# BERT-Base 预训练
python main.py pretrain --model bert \
    -c dataset/data/corpus/train.txt \
    -t dataset/data/corpus/test.txt \
    -v dataset/data/tokenizer/tokenizer.json \
    --hidden 768 --layers 12 --attn_heads 12 \
    --batch_size 32 --seq_len 512 \
    --epochs 40 --lr 1e-4 --warmup_steps 10000 \
    --fp16 --cuda_devices 1,2 \
    --run_name bert_base
```

```bash
# DeBERTa-Base 预训练（多出 --max_relative_positions）
python main.py pretrain --model deberta \
    -c dataset/data/corpus/train.txt \
    -t dataset/data/corpus/test.txt \
    -v dataset/data/tokenizer/tokenizer.json \
    --hidden 768 --layers 12 --attn_heads 12 \
    --max_relative_positions 512 \
    --batch_size 32 --seq_len 512 \
    --epochs 40 --lr 1e-4 --warmup_steps 10000 \
    --fp16 --cuda_devices 0,1 \
    --run_name deberta_base
```

常用选项（`python main.py pretrain --help` 完整列出）：

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `-m / --model` | `bert` 或 `deberta` | `bert` |
| `-hs / --hidden` | 隐层维度 | 768 |
| `-l / --layers` | Transformer 层数 | 12 |
| `-a / --attn_heads` | 注意力头数 | 12 |
| `--max_relative_positions` | 相对位置窗口（DeBERTa 专用） | 512 |
| `--seq_len` | 最大输入长度 | 512 |
| `--batch_size` | 批大小 | 32 |
| `--epochs` | 训练轮数 | 100 |
| `--lr` | AdamW 学习率 | 1e-4 |
| `--warmup_steps` | warmup 步数 | 10000 |
| `--grad_clip / --weight_decay` | 梯度裁剪 / 权重衰减 | 1.0 / 0.01 |
| `--fp16` | 开启 FP16 | off |
| `--cuda_devices` | 多卡列表，如 `"0,1,2,3"` | 全部 |
| `--resume` | 从某个 ckpt 续训 | — |

最优 checkpoint 会按 train-loss 下降保存到 `result/<run_name>/checkpoints/best_model.pt`，里面包含 `model_state / optimizer_state / scheduler_state / scaler_state / config`，可通过 `--resume` 续训。

### Step 4 · 下载并微调 SQuAD v1.1

先下载 Stanford 官方 JSON（两个文件约 34 MB）：

```bash
mkdir -p dataset/tune/squad
curl -L -o dataset/tune/squad/train-v1.1.json \
    https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
curl -L -o dataset/tune/squad/dev-v1.1.json \
    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

文件规模：train 442 篇文章 / 87,599 问答，dev 48 篇文章 / 10,570 问答（每题 3 条参考答案）。

然后在预训练 checkpoint 上微调。架构会从 checkpoint 的 config 自动识别，无需再指定 `--model`：

```bash
python main.py finetune \
    --checkpoint result/bert_base/checkpoints/best_model.pt \
    --vocab_path dataset/data/tokenizer/tokenizer.json \
    --train_file dataset/tune/squad/train-v1.1.json \
    --dev_file   dataset/tune/squad/dev-v1.1.json \
    --seq_len 384 --max_query_len 64 \
    --epochs 2 --batch_size 16 \
    --lr 3e-5 --warmup_ratio 0.1 \
    --fp16 --cuda_devices 0 \
    --run_name squad_bert
```

对 DeBERTa 只需把 `--checkpoint` 换成 DeBERTa 的预训练产物即可；`finetune.py` 会据此重建 encoder、接入 `QAModel`（encoder + `Linear(hidden, 2)`）并端到端训练。每个 epoch 结束都会在 dev 上评估 EM/F1，并按最佳 F1 保存 `result/<run>/checkpoints/best_model.pt`。

### Step 5 · 评估微调后的模型

```bash
python main.py evaluate \
    --checkpoint result/squad_bert/checkpoints/best_model.pt \
    --vocab_path dataset/data/tokenizer/tokenizer.json \
    --dev_file   dataset/tune/squad/dev-v1.1.json \
    --fp16 --cuda_devices 0
```

`evaluate.py` 只加载、前向、打分、写 `eval_predictions.json`（含每题预测 + EM/F1），不做任何训练。`seq_len` / `max_query_len` / `max_answer_tokens` 默认沿用微调时保存的配置。

---

## 使用 YAML 配置 · 单卡复现论文

手写一长串 `--hidden 768 --layers 12 ...` 容易出错，也难以在实验之间对齐。[`config/`](config/) 下预置了 7 份 YAML，用 `--config <path>` 直接加载；CLI flag 仍然可以局部覆盖 YAML。

### 可用配置

| 文件 | 架构 | `L × H × A` | `seq_len` | `batch × accum` | 有效批次 | 对应论文 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [config/bert_mini.yaml](config/bert_mini.yaml) | BERT | 4 × 128 × 4 | 128 | 64 × 4 | 256 | — |
| [config/bert_base.yaml](config/bert_base.yaml) | BERT | 12 × 768 × 12 | 512 | 32 × 8 | **256** | ✅ BERT-Base |
| [config/bert_pro.yaml](config/bert_pro.yaml) | BERT | 24 × 1024 × 16 | 512 | 16 × 16 | **256** | ✅ BERT-Large |
| [config/deberta_mini.yaml](config/deberta_mini.yaml) | DeBERTa | 4 × 128 × 4 | 128 | 64 × 4 | 256 | — |
| [config/deberta_base.yaml](config/deberta_base.yaml) | DeBERTa | 12 × 768 × 12 | 512 | 32 × 8 | **256** | ✅ DeBERTa-Base |
| [config/deberta_pro.yaml](config/deberta_pro.yaml) | DeBERTa | 24 × 1024 × 16 | 512 | 16 × 16 | **256** | ✅ DeBERTa-Large |
| [config/finetune_squad.yaml](config/finetune_squad.yaml) | 继承 ckpt | — | 384 | 16 × 2 | **32** | ✅ SQuAD 标准 |

所有 `batch × accum` 组合通过**梯度累积**达到论文的 effective batch：micro-batch 级 forward + backward，每累积 N 个 micro-batch 才触发一次 `optimizer.step()` + `scheduler.step()`——效果等价于大 batch，显存占用等于 micro-batch。

### 单卡 48 GB 复现论文

**1 · BERT-Base 预训练（effective batch 256，对齐论文）**

```bash
python main.py pretrain --config config/bert_base.yaml \
    -c dataset/data/corpus/train.txt \
    -t dataset/data/corpus/test.txt \
    -v dataset/data/tokenizer/tokenizer.json \
    --cuda_devices 0 \
    --run_name bert_base_paper
```

自动启用：`hidden=768 × layers=12 × heads=12`，`seq_len=512`，`batch_size=32 × gradient_accumulation_steps=8 = 256`，`lr=1e-4`，`warmup=10000`，`weight_decay=0.01`，`epochs=40`，FP16。

**2 · DeBERTa-Base 预训练（同等配置 + 相对位置）**

```bash
python main.py pretrain --config config/deberta_base.yaml \
    -c dataset/data/corpus/train.txt \
    -t dataset/data/corpus/test.txt \
    -v dataset/data/tokenizer/tokenizer.json \
    --cuda_devices 0 \
    --run_name deberta_base_paper
```

**3 · Large 规模（BERT-Large / DeBERTa-Large，effective batch 256）**

```bash
# 替换 config 即可，其余命令一致
python main.py pretrain --config config/bert_pro.yaml ... --run_name bert_large_paper
python main.py pretrain --config config/deberta_pro.yaml ... --run_name deberta_large_paper
```

`bert_pro.yaml` / `deberta_pro.yaml` 预设 `batch=16 × accum=16 = 256`，在 48 GB 单卡 FP16 下能跑。

**4 · SQuAD v1.1 微调（effective batch 32，对齐 SQuAD 论文）**

```bash
python main.py finetune --config config/finetune_squad.yaml \
    --checkpoint result/bert_base_paper/checkpoints/best_model.pt \
    --vocab_path dataset/data/tokenizer/tokenizer.json \
    --train_file dataset/tune/squad/train-v1.1.json \
    --dev_file   dataset/tune/squad/dev-v1.1.json \
    --cuda_devices 0 \
    --run_name squad_bert_base
```

自动启用：`seq_len=384`，`max_query_len=64`，`batch_size=16 × gradient_accumulation_steps=2 = 32`，`lr=3e-5`，`warmup_ratio=0.1`，`epochs=2`，FP16。DeBERTa 同理——把 `--checkpoint` 指向 DeBERTa 的预训练产物即可（架构从 ckpt 自动识别）。

**5 · 评估微调后的模型**

```bash
python main.py evaluate \
    --checkpoint result/squad_bert_base/checkpoints/best_model.pt \
    --vocab_path dataset/data/tokenizer/tokenizer.json \
    --dev_file   dataset/tune/squad/dev-v1.1.json \
    --fp16 --cuda_devices 0
```

### YAML + CLI 覆盖优先级

```
argparse 硬编码 default  ←  YAML (--config)  ←  CLI flag
```

举例：YAML 定义 `batch_size: 32` 和 `gradient_accumulation_steps: 8`，如果想临时试效果，命令行加 `--batch_size 64 --gradient_accumulation_steps 4`（同样 effective 256，但占显存更多、迭代更快），会覆盖 YAML 对应字段，其余 YAML 值保持不变。

---

## Shell 快捷脚本

[`scripts/`](scripts/) 下预置了三个脚本，把「加载 YAML + 传路径 + 启动」的样板统一封装：

| 脚本 | 用途 | 关键位置参数 |
| --- | --- | --- |
| [`scripts/pretrain.sh`](scripts/pretrain.sh) | 预训练 BERT / DeBERTa | `<bert\|deberta> [mini\|base\|pro]` |
| [`scripts/finetune.sh`](scripts/finetune.sh) | 在 SQuAD v1.1 上微调 | `<checkpoint_path_or_run_name>` |
| [`scripts/evaluate.sh`](scripts/evaluate.sh) | 评估微调模型 | `<checkpoint_path_or_run_name>` |

**共性行为：**

- 自动切换到仓库根目录——从任何位置调用都工作
- 从 `config/` 选对应 YAML，自动填入数据路径（支持环境变量覆盖）
- 第二个位置参数之后所有内容直接透传给 `python main.py`
- `finetune.sh` / `evaluate.sh` 自动从 checkpoint 里识别架构，**不用再传 bert / deberta**

### 常用命令

```bash
# 1. 预训练 BERT-Base（论文对齐）
bash scripts/pretrain.sh bert base

# 2. 预训练 DeBERTa-Large，指定 GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/pretrain.sh deberta pro

# 3. 预训练 BERT-Mini，同时覆盖 epoch 数
bash scripts/pretrain.sh bert mini --epochs 3 --run_name quick_test

# 4. 微调到 SQuAD（传预训练 run 名即可）
bash scripts/finetune.sh bert_base_paper

# 5. 或者直接传 checkpoint 路径
bash scripts/finetune.sh result/deberta_base_paper/checkpoints/best_model.pt

# 6. 评估微调后的模型
bash scripts/evaluate.sh squad_bert_base_paper

# 7. 全链路（依赖上一步的 run_name 约定）
bash scripts/pretrain.sh bert base                 --run_name bert_base_paper
bash scripts/finetune.sh bert_base_paper           --run_name squad_bert_base
bash scripts/evaluate.sh squad_bert_base
```

### 环境变量快捷覆盖

| 变量 | 影响脚本 | 默认值 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 所有 | 系统默认 |
| `CORPUS_TRAIN` / `CORPUS_TEST` | pretrain | `dataset/data/corpus/{train,test}.txt` |
| `VOCAB_PATH` | 所有 | `dataset/data/tokenizer/tokenizer.json` |
| `SQUAD_TRAIN` / `SQUAD_DEV` | finetune / evaluate | `dataset/tune/squad/{train,dev}-v1.1.json` |
| `RUN_NAME` | 所有 | `<命令>_<架构>_<时间戳>` |
| `CONFIG` | finetune | `config/finetune_squad.yaml` |

---

## 模型超参数配置

本项目默认提供三档规模，配合 `--hidden/--layers/--attn_heads` 调节：

| 配置 | `hidden` | `layers` | `attn_heads` | 参数量（BERT） | 参数量（DeBERTa，`k=512`） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small  | 256 | 4  | 4  | ~18 M  | ~20 M |
| Base   | 768 | 12 | 12 | ~110 M | ~132 M |
| Large  | 1024 | 24 | 16 | ~340 M | ~390 M |

建议训练超参（Base 规模作参考）：

| 超参 | 预训练默认 | SQuAD 微调默认 |
| --- | --- | --- |
| `seq_len` | 512 | 384 |
| `batch_size` | 32（按显存调整） | 16 |
| `epochs` | 40–100 | 2–3 |
| `lr` | 1e-4 | 3e-5 |
| warmup | 10000 步（绝对） | 总步数 × 10% |
| `weight_decay` | 0.01 | 0.01 |
| `grad_clip` | 1.0 | 1.0 |
| 精度 | FP16 | FP16 |

---

## 数据集介绍

### 预训练：Wikipedia + BookCorpusOpen

- **Wikipedia**: `wikimedia/wikipedia` 的 `20231101.en` 切片，英文维基百科全量文章。
- **BookCorpusOpen**: `lucadiliello/bookcorpusopen`，开源的小说/叙事类语料，补充 Wikipedia 偏事实型文本的长尾风格。

两者都由 [`build_corpus.py`](dataset/build_corpus.py) 以 HF streaming 方式读取、用 NLTK Punkt 切句、按段内相邻句对输出，为 NSP 提供真实的相邻/随机配对信号。全量语料约 20 GB；可通过 `--max_articles / --max_books` 截断。

### 下游任务：SQuAD v1.1

- **任务**：给定问题和段落，从段落中抽取连续的文本跨度作为答案。
- **规模**：train 87,599 问答 / dev 10,570 问答（测试集标签不公开）。
- **为什么契合本项目**：输入是 `[SOS] 问题 [EOS] 段落 [EOS]`，和本项目预训练阶段 `[SOS] 句1 [EOS] 句2 [EOS]` 的两段式 + Segment Embedding 完全一致，`NextSentencePrediction` 学到的句对交互能力可以直接迁移；QA head 只新增 `Linear(hidden, 2)` 用于 start / end 位置打分，训练目标是两路交叉熵的平均。

---

## 实验结果

> ⚠️ 以下表格为**模板**。由于实际训练成本较高（Base 规模预训练在单卡需要数十小时），请在完成真实训练后把 `TBD` 替换为对应指标。metrics.jsonl 与 `eval_predictions.json` 中都能直接读出所需数值。

### 表 1 · 预训练阶段指标（train / test split）

| 模型 | 参数量 | Epochs | 最终 MLM Acc ↑ | 最终 NSP Acc ↑ | 最终 Loss ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT-Base    | ~110 M | TBD | TBD % | TBD % | TBD |
| DeBERTa-Base | ~132 M | TBD | TBD % | TBD % | TBD |

> 指标来源：`result/<run>/metrics.jsonl` 中 `event="epoch"`、`split="test"` 的最后一条记录。

### 表 2 · 预训练模型在 SQuAD 上直接评估（不微调，上限）

仅用随机初始化的 QA head 跑一次 dev（`finetune.py --epochs 0` 可近似得到），用于衡量「预训练表示单独能解出多少 SQuAD」。预期接近随机。

| 模型 | EM ↑ | F1 ↑ |
| --- | ---: | ---: |
| BERT-Base（仅预训练） | TBD | TBD |
| DeBERTa-Base（仅预训练） | TBD | TBD |

### 表 3 · SQuAD v1.1 dev · 微调后

| 模型 | 微调 Epochs | LR | Batch | Dev EM ↑ | Dev F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT-Base    | 2 | 3e-5 | 16 | TBD | TBD |
| DeBERTa-Base | 2 | 3e-5 | 16 | TBD | TBD |

> 参考：公开 BERT-Base 在 SQuAD v1.1 dev 上约为 EM 80 / F1 88，DeBERTa-Base 约为 EM 86 / F1 93；本项目从零预训练所用语料量远少于原论文，实际指标会更低，但趋势上 DeBERTa 应优于 BERT。

### 结果可视化

所有指标都落在 `result/<run>/metrics.jsonl`，每行一条 JSON。用 `jq` 或 pandas 很容易画出 loss 曲线、MLM/NSP acc 曲线、dev F1 随 epoch 变化等。示例：

```python
import json, pandas as pd
rows = [json.loads(l) for l in open("result/bert_base/metrics.jsonl")]
df = pd.DataFrame([r for r in rows if r["event"] == "epoch"])
print(df[df.split == "test"][["epoch", "avg_loss", "mlm_acc", "nsp_acc"]])
```

---

## 输出目录结构

每次运行都会在 `result/` 下新建一个独立子目录：

```
result/<run_name>/
├── config.json              # 本次运行的完整参数快照（training_args + model_args）
├── training.log             # 人类可读日志（与 stderr 一致）
├── metrics.jsonl            # 每个 step / epoch 一条 JSON（机器可读）
├── checkpoints/
│   ├── best_model.pt        # 最优 checkpoint（按 loss / F1）
│   └── last_model.pt        # 最后一次 epoch 的 checkpoint（仅 finetune）
└── eval_predictions.json    # 仅 evaluate 阶段：metrics + 每题预测
```

每个 checkpoint 都是一个字典，关键字段：

- `model_state` — 权重
- `config` — 重建模型所需的全部超参（`hidden` / `layers` / `attn_heads` / `max_relative_positions` / `model_type` / `seq_len` / ...）
- `epoch` / `step` / `best_loss` — 训练进度

`finetune.py` 和 `evaluate.py` 都能从这一份 dict 完整重建 `QAModel` 并验证 key 对齐（加载日志会报告 `missing=0, unexpected=0`）。

---

## 常见问题

**Q: 显存不够怎么办？**
A: 开 `--fp16`；减 `--batch_size`；减 `--seq_len`；或者先用 Small 配置跑通流程。

**Q: 数据下载太慢？**
A: `build_corpus.py` 用 HF streaming 不落盘，但国内可能需要走镜像。先 `export HF_ENDPOINT=https://hf-mirror.com`，或手动下载数据后改脚本读本地路径。

**Q: `evaluate.py` 报 `max_relative_positions` 不匹配？**
A: 说明微调 checkpoint 的 encoder 配置与命令行 override 不一致。不要手动传 `--seq_len` 之类的 override，让 `evaluate.py` 自动读取 checkpoint 里 `config` 即可。

**Q: 可以只跑单卡吗？**
A: 可以，去掉 `--cuda_devices`（走默认单卡）或写 `--cuda_devices 0`。多卡走 `nn.DataParallel`，没做 DDP。

**Q: 预训练可以断点续训吗？**
A: 可以，`--resume result/<run>/checkpoints/best_model.pt`。optimizer / scheduler / scaler / step 都会一并恢复。
