# CMAO

Correct-Mode Advantage Optimization 的离线实验代码，目前已经进入第二阶段：从“方法有信号”的原型，升级到“能做论文级离线证据分析”的实验仓库。

这个仓库当前的重点不是直接训练 RL 模型，而是把下面这条证据链做扎实：

1. 对同一道数学题采样多条 CoT
2. 判断每条答案是否正确
3. 给正确解内部的推理质量打分
4. 给推理模式打标签
5. 计算 `A_ans`、`A_qual`、`A_mode` 和 `A_total`
6. 比较不同重排策略是否优于直接取第一条回答
7. 导出错误案例、全对组案例和 mode 分析案例

当前默认主模型是 `Qwen/Qwen2.5-Math-1.5B-Instruct`，默认先在数学数据集上做离线验证。

## 仓库结构

```text
src/cmao/                 核心实现
configs/experiment/       实验配置
configs/model/            模型配置
configs/scoring/          打分与 advantage 配置
scripts/                  直接可运行的命令脚本
tests/                    单元测试
outputs/                  运行产物
data/                     本地数据说明与可选数据镜像
```

几个核心模块：

- `src/cmao/answer_judge.py`
  负责答案抽取和正确性判定
- `src/cmao/quality_scorer.py`
  负责 CoT 启发式质量分
- `src/cmao/mode_tagger.py`
  负责规则版推理模式标签 `phi(c)`
- `src/cmao/cmao.py`
  负责 CMAO advantage 计算
- `src/cmao/pipeline.py`
  负责把采样、打分、advantage 和报告串起来
- `src/cmao/cli.py`
  负责命令行入口

## 适合谁先用这版

如果你现在想先验证这些问题，这版就够用了：

- 同一个模型对同一道题能不能产生多个“都答对但质量不同”的解
- 启发式 `r_qual` 能不能在正确解里拉开差异
- `A_mode` 会不会给少见但高质量的推理模式更高权重
- `quality` 或 `A_total` 重排，是否比直接取第一条更好

如果你现在就想做在线 RL、GRPO、LoRA 训练，这个仓库还不是最终形态，但它已经把数据接口、评分接口和诊断接口准备好了。

## 安装

项目要求 Python `>=3.11`。

最简单的本地安装方式：

```bash
pip install -e .
```

如果你也要跑测试：

```bash
pip install -e ".[dev]"
```

当前 `pyproject.toml` 里声明的基础依赖包括：

- `torch`
- `transformers`
- `datasets`
- `sympy`

说明：

- 这几个依赖是为了后续真实采样和数学表达式判定准备的
- 如果你只是先看代码结构，不必立刻把环境全装完
- 如果你打算在 Docker 里跑，建议把这些依赖放进镜像里统一安装

## 快速上手

### 1. 先跑单元测试

这一步不依赖外部模型下载，适合先确认仓库逻辑没问题。

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖了：

- `\boxed{}` 和常见数学答案抽取
- 更严格的最终答案抽取优先级
- 分数与小数等价判定
- mode 标签优先级
- mode 命中证据
- quality 打分证据
- 全对组里 `A_qual != 0`
- 错误样本的 `A_qual = 0`

### 2. 用默认配置跑一轮 GSM8K 快验

默认配置文件在 [configs/experiment/gsm8k_quickstart.json](/home/wyatt/research/CMAO/configs/experiment/gsm8k_quickstart.json)。

它的默认设置是：

- 数据集：`gsm8k`
- split：`test`
- 题数：`50`
- `group_size=8`
- `temperature=0.7`
- `top_p=0.95`
- `max_new_tokens=1024`

完整流程：

```bash
python scripts/sample.py \
  --config configs/experiment/gsm8k_quickstart.json \
  --output outputs/gsm8k_samples.json

python scripts/score.py \
  --input outputs/gsm8k_samples.json \
  --output outputs/gsm8k_scores.json \
  --config configs/scoring/default.json

python scripts/advantage.py \
  --input outputs/gsm8k_scores.json \
  --output outputs/gsm8k_advantages.json \
  --config configs/scoring/default.json

python scripts/rerank_eval.py \
  --input outputs/gsm8k_advantages.json \
  --output outputs/gsm8k_report.json

python scripts/report.py \
  --input outputs/gsm8k_advantages.json \
  --output outputs/gsm8k_report_pretty.json

python scripts/analyze_cases.py \
  --input outputs/gsm8k_advantages.json \
  --output-prefix outputs/gsm8k_analysis
```

也可以用统一 CLI：

```bash
cmao sample --config configs/experiment/gsm8k_quickstart.json --output outputs/gsm8k_samples.json
cmao score --input outputs/gsm8k_samples.json --output outputs/gsm8k_scores.json --config configs/scoring/default.json
cmao advantage --input outputs/gsm8k_scores.json --output outputs/gsm8k_advantages.json --config configs/scoring/default.json
cmao rerank_eval --input outputs/gsm8k_advantages.json --output outputs/gsm8k_report.json
cmao report --input outputs/gsm8k_advantages.json --output outputs/gsm8k_report_pretty.json
cmao analyze_cases --input outputs/gsm8k_advantages.json --output-prefix outputs/gsm8k_analysis
```

## 每一步做什么

### `sample`

从配置里读取：

- 模型名
- 数据集
- 采样超参数

然后对每道题生成一个 group 的候选解，保存为 JSON。

运行时会显示题目级进度条，便于观察当前已经采样到第几题。

输出文件里每组数据大致长这样：

```json
{
  "problem": {
    "id": "gsm8k-0",
    "source": "gsm8k",
    "prompt": "...",
    "gold_answer": "42",
    "metadata": {}
  },
  "samples": [
    {
      "problem_id": "gsm8k-0",
      "sample_id": "gsm8k-0-sample-0",
      "cot_text": "...",
      "final_answer": "42",
      "raw_text": "...",
      "generation_meta": {
        "model_name": "Qwen/Qwen2.5-Math-1.5B-Instruct"
      }
    }
  ]
}
```

### `score`

对 `sample` 结果做三件事：

- 判定 `answer_correct`
- 计算 `quality_score`
- 打 `mode_label`

第二阶段开始，`score` 还会额外输出：

- `quality_evidence`
- `mode_evidence`
- `answer_extraction`
- `answer_judgment`

默认评分配置在 [configs/scoring/default.json](/home/wyatt/research/CMAO/configs/scoring/default.json)。

当前 `r_qual` 采用组合分数：

```text
r_qual =
0.20 * format +
0.35 * local_check +
0.20 * structure +
0.15 * self_verify +
0.10 * concise
```

含义如下：

- `format`
  最终答案是否容易抽取，是否有明显收束
- `local_check`
  中间算式或等式能否被程序局部校验
- `structure`
  推理结构是否清晰、是否重复空转
- `self_verify`
  是否出现代回、检查、sanity check
- `concise`
  是否过度冗长

第二阶段新增但暂不并入总分的分析特征：

- `answer_consistency`
- `reasoning_redundancy`

### `advantage`

对每个 group 计算：

- `A_ans`
- `A_qual`
- `A_mode`
- `A_total`

其中：

- `A_qual` 只在正确样本内标准化
- `A_mode` 会对“少见但高质量”的 mode 给出更大奖励
- `A_total` 默认是三项的等权和

### `rerank_eval`

对每道题比较几种选样策略：

- `greedy`
- `majority_vote`
- `quality`
- `a_total`
- `a_total_without_mode`
- `quality_only_correct_samples`

输出聚合报告，包括：

- 各策略准确率
- 分 subset 指标：全对组、部分对组、全错组
- 全对组数量
- 全对组内部质量方差
- 正确样本的 mode 分布
- 答案抽取失败率
- 抽取后判定失败率
- quality 消融结果
- 每题的 group-level 诊断信息

## Pilot 训练

当前仓库已经包含一版 `CMAO` pilot 训练骨架，推荐流程是：

1. 先生成并打分一批 `advantaged groups`
2. 再把这些 group 扁平化成训练 JSONL
3. 最后运行 `LoRA + CMAO clipped loss` 做小规模训练

训练数据准备：

```bash
python scripts/prepare_train_data.py \
  --input outputs/gsm8k_advantages.json \
  --output outputs/train/gsm8k_train_records.jsonl
```

训练命令：

```bash
python scripts/train_policy.py \
  --config configs/training/gsm8k_cmao_lora.json \
  --input outputs/train/gsm8k_train_records.jsonl
```

统一 CLI 也支持：

```bash
cmao prepare_train_data --input outputs/gsm8k_advantages.json --output outputs/train/gsm8k_train_records.jsonl
cmao train_policy --config configs/training/gsm8k_cmao_lora.json --input outputs/train/gsm8k_train_records.jsonl
```

默认提供两份训练配置：

- `configs/training/gsm8k_grpo_baseline_lora.json`
  只使用 `A_ans`
- `configs/training/gsm8k_cmao_lora.json`
  使用 `A_ans + 0.5 * A_qual + 0.1 * A_mode`

这版 trainer 的定位是机制验证，不是最终论文版在线 RL 系统：

- 底层用 `Transformers + Accelerate + PEFT`
- 当前默认训练方式是 `LoRA`
- 训练数据来自现有离线 pipeline，因此最适合先在 `GSM8K` 上做小规模实验
- 后续如果要改成更强的在线 rollout / update loop，可以继续复用这里的训练数据格式和 loss 接口

### `report`

把保存好的报告 JSON 打印出来，或者保存成文件。

如果输入的是带 `groups` 的中间文件，它也会现算一份聚合报告。

如果你不想在终端看到长输出，直接传：

```bash
python scripts/report.py \
  --input outputs/gsm8k_advantages.json \
  --output outputs/gsm8k_report_pretty.json
```

### `analyze_cases`

从 `scored` 或 `advantaged` 文件里导出代表性案例，当前重点包括：

- `greedy` 错但 `quality` 对
- `quality` 错但 `majority_vote` 对
- 全对组但质量差异明显
- rare mode 且质量高的样本

会输出两个文件：

- `*_cases.jsonl`
- `*_summary.json`

命令行现在只会打印这两个文件的路径，不会把完整内容直接打到终端。

## 支持的数据集

当前内置逻辑名：

- `gsm8k`
- `math-500`
- `math-lighteval`

实现位置在 [datasets.py](/home/wyatt/research/CMAO/src/cmao/datasets.py)。

默认优先从 Hugging Face 数据集读取。

## 使用本地数据

除了 Hugging Face 数据集，也可以直接读取本地 `.json` 或 `.jsonl`。

建议把本地文件放在 [data/README.md](/home/wyatt/research/CMAO/data/README.md) 里描述的目录结构下，例如：

- `data/raw/`
- `data/processed/`
- `data/cache/`

本地记录会尽量从这些字段里抽取题目和答案：

- prompt 候选：`prompt` / `problem` / `question` / `input`
- answer 候选：`gold_answer` / `answer` / `solution` / `final_answer` / `target`

你可以写一个本地配置，例如：

```json
{
  "model": {
    "name": "Qwen/Qwen2.5-Math-1.5B-Instruct"
  },
  "dataset": {
    "path": "data/processed/my_math_subset.jsonl",
    "name": "local-math",
    "limit": 20
  },
  "sampling": {
    "group_size": 4,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 1024,
    "do_sample": true
  }
}
```

对应的 JSONL 记录最小格式可以是：

```json
{"id": "p1", "prompt": "What is 2+3?", "gold_answer": "5"}
{"id": "p2", "question": "Solve x+1=4", "answer": "3"}
```

## 当前的 mode 标签

规则版 `phi(c)` 目前支持这些标签：

- `tool_integrated`
- `case_split`
- `backsolve_or_check`
- `enumeration_or_counting`
- `equation_manipulation`
- `direct_arithmetic`
- `other_math`

这版先追求可解释和容易调试，后面如果你想换成小分类器或 verifier，可以直接替换实现，不需要重写整个 pipeline。

## 当前的限制

这版是一期原型，已经能做离线实验，但有一些明确限制：

- 还没有在线 RL / GRPO 训练环节
- `quality_scorer` 还是启发式，不是 PRM / reward model
- `mode_tagger` 目前是规则系统，不是学习式分类器
- 数学表达式判定目前偏保守，复杂 LaTeX 场景还可能需要加强
- 推理生成后端当前只接了 `transformers`

## 推荐的使用顺序

第一次使用时，建议按这个顺序：

1. 跑单测，确认代码正常
2. 用 `gsm8k_quickstart.json` 跑 10 到 50 题
3. 看 `report` 里 `all_correct_group_count` 和 `avg_all_correct_quality_variance`
4. 观察 `quality` 和 `a_total` 是否优于 `greedy`
5. 再扩到 `MATH-500`

## 后续扩展建议

如果你接下来准备继续做正式实验，最自然的扩展顺序是：

1. 把 `quality_scorer` 升级成 verifier 或 PRM
2. 把 `mode_tagger` 从规则升级成小分类模型
3. 接 LoRA / QLoRA 训练
4. 再接在线 CMAO 或 GRPO 风格训练循环

## 常见问题

### 为什么 `sample` 最慢？

因为这一步真的会调用模型生成多个候选解。后面的 `score`、`advantage` 和 `report` 都是离线处理。

### 为什么要先做离线闭环？

因为你要先验证 CMAO 信号本身有没有意义。否则一开始就上 RL，出了问题很难知道是奖励设计的问题、训练的问题，还是模型采样的问题。

### 我只想先改 scoring 逻辑，应该看哪里？

先看：

- [quality_scorer.py](/home/wyatt/research/CMAO/src/cmao/quality_scorer.py)
- [answer_judge.py](/home/wyatt/research/CMAO/src/cmao/answer_judge.py)
- [mode_tagger.py](/home/wyatt/research/CMAO/src/cmao/mode_tagger.py)

### 我想看 advantage 是怎么算的？

看 [cmao.py](/home/wyatt/research/CMAO/src/cmao/cmao.py)。

---

如果你愿意，我下一步可以继续帮你补两样很实用的东西：

- 一个更完整的 `docker-compose.yml`
- 一个“从零跑第一次实验”的 `QUICKSTART.md`




python scripts/prepare_train_data.py \
  --input outputs/gsm8k_advantages.json \
  --output outputs/train/gsm8k_train_records.jsonl


2. 重新训练三组

python scripts/train_policy.py \
  --config configs/training/gsm8k_grpo_baseline_lora.json \
  --input outputs/train/gsm8k_train_records.jsonl
python scripts/train_policy.py \
  --config configs/training/gsm8k_cmao_lite_lora.json \
  --input outputs/train/gsm8k_train_records.jsonl
python scripts/train_policy.py \
  --config configs/training/gsm8k_cmao_lora.json \
  --input outputs/train/gsm8k_train_records.jsonl
  
3. 重新合并 LoRA

python scripts/merge_lora.py \
  --adapter outputs/train/gsm8k_grpo_baseline_lora/checkpoint-final \
  --output outputs/merged/gsm8k_grpo_baseline
python scripts/merge_lora.py \
  --adapter outputs/train/gsm8k_cmao_lite_lora/checkpoint-final \
  --output outputs/merged/gsm8k_cmao_lite
python scripts/merge_lora.py \
  --adapter outputs/train/gsm8k_cmao_lora/checkpoint-final \
  --output outputs/merged/gsm8k_cmao_full

4. 重新评测

base：

python scripts/sample.py --config configs/experiment/eval_gsm8k_base_100.json --output outputs/eval/gsm8k_base_samples.json
python scripts/score.py --input outputs/eval/gsm8k_base_samples.json --output outputs/eval/gsm8k_base_scores.json --config configs/scoring/default.json
python scripts/advantage.py --input outputs/eval/gsm8k_base_scores.json --output outputs/eval/gsm8k_base_advantages.json --config configs/scoring/default.json
python scripts/report.py --input outputs/eval/gsm8k_base_advantages.json --output outputs/eval/gsm8k_base_report.json

GRPO：

python scripts/sample.py --config configs/experiment/eval_gsm8k_grpo_100.json --output outputs/eval/gsm8k_grpo_samples.json
python scripts/score.py --input outputs/eval/gsm8k_grpo_samples.json --output outputs/eval/gsm8k_grpo_scores.json --config configs/scoring/default.json
python scripts/advantage.py --input outputs/eval/gsm8k_grpo_scores.json --output outputs/eval/gsm8k_grpo_advantages.json --config configs/scoring/default.json
python scripts/report.py --input outputs/eval/gsm8k_grpo_advantages.json --output outputs/eval/gsm8k_grpo_report.json

CMAO-lite：

python scripts/sample.py --config configs/experiment/eval_gsm8k_cmao_lite_100.json --output outputs/eval/gsm8k_cmao_lite_samples.json
python scripts/score.py --input outputs/eval/gsm8k_cmao_lite_samples.json --output outputs/eval/gsm8k_cmao_lite_scores.json --config configs/scoring/default.json
python scripts/advantage.py --input outputs/eval/gsm8k_cmao_lite_scores.json --output outputs/eval/gsm8k_cmao_lite_advantages.json --config configs/scoring/default.json
python scripts/report.py --input outputs/eval/gsm8k_cmao_lite_advantages.json --output outputs/eval/gsm8k_cmao_lite_report.json
Full-CMAO：

python scripts/sample.py --config configs/experiment/eval_gsm8k_cmao_full_100.json --output outputs/eval/gsm8k_cmao_full_samples.json
python scripts/score.py --input outputs/eval/gsm8k_cmao_full_samples.json --output outputs/eval/gsm8k_cmao_full_scores.json --config configs/scoring/default.json
python scripts/advantage.py --input outputs/eval/gsm8k_cmao_full_scores.json --output outputs/eval/gsm8k_cmao_full_advantages.json --config configs/scoring/default.json
python scripts/report.py --input outputs/eval/gsm8k_cmao_full_advantages.json --output outputs/eval/gsm8k_cmao_full_report.json