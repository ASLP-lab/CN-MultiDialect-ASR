# On-Policy Self-Distillation for Multi-Dialect ASR: Mastering Dialects, Retaining Mandarin

> Official repository of the paper "On-Policy Self-Distillation for Multi-Dialect ASR: Mastering Dialects, Retaining Mandarin".

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-blue)](#)
[![Demo](https://img.shields.io/badge/Demo-Listen%20Online-brightgreen)](https://aslp-lab.github.io/MultiDialect-ASR/demo/)
[![Model](https://img.shields.io/badge/Model-ASLP--lab%2FMultiDialect--ASR-orange)](https://huggingface.co/ASLP-lab/MultiDialect-ASR/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ASLP--lab%2FMultiDialect--ASR-yellow)](https://huggingface.co/ASLP-lab/MultiDialect-ASR/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](https://huggingface.co/ASLP-lab/MultiDialect-ASR/)

</div>

<div align="center">
  <img src="assets/opsd.png" alt="OPSD framework" width="90%">
  <p><em>Overview of the staged adaptation pipeline. Top: base model, CPT, SFT, and OPSD. Bottom: OPSD with student on-policy prefixes, a frozen teacher conditioned on the reference transcript as privileged context, soft targets q<sub>t</sub>, and token-level KL.</em></p>
</div>

## Introduction

Large-scale ASR models such as Qwen3-ASR already achieve strong Mandarin recognition and have some ability to recognize Chinese dialects. In real-world speech, however, dialect recognition remains limited. Direct dialect fine-tuning can lower dialect CER, but it often raises Mandarin CER at the same time.

This repository studies how to adapt a capable ASR model for multi-dialect recognition without degrading Mandarin recognition. We adopt a three-stage adaptation pipeline:

1. **CPT**: continual pre-training on large-scale Mandarin-dialect speech to strengthen the Chinese ASR foundation.
2. **SFT**: dialect supervised fine-tuning with increased dialect sampling weight to lower dialect CER.
3. **OPSD**: On-Policy Self-Distillation as the final refinement objective.

OPSD addresses the train–test mismatch in autoregressive ASR by training the student on its own decoded prefixes, while a frozen teacher, conditioned on the reference transcript as privileged context, provides soft token-level targets. Under matched refinement data and schedule, OPSD improves dialect recognition without raising Mandarin CER, whereas continued teacher-forced fine-tuning increases Mandarin CER.

We instantiate the framework with Qwen3-ASR-1.7B and evaluate it on public and internal Mandarin and dialect test sets. Model weights are available at [ASLP-lab/MultiDialect-ASR](https://huggingface.co/ASLP-lab/MultiDialect-ASR/).

## Key Features

* **Mandarin–dialect balanced adaptation**: improves Chinese dialect ASR while retaining or improving Mandarin recognition.
* **Three-stage pipeline**: CPT builds the foundation, dialect SFT specializes the model, and OPSD refines the final checkpoint.
* **On-Policy Self-Distillation**: trains under student-decoded prefixes with soft teacher targets, reducing the train–test mismatch of teacher-forced ASR training.
* **Strong empirical trade-off**: relative to Continued SFT on the same refinement data, OPSD lowers Mandarin, Dialect, Internal, and Overall CER.
* **Compatible with Qwen3-ASR inference**: released checkpoints follow the same usage interface as Qwen3-ASR via the `qwen-asr` package.

## Demo

Listen to dialect recognition examples online:

**[Open Demo](https://aslp-lab.github.io/MultiDialect-ASR/demo/)**

The page covers Cantonese, Minnan, Sichuan, and Wu. Each dialect includes 5 audio samples with model transcriptions.

Local preview:

```bash
python3 -m http.server 8766 --directory demo
```

Then open `http://127.0.0.1:8766/`.

> After pushing to GitHub, enable **Settings → Pages → Deploy from branch `main` / root** (or `/docs`) so the online demo link works.

## Quickstart

Inference is compatible with [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR). We recommend installing the official `qwen-asr` package in a clean environment.

### Environment Setup

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr
```

For faster inference with the vLLM backend:

```bash
pip install -U qwen-asr[vllm]
```

### Model Download

You can load the model directly from Hugging Face, or download it locally first:

```bash
# Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download ASLP-lab/MultiDialect-ASR --local-dir ./MultiDialect-ASR

# ModelScope (recommended for users in Mainland China)
pip install -U modelscope
modelscope download --model ASLP-lab/MultiDialect-ASR --local_dir ./MultiDialect-ASR
```

Model card: [https://huggingface.co/ASLP-lab/MultiDialect-ASR/](https://huggingface.co/ASLP-lab/MultiDialect-ASR/)

### Python Inference

The usage is the same as Qwen3-ASR. Load the model with `Qwen3ASRModel.from_pretrained` and call `transcribe`:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "ASLP-lab/MultiDialect-ASR",  # or "./MultiDialect-ASR" for a local path
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="path/to/audio.wav",
    language="Chinese",  # or None for automatic language detection
)

print(results[0].language)
print(results[0].text)
```

Batch inference is also supported:

```python
results = model.transcribe(
    audio=[
        "path/to/mandarin.wav",
        "path/to/dialect.wav",
    ],
    language=["Chinese", "Chinese"],
)

for r in results:
    print(r.language, r.text)
```

For more advanced usage, including vLLM backend, streaming inference, and forced alignment, please refer to the [Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR).

## Method Overview

| Stage  | Training data                                                                | Goal                                                 | Objective      |
| ------ | ---------------------------------------------------------------------------- | ---------------------------------------------------- | -------------- |
| `CPT`  | Full Mandarin-dialect collection (`~100k` hours)                             | Build a stronger Chinese ASR foundation              | Cross-entropy  |
| `SFT`  | Same sources with higher dialect sampling weight and a small Mandarin anchor | Lower dialect CER                                    | Cross-entropy  |
| `OPSD` | Dialect refinement subset (`~5k` hours)                                      | Improve dialect recognition without hurting Mandarin | Token-level KL |

### Stage 1: CPT

Starting from Qwen3-ASR-1.7B, we continually pre-train on a large Mandarin-dialect corpus that combines public Mandarin corpora, public dialect corpora, and internal Chinese speech. This stage strengthens the overall ASR foundation before dialect-focused adaptation.

### Stage 2: SFT

Dialect SFT keeps the same training sources but changes the Mandarin-dialect sampling ratio. All dialect training data are retained, while only a small amount of Mandarin data is kept as an anchor. This stage improves dialect CER, but it can still raise Mandarin CER.

### Stage 3: OPSD

We apply OPSD to the SFT checkpoint as the final refinement objective.

1. The student samples its own hypothesis from the input speech with training temperature `tau = 0.8`.
2. A frozen teacher, initialized from the same SFT checkpoint, sees the reference transcript as privileged context and predicts soft token targets on the student prefixes.
3. The student is updated by matching the teacher distribution with token-level KL divergence.

At inference time, only the student pathway is used. Unlike continued teacher-forced fine-tuning, OPSD trains under decoding states closer to inference and avoids another hard one-hot update on dialect data.

## Experimental Setup

### Training Data

The full training collection contains approximately `100k` hours of Mandarin-dialect speech.

| Source                | Dialect coverage         | Hours    |
| --------------------- | ------------------------ | -------- |
| WenetSpeech           | Mandarin                 | `~22.4k` |
| AISHELL-1             | Mandarin                 | `~178`   |
| AISHELL-2             | Mandarin                 | `~1,000` |
| AliMeeting            | Mandarin                 | `~0.1k`  |
| Common Voice 17.0     | Mandarin                 | `~234`   |
| MAGICDATA Read Speech | Mandarin                 | `~755`   |
| KeSpeech              | Accented Mandarin        | `~1.5k`  |
| WenetSpeech-Yue       | Cantonese                | `~21.8k` |
| WenetSpeech-Chuan     | Sichuan                  | `~10.0k` |
| WenetSpeech-Wu        | Wu                       | `~8.0k`  |
| Internal data         | Mandarin + 23 dialects   | `~34.1k` |
| Total                 | Mandarin-dialect mixture | `~100k`  |

For OPSD and Continued SFT, we further select a refinement set of approximately `5k` hours from dialect training data. These samples are filtered for metadata reliability, usable audio, reasonable transcript length, and high CER under the SFT model. No development or test utterances are used.

### Evaluation Sets

We report four macro-averaged CER metrics:

- `Mandarin Avg.`: 8 Mandarin test sets.
- `Dialect Avg.`: 5 public dialect test sets.
- `Internal Avg.`: 18 internal dialect test sets.
- `Overall Avg.`: macro-average over all 31 test sets.

#### Mandarin test sets

`AISHELL-1`, `AISHELL-2`, `KeSpeech`, `SpeechIO-1`, `SpeechIO-2`, `SpeechIO-3`, `WenetSpeech Test_Meeting`, `WenetSpeech Test_Net`

#### Public dialect test sets

`WenetSpeech-Yue Long`, `WenetSpeech-Yue Short`, `WenetSpeech-Chuan Easy`, `WenetSpeech-Chuan Hard`, `WenetSpeech-Wu`

#### Internal dialect test sets

`Anhui`, `Cantonese`, `Changsha`, `Chaoshan`, `Dongbei`, `Henan`, `Kejia`, `Minnan`, `Nanchang`, `Nanjing`, `Shanxi`, `Shaanxi`, `Shandong`, `Shanghai`, `Sichuan`, `Suzhou`, `Wuhan`, `Xuzhou`

## Main Results

### Aggregate CER (%)

| Model     | Mandarin Avg. | Dialect Avg. | Internal Avg. | Overall Avg. |
| --------- | ------------- | ------------ | ------------- | ------------ |
| GLM-ASR   | 5.28          | 37.21        | 41.55         | 31.49        |
| Fun-ASR   | 4.59          | 18.43        | 27.10         | 19.89        |
| Qwen3-ASR | 3.46          | 15.37        | 21.01         | 15.57        |
| CPT       | 3.78          | 13.74        | 15.09         | 11.95        |
| SFT       | 3.40          | 13.16        | 13.30         | 10.72        |
| **OPSD**  | **3.27**      | **12.79**    | **12.42**     | **10.12**    |

### Mandarin CER (%)

| Evaluation set | Qwen3-ASR | CPT      | SFT  | OPSD     |
| -------------- | --------- | -------- | ---- | -------- |
| AISHELL-1      | 1.57      | 1.47     | 1.43 | **1.38** |
| AISHELL-2      | 2.79      | 2.62     | 2.58 | **2.52** |
| KeSpeech       | 5.11      | 4.72     | 4.84 | **4.56** |
| SpeechIO-1     | 0.75      | **0.71** | 0.89 | 0.86     |
| SpeechIO-2     | 3.83      | 4.17     | 3.50 | **3.39** |
| SpeechIO-3     | 1.39      | 1.33     | 1.35 | **1.27** |
| Test_Meeting   | 6.74      | 8.50     | 7.22 | **6.85** |
| Test_Net       | 5.46      | 6.74     | 5.38 | **5.30** |
| Mandarin Avg.  | 3.46      | 3.78     | 3.40 | **3.27** |

### Public dialect CER (%)

| Evaluation set         | Dialect   | Qwen3-ASR | CPT   | SFT   | OPSD      |
| ---------------------- | --------- | --------- | ----- | ----- | --------- |
| WenetSpeech-Yue Long   | Cantonese | 9.99      | 9.40  | 9.01  | **8.80**  |
| WenetSpeech-Yue Short  | Cantonese | 6.93      | 6.09  | 5.60  | **5.31**  |
| WenetSpeech-Chuan Easy | Sichuan   | 12.38     | 12.91 | 12.30 | **11.86** |
| WenetSpeech-Chuan Hard | Sichuan   | 21.79     | 23.24 | 22.27 | **21.74** |
| WenetSpeech-Wu         | Wu        | 25.74     | 17.06 | 16.60 | **16.26** |
| Dialect Avg.           |           | 15.37     | 13.74 | 13.16 | **12.79** |

### Internal dialect CER (%)

| Dialect       | Qwen3-ASR | CPT   | SFT       | OPSD      |
| ------------- | --------- | ----- | --------- | --------- |
| Anhui         | 18.95     | 14.63 | 13.99     | **13.08** |
| Cantonese     | 10.06     | 8.71  | 7.87      | **7.74**  |
| Changsha      | 14.79     | 11.98 | 10.83     | **10.23** |
| Chaoshan      | 45.59     | 32.26 | 27.34     | **25.21** |
| Dongbei       | 6.45      | 6.09  | 5.95      | **5.80**  |
| Henan         | 8.46      | 7.26  | 6.27      | **5.99**  |
| Kejia         | 60.47     | 37.70 | 32.01     | **28.60** |
| Minnan        | 30.03     | 23.09 | 19.79     | **18.59** |
| Nanchang      | 33.41     | 20.47 | 18.63     | **15.58** |
| Nanjing       | 13.37     | 10.58 | 9.67      | **9.33**  |
| Shanxi        | 28.53     | 21.88 | 20.17     | **18.69** |
| Shaanxi       | 9.68      | 6.93  | 6.69      | **6.28**  |
| Shandong      | 8.78      | 8.39  | 7.82      | **7.64**  |
| Shanghai      | 15.78     | 12.86 | **11.94** | 12.07     |
| Sichuan       | 5.99      | 6.13  | **5.13**  | 5.38      |
| Suzhou        | 50.35     | 26.73 | 22.12     | **20.73** |
| Wuhan         | 11.30     | 10.22 | 8.12      | **7.59**  |
| Xuzhou        | 6.12      | 5.66  | 5.10      | **5.04**  |
| Internal Avg. | 21.01     | 15.09 | 13.30     | **12.42** |

## Citation

If you use this work, please consider citing:

```bibtex
@article{opsd2026,
  title={On-Policy Self-Distillation for Multi-Dialect ASR: Mastering Dialects, Retaining Mandarin},
  author={Anonymous Authors},
  journal={Anonymous submission},
  year={2026}
}
```

## License

The released model is licensed under [Apache 2.0](https://huggingface.co/ASLP-lab/MultiDialect-ASR/).
