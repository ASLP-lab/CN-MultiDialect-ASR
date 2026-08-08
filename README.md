# On-Policy Self-Distillation for Multi-Dialect ASR: Mastering Dialects, Retaining Mandarin

> Official repository of the paper "On-Policy Self-Distillation for Multi-Dialect ASR: Mastering Dialects, Retaining Mandarin".

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-blue)](#)
[![Demo](https://img.shields.io/badge/Demo-Listen%20in%20README-brightgreen)](#demo)
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

Dialect recognition examples with model transcriptions. Click an audio link to listen on GitHub.

Dialects: **Cantonese**, **Minnan**, **Sichuan**, **Wu** (5 samples each).

### Cantonese

**Sample 1** · [Play audio](demo/audio/cantonese/01_0001370323.wav)

<audio controls preload="none" src="demo/audio/cantonese/01_0001370323.wav"></audio>

> 首阶段重建嘅一 B 期，原定同一 A 期一齐展开，包括华建、华泰同华昌楼住户可以搬去华富北以及华乐径嘅安置屋村。房委会话工程期间发现华富北嘅岩石层比预计高要改动设计，预计要延迟一年半到2030年先至入伙。

**Sample 2** · [Play audio](demo/audio/cantonese/02_0000341459.wav)

<audio controls preload="none" src="demo/audio/cantonese/02_0000341459.wav"></audio>

> 据统计，2019年就有大概五百万人因为肥胖而过早死亡。世界肥胖联盟甚至估计2035年肥胖人口会膨胀到四十亿肥，最多只系影响外观啫。我都唔 CARE 人哋点睇，我大把明星肥得嚟都好睇啦。嗱，你咁谂又未必啱。

**Sample 3** · [Play audio](demo/audio/cantonese/03_0000776688.wav)

<audio controls preload="none" src="demo/audio/cantonese/03_0000776688.wav"></audio>

> 其中一个常见需要思考同决定嘅治疗选项，就系心肺复苏术。一般嘅情况下，当病人嘅心脏停止跳动，肺部停止运作。我哋会用心肺复苏术 C P R 嚟到试图让心脏重新跳动，肺部重新运作。

**Sample 4** · [Play audio](demo/audio/cantonese/04_0000921472.wav)

<audio controls preload="none" src="demo/audio/cantonese/04_0000921472.wav"></audio>

> 然后连同你嘅身份证，同埋香港嘅 DRIVER LICENSE 呢三份文件分别去三个部门去做认证嘅或者宣誓嘅咁。第一呢就系将呢三份文件带去香港民政事务处去做宣誓。

**Sample 5** · [Play audio](demo/audio/cantonese/05_gd0007069_186070_198430.wav)

<audio controls preload="none" src="demo/audio/cantonese/05_gd0007069_186070_198430.wav"></audio>

> 佢最想要有露台，佢想感受呢个国家嘅生气，以佢二十八万呢个预算有少少难度，佢可能要做少少妥协，不过我会尽力帮佢揾到佢想要嘅单位。


### Minnan

**Sample 1** · [Play audio](demo/audio/minnan/01_0157_002_phone_0245.wav)

<audio controls preload="none" src="demo/audio/minnan/01_0157_002_phone_0245.wav"></audio>

> 学淡薄即个种的健康的知识，最基本的种的健康观念吼，对咱一个家庭的贡献是野大的。

**Sample 2** · [Play audio](demo/audio/minnan/02_0175_003_phone_0088.wav)

<audio controls preload="none" src="demo/audio/minnan/02_0175_003_phone_0088.wav"></audio>

> 阿像咱平常时咱伫咧红工厂咧，即种诶，咱计无迄种意识说咱着来去做一个体检吼。

**Sample 3** · [Play audio](demo/audio/minnan/03_0192_002_phone_0114.wav)

<audio controls preload="none" src="demo/audio/minnan/03_0192_002_phone_0114.wav"></audio>

> 这迄种以以前随开始毋免充钱啊，然后就是说伊开始嘛嘛嘛，毋是叫嘛，毋是叫食鸡。

**Sample 4** · [Play audio](demo/audio/minnan/04_0236_008_phone_0098.wav)

<audio controls preload="none" src="demo/audio/minnan/04_0236_008_phone_0098.wav"></audio>

> 嗯，公园有迄种，然后伊去公园做，然后拍迄个太极，然后我是说爬山诶，老岁仔较济。

**Sample 5** · [Play audio](demo/audio/minnan/05_0165_001_phone_0178.wav)

<audio controls preload="none" src="demo/audio/minnan/05_0165_001_phone_0178.wav"></audio>

> 相信我诶，囡仔是一个阿聪明诶，囡仔，你只要努力，伊说就有法通诶，考悬分安尼。


### Sichuan

**Sample 1** · [Play audio](demo/audio/sichuan/01_00959004159_3W4nR_15_21220.wav)

<audio controls preload="none" src="demo/audio/sichuan/01_00959004159_3W4nR_15_21220.wav"></audio>

> 然后你看到你跟他们不同，就会觉得是自己有问题。初中是平行班的噻，属于那种比较乖的。平行班就是到了新的集体里面，感觉大家就是看事情的时候，并不是那么在意说我有没有符合规则，更多的是我这个想法到底是啥子样子的，我有没有把我想法表达出来，我有没有证明我的想法。

**Sample 2** · [Play audio](demo/audio/sichuan/02_012856315471_5cvYQ_135_9340.wav)

<audio controls preload="none" src="demo/audio/sichuan/02_012856315471_5cvYQ_135_9340.wav"></audio>

> 是各人尽量保持一个中立的态度嘞，那种表情这种时间哈持续过长了，你可能身体就会开始紧张。

**Sample 3** · [Play audio](demo/audio/sichuan/03_012856315423_BKFrD_20_9100.wav)

<audio controls preload="none" src="demo/audio/sichuan/03_012856315423_BKFrD_20_9100.wav"></audio>

> 你看到没得嘛，他把那个牛胃抱起进去了，可能是把牛胃打整出来，把那个毛肚儿弄起出来。

**Sample 4** · [Play audio](demo/audio/sichuan/04_012856315868_BHycL_172_7580.wav)

<audio controls preload="none" src="demo/audio/sichuan/04_012856315868_BHycL_172_7580.wav"></audio>

> 明朝时候的休假制度也必须要介绍一下，让我们来看哈古人的休息制度到底是啥子样子。

**Sample 5** · [Play audio](demo/audio/sichuan/05_009590040760_GfOio_12_11060.wav)

<audio controls preload="none" src="demo/audio/sichuan/05_009590040760_GfOio_12_11060.wav"></audio>

> 他们都自信满满的围到王宫，后头等待着他们公主的到来。正午的时候，国王带着三个公主。


### Wu

**Sample 1** · [Play audio](demo/audio/wu/01_223327131588_A1Zmc_99_15540.wav)

<audio controls preload="none" src="demo/audio/wu/01_223327131588_A1Zmc_99_15540.wav"></audio>

> 合作伙伙伴的关系对伐阿拉不仅仅是讲，哎，我就是政府拨我钞票，我就去做。其实不是介不是介上下级的这种关系的，更加多的是哎我也想解决这只问题，我也看到了这只问题。哎，葛末现在正好政府也辣，解决阿拉能不能一道作为。

**Sample 2** · [Play audio](demo/audio/wu/02_223327128685_P876Q_357_12820.wav)

<audio controls preload="none" src="demo/audio/wu/02_223327128685_P876Q_357_12820.wav"></audio>

> 有劲的事体对伐比较奇葩的事体，也欢迎㑚帮阿拉一道吐槽吐槽，让自家心里向真正能够放下来，让阿拉每天侪能够过了开开心心。这是阿拉做这档节目的初衷。

**Sample 3** · [Play audio](demo/audio/wu/03_22332711653_bHQsF_120_15840.wav)

<audio controls preload="none" src="demo/audio/wu/03_22332711653_bHQsF_120_15840.wav"></audio>

> 这个告目标好像不是老搭界，新年愿望就可以了。愿望蛮好的，就讲哪能讲法子呢？每个人有每个人的想法，但是小薇我觉着侬讲了老好老好的，让我想到一点。

**Sample 4** · [Play audio](demo/audio/wu/04_22332712433_CHh8v_4_15860.wav)

<audio controls preload="none" src="demo/audio/wu/04_22332712433_CHh8v_4_15860.wav"></audio>

> 为伊的部队到达做准备。再过几个礼拜，就是五十一周岁的迪安辣辣陆军将领里向算不上一个了不起的人物。伊辣辣接受了预备役军队训练团的训练以后。

**Sample 5** · [Play audio](demo/audio/wu/05_223327123738_NpsG3_248_10960.wav)

<audio controls preload="none" src="demo/audio/wu/05_223327123738_NpsG3_248_10960.wav"></audio>

> 葛末阿拉立辣海嘛，人家就辣边浪向问阿拉日本人就老好奇的伊讲，哎哟侬这和服老好看哦，这啥地方的衣裳，伊拉一开始以为是韩国人的这种衣裳。

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
