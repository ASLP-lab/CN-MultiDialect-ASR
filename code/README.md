# Training

## Data

```json
{
  "key": "utt_0001",
  "messages": [
    {"role": "user", "content": "<audio>"},
    {"role": "assistant", "content": "language Chinese<asr_text>今天天气不错"}
  ],
  "audios": ["/path/to/utt_0001.wav"]
}
```

`language`: `Chinese` / `English` / `Cantonese`.

```bash
python code/convert_data.py --input raw.jsonl --output train.jsonl
```

## SFT

```bash
DATA=/path/to/train.jsonl MODEL=Qwen/Qwen3-ASR-1.7B bash code/sft.sh
```

## OPSD

```bash
MODEL=/path/to/sft/checkpoint DATA=/path/to/train.jsonl bash code/opsd.sh
```

`plugins/opsd_plugin.py` builds the teacher prompt from the reference transcript.
