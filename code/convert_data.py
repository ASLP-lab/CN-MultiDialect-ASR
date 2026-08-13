"""Convert ASR jsonl (wav/text) to the training format.

Input fields (any of):
  wav | source     audio path
  text_final | txt | text | target     transcript
  language         optional, one of Chinese / English / Cantonese
  key              optional utterance id

Output:
  {"key": "...", "messages": [...], "audios": ["path.wav"]}
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    import orjson

    def json_loads(line):
        return orjson.loads(line)

    def json_dumps(record: dict) -> bytes:
        return orjson.dumps(record) + b'\n'

except ImportError:

    def json_loads(line):
        return json.loads(line)

    def json_dumps(record: dict) -> bytes:
        return (json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8')


def clean_text(text):
    if not text:
        return ''
    return ' '.join(str(text).split())


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return 'Chinese'
    lang = str(language).strip()
    low = lang.lower()
    if low in {'chinese', 'cantonese', 'english'}:
        return low[:1].upper() + low[1:]
    return 'Chinese'


def convert_record(record: dict, language: str = 'Chinese') -> Optional[dict]:
    key = record.get('key', '')
    wav = record.get('wav', record.get('source', ''))
    txt = clean_text(
        record.get('text_final')
        or record.get('txt')
        or record.get('text')
        or record.get('target')
        or ''
    )
    if not txt:
        return None
    lang = normalize_language(record.get('language') or language)
    return {
        'key': key,
        'messages': [
            {'role': 'user', 'content': '<audio>'},
            {'role': 'assistant', 'content': f'language {lang}<asr_text>{txt}'},
        ],
        'audios': [wav],
    }


def validate_wav_path(record: dict, check_exists: bool = False) -> tuple[bool, str]:
    wav = record.get('wav') or record.get('source') or ''
    if not isinstance(wav, str) or not wav.strip():
        return False, 'missing_wav'
    if wav.rstrip('/').endswith('/train'):
        return False, 'wav_is_directory'
    if not check_exists:
        return True, ''
    path = Path(wav)
    if path.is_dir():
        return False, 'wav_is_directory'
    if not path.is_file():
        return False, 'wav_not_found'
    return True, ''


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    language: str = 'Chinese',
    debug_limit: Optional[int] = None,
    check_exists: bool = False,
    write_batch_size: int = 8192,
) -> None:
    total = success = skipped = 0
    reason_counts: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open('rb') as fin, output_path.open('wb', buffering=16 * 1024 * 1024) as fout:
        write_buf: list[bytes] = []
        for line in tqdm(fin):
            if debug_limit is not None and total >= debug_limit:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json_loads(line)
            except Exception:
                skipped += 1
                reason_counts['invalid_json'] = reason_counts.get('invalid_json', 0) + 1
                continue

            valid, reason = validate_wav_path(record, check_exists=check_exists)
            if not valid:
                skipped += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                continue

            converted = convert_record(record, language=language)
            if converted is None:
                skipped += 1
                reason_counts['empty_text'] = reason_counts.get('empty_text', 0) + 1
                continue

            write_buf.append(json_dumps(converted))
            success += 1
            if len(write_buf) >= write_batch_size:
                fout.writelines(write_buf)
                write_buf.clear()

        if write_buf:
            fout.writelines(write_buf)

    print(f'Done. total={total}, success={success}, skipped={skipped}, output={output_path}')
    print(f'skip_detail: {reason_counts}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert ASR jsonl to SFT / OPSD training format.')
    parser.add_argument('--input', type=Path, required=True, help='Input jsonl path.')
    parser.add_argument('--output', type=Path, required=True, help='Output jsonl path.')
    parser.add_argument('--language', type=str, default='Chinese',
                        help='Fallback language tag: Chinese / English / Cantonese.')
    parser.add_argument('--debug_limit', type=int, default=None,
                        help='Only process first N non-empty lines.')
    parser.add_argument('--check_exists', action='store_true',
                        help='Skip rows whose wav path does not exist.')
    parser.add_argument('--write_batch_size', type=int, default=8192)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_jsonl(
        args.input,
        args.output,
        language=args.language,
        debug_limit=args.debug_limit,
        check_exists=args.check_exists,
        write_batch_size=args.write_batch_size,
    )
