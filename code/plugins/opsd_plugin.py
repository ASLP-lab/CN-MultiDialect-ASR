"""ASR OPSD plugin: student sees audio; teacher sees audio + reference transcript."""
import os
import re
from typing import Any, Dict, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset


def _strip_lang_prefix(text: str) -> str:
    return re.sub(r'^language\s+\S+<asr_text>', '', text)


class ASROPSDPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = row.get('messages', [])
        audios = row.get('audios', [])
        if not messages or not audios:
            return None

        ref_text = ''
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                ref_text = msg.get('content', '')
                break
        if not ref_text:
            return None

        clean_text = _strip_lang_prefix(ref_text)
        return {
            'messages': messages,
            'teacher_prompt': f'<audio>\n\nHere is a transcription of this audio:\n{clean_text}',
            'audios': audios,
        }


register_dataset(
    DatasetMeta(
        dataset_name='asr-opsd',
        dataset_path=os.environ.get('OPSD_DATA_PATH', ''),
        preprocess_func=ASROPSDPreprocessor(),
        tags=['asr', 'opsd'],
    ))
