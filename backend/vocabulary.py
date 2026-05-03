import json
import pathlib

_TOKEN_FILE = pathlib.Path(__file__).parent.parent / "frontend/public/data/tokens.json"
_tokens: dict | None = None
_qwen_text_to_id: dict[str, int] | None = None


def _load() -> dict:
    global _tokens
    if _tokens is None:
        with open(_TOKEN_FILE) as f:
            data = json.load(f)
        _tokens = {
            "qwen": {int(k): v for k, v in data["qwen"].items()},
            "whisper": {int(k): v for k, v in data["whisper"].items()},
        }
    return _tokens


def get_qwen_token_ids() -> list[int]:
    return list(_load()["qwen"].keys())


def get_whisper_token_ids() -> list[int]:
    return list(_load()["whisper"].keys())


def get_qwen_vocab() -> dict[int, str]:
    """Return full {qwen_token_id: text} dict for all known vocabulary."""
    return dict(_load()["qwen"])


def whisper_to_qwen_id(whisper_id: int) -> int | None:
    """Map a Whisper token ID to the corresponding Qwen token ID via shared text."""
    global _qwen_text_to_id
    tokens = _load()
    text = tokens["whisper"].get(whisper_id)
    if text is None:
        return None
    if _qwen_text_to_id is None:
        _qwen_text_to_id = {v: k for k, v in tokens["qwen"].items()}
    return _qwen_text_to_id.get(text)
