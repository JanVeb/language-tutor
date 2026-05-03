import json
import pathlib
import unicodedata
from dataclasses import dataclass, asdict

import numpy as np

SCORES_FILE = pathlib.Path(__file__).parent / "scores.json"

INITIAL_BIAS = 0.0        # known vocabulary starts neutral
NEW_WORD_BIAS = 5.0       # used when Step 8 introduces a brand-new word to vocabulary
USAGE_STEP = 0.1          # bias change per turn (used: -0.1, not used: +0.1)
MAX_TOTAL_BIAS = 5.0      # cap on base_bias + pron_boost combined
MIN_BASE_BIAS = -3.0      # floor — overused words stay accessible, never hard-blocked

# Pronunciation boost added on top of base_bias (capped by MAX_TOTAL_BIAS):
#   rank 1 (Whisper's top choice) → 0 extra, user nailed it
#   rank 2+ → +0.1..+0.5 so the model repeats the word for more practice
PRON_BASE_BOOST = 0.1
PRON_MAX_BOOST = 0.5
PRON_RANK_THRESHOLD = 99  # ranks 2..100 span the full boost range


def _is_word_token(text: str) -> bool:
    """Return True only for tokens that contain at least one letter or CJK character."""
    return any(
        unicodedata.category(c).startswith("L") or "一" <= c <= "鿿"
        for c in text
    )


@dataclass
class WordEntry:
    text: str
    bias: float = INITIAL_BIAS   # usage-driven; 0 for known words, NEW_WORD_BIAS for new
    pron_boost: float = 0.0      # pronunciation-driven; resets when user says it cleanly
    usage_count: int = 0

    def total_bias(self) -> float:
        return min(self.bias + self.pron_boost, MAX_TOTAL_BIAS)


class ScoresStore:
    _instance: "ScoresStore | None" = None

    def __init__(self):
        self._entries: dict[int, WordEntry] = {}
        self._load()

    @classmethod
    def get(cls) -> "ScoresStore":
        if cls._instance is None:
            cls._instance = ScoresStore()
        return cls._instance

    def _load(self):
        if SCORES_FILE.exists():
            with open(SCORES_FILE) as f:
                data = json.load(f)
            for tid_str, fields in data.items():
                try:
                    self._entries[int(tid_str)] = WordEntry(**fields)
                except TypeError:
                    pass  # skip entries whose schema has drifted

    def save(self):
        data = {str(tid): asdict(e) for tid, e in self._entries.items()}
        with open(SCORES_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _ensure(self, token_id: int, text: str) -> WordEntry:
        if token_id not in self._entries:
            self._entries[token_id] = WordEntry(text=text)
        return self._entries[token_id]

    def update_after_turn(
        self,
        all_vocab: dict[int, str],    # {qwen_token_id: text} — full known vocab
        used_ids: set[int],           # qwen token ids that appeared this turn
        pron_ranks: dict[int, int],   # {qwen_token_id: rank} from Whisper (1 = best)
    ):
        """Update biases after one user+LLM exchange. Skips punctuation/non-word tokens."""
        for tid, text in all_vocab.items():
            if not _is_word_token(text):
                continue
            entry = self._ensure(tid, text)
            if tid in used_ids:
                entry.usage_count += 1
                entry.bias = max(MIN_BASE_BIAS, entry.bias - USAGE_STEP)
            else:
                entry.bias = min(MAX_TOTAL_BIAS, entry.bias + USAGE_STEP)

        for tid, rank in pron_ranks.items():
            if tid in self._entries:
                if rank <= 1:
                    self._entries[tid].pron_boost = 0.0
                else:
                    t = min(rank - 1, PRON_RANK_THRESHOLD) / PRON_RANK_THRESHOLD
                    self._entries[tid].pron_boost = (
                        PRON_BASE_BOOST + t * (PRON_MAX_BOOST - PRON_BASE_BOOST)
                    )

        self.save()

    def get_dynamic_bias_array(
        self, vocab_size: int, all_vocab_ids: set[int], eos_ids: set[int]
    ) -> np.ndarray:
        """
        Returns [vocab_size] float32 array:
          -inf           for non-vocab tokens (hard block)
          total_bias()   for known vocab tokens; range [MIN_BASE_BIAS, MAX_TOTAL_BIAS]
          0.0            for EOS tokens (always reachable)
        """
        arr = np.full(vocab_size, float("-inf"), dtype=np.float32)
        for tid in all_vocab_ids:
            if 0 <= tid < vocab_size:
                entry = self._entries.get(tid)
                if entry is not None:
                    arr[tid] = entry.total_bias()
                else:
                    # Punctuation and unscored tokens: neutral (0.0), not biased
                    arr[tid] = 0.0
        for eid in eos_ids:
            if 0 <= eid < vocab_size:
                arr[eid] = 0.0
        return arr
