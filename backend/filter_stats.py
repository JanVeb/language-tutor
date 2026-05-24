import json
import pathlib
import threading

STATS_FILE = pathlib.Path(__file__).parent / "filter_stats.json"


class FilterStatsStore:
    _instance: "FilterStatsStore | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._counts: dict[str, int] = {}
        self._load()

    @classmethod
    def get(cls) -> "FilterStatsStore":
        if cls._instance is None:
            cls._instance = FilterStatsStore()
        return cls._instance

    def _load(self):
        if STATS_FILE.exists():
            with open(STATS_FILE) as f:
                self._counts = json.load(f)

    def record_many(self, token_texts: list[str]):
        with self._lock:
            for text in token_texts:
                self._counts[text] = self._counts.get(text, 0) + 1

    def save(self):
        with self._lock:
            with open(STATS_FILE, "w") as f:
                json.dump(self._counts, f, ensure_ascii=False, indent=2)

    def get_sorted(self) -> list[dict]:
        with self._lock:
            return sorted(
                [{"token": k, "count": v} for k, v in self._counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            )

    def clear(self):
        with self._lock:
            self._counts = {}
        self.save()
