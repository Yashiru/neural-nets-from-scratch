"""Dataset helpers shared across experiments."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_words(filename="names.txt"):
    """Load a newline-separated word list from the data directory."""
    path = DATA_DIR / filename
    return path.read_text(encoding="utf-8").splitlines()


def build_vocab(words):
    """Build stoi/itos maps from a list of words. Index 0 is reserved for '.'."""
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
