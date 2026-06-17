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


def load_text(filename):
    """Load a raw text file from the data directory as a single string."""
    path = DATA_DIR / filename
    return path.read_text(encoding="utf-8")


def build_char_vocab(text):
    """Build stoi/itos from the unique characters of a text (no reserved token)."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
