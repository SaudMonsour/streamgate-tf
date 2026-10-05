"""Verified public text, contiguous splits and boundary-safe byte windows."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import numpy as np

SOURCE = "https://www.gutenberg.org/ebooks/1661.txt.utf-8"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("source boundary markers absent or changed")
    return text[start.end():end.start()].strip().encode("utf-8")


def load_corpus(root=Path("data"), verify_remote=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.txt"
    manifest_path = root / "manifest.json"
    expected = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if verify_remote or not source.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            raw = response.read()
        if source.exists() and source.read_bytes() != raw:
            raise ValueError("public source differs from local snapshot")
        source.write_bytes(raw)
    raw = source.read_bytes()
    body = clean(raw)
    if expected and (digest(raw) != expected["source_sha256"] or
                     digest(body) != expected["body_sha256"]):
        raise ValueError("dataset hashes differ from published manifest")
    n = len(body)
    boundaries = [0, int(n * 0.8), int(n * 0.9), n]
    manifest = {
        "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle",
        "source_url": SOURCE, "catalog_url": "https://www.gutenberg.org/ebooks/1661",
        "rights": "Public domain in the USA, per Project Gutenberg catalog",
        "source_bytes": len(raw), "source_sha256": digest(raw), "body_bytes": n,
        "body_sha256": digest(body), "encoding": "UTF-8; fixed vocabulary of 256 bytes",
        "processing": "Decode UTF-8 BOM; normalize CRLF/CR to LF; strip Gutenberg header/footer at markers; strip outer whitespace; UTF-8 encode. Internal whitespace and contents list preserved.",
        "split_boundaries": boundaries, "split_policy": "contiguous 80/10/10 by byte offset",
        "duplicate_policy": "No deduplication of natural language text; zero rows removed",
        "missing_policy": "Not tabular; full source strictly decodes as UTF-8; no missing-value imputation"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    data = np.frombuffer(body, dtype=np.uint8).astype(np.int32)
    return tuple(data[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])), manifest


def windows(data, context=64):
    # Target sets never overlap. Last input is the first byte of the next window.
    starts = np.arange(0, len(data) - context, context, dtype=np.int32)
    idx = starts[:, None] + np.arange(context)[None, :]
    return data[idx], data[idx + 1], starts
