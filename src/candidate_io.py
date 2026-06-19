from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET


def read_jsonl(path: str | Path) -> Iterator[dict]:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    mode = "rt"
    with opener(file_path, mode, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_docx_text(path: str | Path) -> str:
    file_path = Path(path)
    with zipfile.ZipFile(file_path) as docx:
        xml = docx.read("word/document.xml")
    root = ET.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)

