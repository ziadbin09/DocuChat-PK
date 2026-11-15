"""Structure-aware chunking.

Text is split on paragraph boundaries first, then sentences, and packed greedily
up to a character budget. Consecutive chunks share a sentence-level overlap so a
fact that straddles a boundary is still retrievable. Each chunk remembers its
page range and the nearest heading above it.
"""
import re
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<=[.!?۔])\s+")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    heading: str = ""


@dataclass
class _Unit:
    text: str
    page: int
    heading: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def _units(pages: list[tuple[int, str]], max_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    heading = ""
    for page, text in pages:
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            first_line = para.split("\n", 1)[0]
            m = HEADING.match(first_line)
            if m:
                heading = m.group(2).strip()
                para = para[len(first_line):].strip()
                if not para:
                    continue
            if len(para) <= max_chars:
                units.append(_Unit(para, page, heading))
                continue
            for sent in split_sentences(para):
                # A single "sentence" longer than the budget (tables, code) is hard-split.
                for i in range(0, len(sent), max_chars):
                    units.append(_Unit(sent[i : i + max_chars], page, heading))
    return units


def chunk_pages(pages: list[tuple[int, str]], max_chars: int = 1100, overlap_chars: int = 180) -> list[Chunk]:
    units = _units(pages, max_chars)
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    size = 0

    def flush() -> None:
        if current:
            chunks.append(
                Chunk(
                    text="\n\n".join(u.text for u in current),
                    page_start=current[0].page,
                    page_end=current[-1].page,
                    heading=current[-1].heading or current[0].heading,
                )
            )

    for unit in units:
        heading_changed = bool(current) and unit.heading != current[-1].heading
        if current and (size + len(unit.text) > max_chars or heading_changed):
            flush()
            if heading_changed:
                current, size = [], 0
            else:
                # Carry the tail of the previous chunk forward as overlap.
                tail, tail_size = [], 0
                for prev in reversed(current):
                    sentences = split_sentences(prev.text)
                    piece = sentences[-1] if sentences else prev.text
                    if tail_size + len(piece) > overlap_chars:
                        break
                    tail.insert(0, _Unit(piece, prev.page, prev.heading))
                    tail_size += len(piece)
                    if len(sentences) > 1:
                        break
                current, size = tail, tail_size
        current.append(unit)
        size += len(unit.text)
    flush()
    return chunks
