"""Safe local document extraction and retrieval."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from docx import Document as WordDocument
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xls", ".txt"}


class DocumentError(ValueError):
    """Raised when an uploaded document cannot be processed."""


@dataclass
class ExtractedDocument:
    name: str
    suffix: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def characters(self) -> int:
        return len(self.text)


def save_upload(uploaded_file, destination_dir: Path) -> Path:
    """Persist an uploaded file with a sanitized basename."""
    filename = Path(uploaded_file.name).name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            "不支持该文件格式，请上传 PDF、DOCX、XLSX、XLS 或 TXT。"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / filename
    counter = 1
    while target.exists():
        target = destination_dir / f"{Path(filename).stem}_{counter}{suffix}"
        counter += 1
    with target.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target


def validate_file_size(path: Path, max_mb: int) -> None:
    size = path.stat().st_size
    if size > max_mb * 1024 * 1024:
        raise DocumentError(f"文件超过 {max_mb} MB 限制。")
    if size == 0:
        raise DocumentError("上传文件为空。")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(path: Path) -> tuple[str, dict[str, object]]:
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages).strip()
        if not text:
            raise DocumentError(
                "PDF 未提取到可读文字；它可能是扫描件，需要先进行 OCR。"
            )
        return text, {"pages": len(reader.pages)}
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"PDF 读取失败：{exc}") from exc


def _extract_docx(path: Path) -> tuple[str, dict[str, object]]:
    try:
        document = WordDocument(str(path))
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
        for table_index, table in enumerate(document.tables, start=1):
            parts.append(f"\n[表格 {table_index}]")
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(parts).strip()
        if not text:
            raise DocumentError("Word 文档中没有可读取的文字。")
        return text, {
            "paragraphs": len(document.paragraphs),
            "tables": len(document.tables),
        }
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Word 读取失败：{exc}") from exc


def _extract_excel(
    path: Path,
) -> tuple[str, dict[str, object], dict[str, pd.DataFrame]]:
    try:
        frames = pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        raise DocumentError(f"Excel 读取失败：{exc}") from exc
    if not frames:
        raise DocumentError("Excel 工作簿中没有工作表。")

    sections: list[str] = []
    metadata: dict[str, object] = {"sheets": len(frames)}
    clean_frames: dict[str, pd.DataFrame] = {}
    for sheet_name, frame in frames.items():
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        clean_frames[str(sheet_name)] = frame
        sections.append(
            f"[工作表：{sheet_name}] 行数={len(frame)} 列数={len(frame.columns)}"
        )
        sections.append("列名：" + "、".join(str(column) for column in frame.columns))
        if not frame.empty:
            sections.append(
                frame.head(30).to_csv(index=False, lineterminator="\n")
            )
    text = "\n".join(sections).strip()
    if not text:
        raise DocumentError("Excel 工作簿中没有可读取内容。")
    return text, metadata, clean_frames


def extract_document(path: Path, max_mb: int = 20) -> ExtractedDocument:
    validate_file_size(path, max_mb)
    suffix = path.suffix.lower()
    dataframes: dict[str, pd.DataFrame] = {}
    if suffix == ".pdf":
        text, metadata = _extract_pdf(path)
    elif suffix == ".docx":
        text, metadata = _extract_docx(path)
    elif suffix in {".xlsx", ".xls"}:
        text, metadata, dataframes = _extract_excel(path)
    elif suffix == ".txt":
        try:
            text = _decode_text(path.read_bytes()).strip()
        except OSError as exc:
            raise DocumentError(f"TXT 读取失败：{exc}") from exc
        if not text:
            raise DocumentError("TXT 文件没有可读取文本。")
        metadata = {"lines": len(text.splitlines())}
    else:
        raise DocumentError("不支持该文件格式。")

    text = text[:120_000]
    metadata["characters"] = len(text)
    metadata["truncated"] = len(text) >= 120_000
    return ExtractedDocument(
        name=path.name,
        suffix=suffix,
        text=text,
        metadata=metadata,
        dataframes=dataframes,
    )


def split_into_chunks(text: str, size: int = 1800, overlap: int = 220) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            paragraph_end = text.rfind("\n", start + int(size * 0.6), end)
            if paragraph_end > start:
                end = paragraph_end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def select_relevant_chunks(
    text: str, question: str, limit: int = 4
) -> list[str]:
    chunks = split_into_chunks(text)
    question_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", question))
    scored = []
    for index, chunk in enumerate(chunks):
        score = sum(chunk.count(token) for token in question_tokens)
        score += 0.01 * len(chunk)
        scored.append((score, index, chunk))
    scored.sort(reverse=True)
    selected = sorted(scored[:limit], key=lambda item: item[1])
    return [item[2] for item in selected]


def extract_key_phrases(text: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
    stop_words = {
        "我们",
        "你们",
        "他们",
        "以及",
        "进行",
        "一个",
        "可以",
        "需要",
        "相关",
        "工作",
        "项目",
        "文件",
        "内容",
    }
    counts = Counter(word for word in words if word not in stop_words)
    return [word for word, _ in counts.most_common(limit)]


def extractive_summary(text: str, max_sentences: int = 6) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？!?；;])|\n+", text)
        if len(sentence.strip()) >= 12
    ]
    if not sentences:
        return text[:800]
    keywords = set(extract_key_phrases(text, 20))
    scored = []
    for index, sentence in enumerate(sentences):
        score = sum(sentence.count(keyword) for keyword in keywords)
        scored.append((score, index, sentence))
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    ordered = sorted(top, key=lambda item: item[1])
    return "".join(item[2] if item[2][-1] in "。！？!?" else item[2] + "。" for item in ordered)


def read_uploaded_bytes(uploaded_file) -> bytes:
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)
    if isinstance(uploaded_file, BinaryIO):
        return uploaded_file.read()
    return uploaded_file.getbuffer().tobytes()
