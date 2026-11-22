from pathlib import Path

import pandas as pd
from docx import Document
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from workmate.documents import (
    extract_document,
    select_relevant_chunks,
    split_into_chunks,
)


def test_txt_and_docx_extraction(tmp_path: Path):
    txt = tmp_path / "notes.txt"
    txt.write_text("项目截止日期是2026年12月15日。负责人是张工。", encoding="utf-8")
    document = extract_document(txt)
    assert "张工" in document.text

    docx_path = tmp_path / "note.docx"
    word = Document()
    word.add_paragraph("合同验收负责人是李经理。")
    word.save(docx_path)
    document = extract_document(docx_path)
    assert "李经理" in document.text


def test_excel_extraction(tmp_path: Path):
    path = tmp_path / "sales.xlsx"
    pd.DataFrame(
        {
            "日期": ["2026-01-02", "2026-02-03"],
            "销售额": [100.5, 200.0],
        }
    ).to_excel(path, index=False)
    document = extract_document(path)
    assert document.dataframes
    assert "销售额" in document.text


def test_pdf_extraction_and_chunking(tmp_path: Path):
    path = tmp_path / "contract.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path))
    pdf.setFont("STSong-Light", 12)
    pdf.drawString(72, 760, "合同金额为 128000 元。验收负责人为李经理。")
    pdf.save()
    document = extract_document(path)
    assert "128000" in document.text
    long_text = "第一段项目内容。" * 300
    chunks = split_into_chunks(long_text, size=300, overlap=30)
    assert len(chunks) > 1
    selected = select_relevant_chunks(long_text, "项目内容", limit=2)
    assert selected
