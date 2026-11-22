"""Create deterministic demo files for the UI and tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workmate.config import get_settings


def create_sales_excel(path: Path) -> None:
    random.seed(2026)
    products = [
        ("智能办公套件", 1299),
        ("数据分析服务", 2800),
        ("企业培训包", 1800),
        ("协同管理模块", 2400),
    ]
    regions = ["华北", "华东", "华南", "西部"]
    salespeople = ["陈晨", "李然", "王宁", "赵欣"]
    rows = []
    for month in range(1, 13):
        for index in range(8):
            product, unit_price = random.choice(products)
            quantity = random.randint(2, 12)
            seasonal = 1 + (0.12 if month in {3, 6, 9, 12} else 0)
            amount = round(unit_price * quantity * seasonal, 2)
            day = min(28, 3 + index * 3)
            rows.append(
                {
                    "订单编号": f"SO2026{month:02d}{index + 1:03d}",
                    "日期": f"2026-{month:02d}-{day:02d}",
                    "产品": product,
                    "区域": random.choice(regions),
                    "销售员": random.choice(salespeople),
                    "数量": quantity,
                    "单价": unit_price,
                    "销售额": amount,
                }
            )
    pd.DataFrame(rows).to_excel(path, index=False)


def create_docx(path: Path) -> None:
    document = Document()
    document.add_heading("WorkMate AI 项目说明", level=1)
    document.add_paragraph(
        "本项目用于验证智能办公助手的文档分析能力。文档包含项目目标、"
        "交付范围和验收要求。"
    )
    document.add_heading("交付范围", level=2)
    document.add_paragraph("智能聊天、文件分析、自动任务执行和历史记录。")
    document.add_paragraph("任务执行结果必须支持报告、图表和数据文件下载。")
    document.add_heading("验收信息", level=2)
    table = document.add_table(rows=1, cols=3)
    table.rows[0].cells[0].text = "角色"
    table.rows[0].cells[1].text = "负责人"
    table.rows[0].cells[2].text = "截止日期"
    row = table.add_row().cells
    row[0].text = "技术验收"
    row[1].text = "张工"
    row[2].text = "2026-12-15"
    document.save(path)


def create_pdf(path: Path) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path))
    pdf.setFont("STSong-Light", 16)
    pdf.drawString(64, 790, "WorkMate AI 示例合同摘要")
    pdf.setFont("STSong-Light", 11)
    lines = [
        "合同金额：人民币 128,000 元。",
        "项目周期：2026-12-01 至 2027-03-31。",
        "验收负责人：李经理。",
        "付款方式：验收通过后 15 个工作日内付款。",
        "风险提示：需求变更需双方书面确认。",
    ]
    y = 740
    for line in lines:
        pdf.drawString(64, y, line)
        y -= 30
    pdf.save()


def create_txt(path: Path) -> None:
    path.write_text(
        "WorkMate AI 周会纪要\n"
        "1. 完成销售 Excel 自动分析流程。\n"
        "2. 使用 PyTorch 分类器选择安全工具。\n"
        "3. 下周三前完成演示环境验收。\n"
        "负责人：产品研发组\n",
        encoding="utf-8",
    )


def main() -> None:
    settings = get_settings()
    output = settings.data_dir / "demo"
    output.mkdir(parents=True, exist_ok=True)
    create_sales_excel(output / "demo_sales_2026.xlsx")
    create_docx(output / "demo_project_note.docx")
    create_pdf(output / "demo_contract.pdf")
    create_txt(output / "demo_meeting_notes.txt")
    print(f"DEMO_DATA={output}")
    for path in sorted(output.iterdir()):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
