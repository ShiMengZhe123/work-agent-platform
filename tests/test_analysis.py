from pathlib import Path

import pandas as pd

from workmate.analysis import ExcelSalesAnalyzer
from workmate.agent import AgentExecutor, AgentPlanner
from workmate.intent_model import PyTorchIntentModel


def make_sales_file(path: Path) -> None:
    pd.DataFrame(
        {
            "日期": [
                "2026-01-02",
                "2026-01-20",
                "2026-02-05",
                "2026-03-08",
            ],
            "产品": ["A", "B", "A", "B"],
            "销售额": [100, 200, 150, 250],
        }
    ).to_excel(path, index=False)


def test_sales_analysis_outputs_chart_and_report(tmp_path: Path):
    source = tmp_path / "sales.xlsx"
    make_sales_file(source)
    analyzer = ExcelSalesAnalyzer(tmp_path / "outputs")
    result = analyzer.analyze(source)
    assert result.metrics["total_sales"] == 700
    assert result.metrics["peak_month"] == "2026-01"
    assert result.chart_path.exists()
    assert result.csv_path.exists()
    assert "销售额合计" in result.report_markdown


def test_agent_plan_is_whitelisted(tmp_path: Path):
    source = tmp_path / "sales.xlsx"
    make_sales_file(source)
    model = PyTorchIntentModel(tmp_path / "intent.pt", force_train=True)
    planner = AgentPlanner(model, tmp_path / "outputs")
    plan = planner.plan(
        "分析上传的销售Excel，计算各月份销售额，生成统计图。",
        has_excel=True,
        excel_path=source,
    )
    assert plan.intent == "sales_analysis"
    assert [step.tool for step in plan.steps] == [
        "load_excel",
        "detect_columns",
        "aggregate_monthly",
        "render_chart",
        "build_report",
    ]
    result = AgentExecutor(tmp_path / "outputs").execute_sales(source)
    assert result.monthly["销售额"].sum() == 700
