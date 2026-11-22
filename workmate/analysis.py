"""Deterministic, whitelisted Excel sales analysis tools."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class AnalysisError(ValueError):
    """Raised when the workbook cannot support the requested analysis."""


DATE_KEYWORDS = ("日期", "时间", "月份", "年月", "date", "month", "time")
SALES_KEYWORDS = (
    "销售额",
    "金额",
    "营收",
    "收入",
    "成交额",
    "总价",
    "价款",
    "sales",
    "amount",
    "revenue",
)


@dataclass
class ExcelAnalysisResult:
    sheet_name: str
    date_column: str
    sales_column: str
    monthly: pd.DataFrame
    metrics: dict[str, object]
    chart_path: Path
    report_markdown: str
    csv_path: Path


def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("￥", "", regex=False)
        .str.replace("¥", "", regex=False)
        .str.replace("元", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


class ExcelSalesAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_best_sheet(self, path: Path) -> tuple[str, pd.DataFrame]:
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as exc:
            raise AnalysisError(f"Excel 读取失败：{exc}") from exc
        candidates = []
        for name, frame in sheets.items():
            cleaned = frame.dropna(how="all").dropna(axis=1, how="all")
            if not cleaned.empty:
                candidates.append((name, cleaned))
        if not candidates:
            raise AnalysisError("Excel 没有可分析的数据。")
        return max(candidates, key=lambda item: len(item[1]))

    def detect_columns(
        self, frame: pd.DataFrame
    ) -> tuple[str, str, pd.DataFrame]:
        if frame.empty:
            raise AnalysisError("数据表为空。")
        date_column = next(
            (
                str(column)
                for column in frame.columns
                if any(keyword in str(column).lower() for keyword in DATE_KEYWORDS)
            ),
            "",
        )
        if not date_column:
            best_ratio = 0.0
            for column in frame.columns:
                parsed = pd.to_datetime(frame[column], errors="coerce")
                ratio = float(parsed.notna().mean())
                if ratio > best_ratio:
                    best_ratio = ratio
                    date_column = str(column)
            if best_ratio < 0.6:
                raise AnalysisError("未找到可作为日期或月份的列。")

        sales_column = next(
            (
                str(column)
                for column in frame.columns
                if any(keyword in str(column).lower() for keyword in SALES_KEYWORDS)
            ),
            "",
        )
        if not sales_column:
            numeric_columns = []
            for column in frame.columns:
                if column == date_column:
                    continue
                parsed = parse_numeric_series(frame[column])
                ratio = float(parsed.notna().mean())
                if ratio >= 0.7:
                    numeric_columns.append((str(column), ratio))
            excluded = ("编号", "id", "数量", "单价", "折扣", "税率")
            numeric_columns = [
                item
                for item in numeric_columns
                if not any(word in item[0].lower() for word in excluded)
            ]
            if not numeric_columns:
                raise AnalysisError("未找到可计算的销售额或金额列。")
            sales_column = numeric_columns[0][0]

        working = frame.copy()
        working[date_column] = pd.to_datetime(
            working[date_column], errors="coerce"
        )
        working[sales_column] = parse_numeric_series(working[sales_column])
        working = working.dropna(subset=[date_column, sales_column])
        if working.empty:
            raise AnalysisError("日期和销售额列没有有效数据。")
        return date_column, sales_column, working

    def aggregate_monthly(
        self,
        frame: pd.DataFrame,
        date_column: str,
        sales_column: str,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        working = frame.copy()
        working["月份"] = working[date_column].dt.to_period("M").astype(str)
        monthly = (
            working.groupby("月份", as_index=False)[sales_column]
            .sum()
            .rename(columns={sales_column: "销售额"})
        )
        monthly = monthly.sort_values("月份").reset_index(drop=True)
        if monthly.empty:
            raise AnalysisError("按月汇总后没有结果。")
        peak_index = int(monthly["销售额"].idxmax())
        low_index = int(monthly["销售额"].idxmin())
        metrics: dict[str, object] = {
            "records": int(len(frame)),
            "months": int(len(monthly)),
            "total_sales": float(monthly["销售额"].sum()),
            "average_monthly": float(monthly["销售额"].mean()),
            "peak_month": str(monthly.loc[peak_index, "月份"]),
            "peak_sales": float(monthly.loc[peak_index, "销售额"]),
            "low_month": str(monthly.loc[low_index, "月份"]),
            "low_sales": float(monthly.loc[low_index, "销售额"]),
        }
        return monthly, metrics

    def create_chart(
        self, monthly: pd.DataFrame, sales_column_title: str = "月度销售额"
    ) -> Path:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
        fig.patch.set_facecolor("#F7F9FC")
        ax.set_facecolor("#FFFFFF")
        bars = ax.bar(
            monthly["月份"],
            monthly["销售额"],
            color="#3B82F6",
            width=0.62,
        )
        ax.plot(
            monthly["月份"],
            monthly["销售额"],
            color="#EF7C3B",
            marker="o",
            linewidth=2.2,
            markersize=5,
        )
        ax.set_title(sales_column_title, fontsize=18, pad=16, weight="bold")
        ax.set_xlabel("月份")
        ax.set_ylabel("销售额")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, monthly["销售额"]):
            ax.annotate(
                f"{value:,.0f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
            )
        fig.tight_layout()
        path = self.output_dir / f"sales_chart_{uuid.uuid4().hex[:10]}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def build_report(
        self,
        *,
        file_name: str,
        sheet_name: str,
        monthly: pd.DataFrame,
        metrics: dict[str, object],
        ai_commentary: str = "",
    ) -> str:
        lines = [
            f"# 销售数据分析报告",
            "",
            f"- 数据文件：`{file_name}`",
            f"- 工作表：`{sheet_name}`",
            f"- 有效记录：{metrics['records']} 条",
            f"- 覆盖月份：{metrics['months']} 个",
            "",
            "## 核心指标",
            "",
            f"- 销售额合计：**{metrics['total_sales']:,.2f}**",
            f"- 月均销售额：**{metrics['average_monthly']:,.2f}**",
            (
                f"- 峰值月份：**{metrics['peak_month']}**，"
                f"{metrics['peak_sales']:,.2f}"
            ),
            (
                f"- 最低月份：**{metrics['low_month']}**，"
                f"{metrics['low_sales']:,.2f}"
            ),
            "",
            "## 月度汇总",
            "",
            "| 月份 | 销售额 |",
            "|---|---:|",
        ]
        for _, row in monthly.iterrows():
            lines.append(f"| {row['月份']} | {row['销售额']:,.2f} |")
        if ai_commentary.strip():
            lines.extend(["", "## 分析结论", "", ai_commentary.strip()])
        lines.extend(
            [
                "",
                "## 执行说明",
                "",
                "本报告由白名单数据分析工具自动生成。日期列和销售额列均经过"
                "本地规则校验，未执行模型生成的 Python 代码或系统命令。",
            ]
        )
        return "\n".join(lines)

    def analyze(
        self,
        path: Path,
        *,
        ai_commentary: str = "",
        progress=None,
    ) -> ExcelAnalysisResult:
        def notify(step: int, label: str, detail: str) -> None:
            if progress:
                progress(step, label, detail)

        notify(1, "读取Excel", "正在读取工作簿")
        sheet_name, frame = self.load_best_sheet(path)
        notify(2, "识别字段", "正在定位日期列和销售额列")
        date_column, sales_column, working = self.detect_columns(frame)
        notify(3, "聚合销售", "正在按月份汇总销售额")
        monthly, metrics = self.aggregate_monthly(
            working, date_column, sales_column
        )
        notify(4, "生成图表", "正在生成月度销售趋势图")
        chart_path = self.create_chart(monthly)
        notify(5, "生成报告", "正在整理统计指标和可下载文件")
        csv_path = self.output_dir / f"sales_monthly_{uuid.uuid4().hex[:10]}.csv"
        monthly.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report = self.build_report(
            file_name=path.name,
            sheet_name=sheet_name,
            monthly=monthly,
            metrics=metrics,
            ai_commentary=ai_commentary,
        )
        return ExcelAnalysisResult(
            sheet_name=sheet_name,
            date_column=date_column,
            sales_column=sales_column,
            monthly=monthly,
            metrics=metrics,
            chart_path=chart_path,
            report_markdown=report,
            csv_path=csv_path,
        )
