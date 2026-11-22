"""Safe task planning and whitelisted tool execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from workmate.analysis import ExcelAnalysisResult, ExcelSalesAnalyzer
from workmate.intent_model import IntentPrediction, PyTorchIntentModel
from workmate.llm import LLMClient


class AgentError(ValueError):
    """Raised when a task cannot be safely planned or executed."""


@dataclass(frozen=True)
class AgentStep:
    step_no: int
    name: str
    tool: str
    description: str


@dataclass(frozen=True)
class AgentPlan:
    title: str
    intent: str
    confidence: float
    classifier: str
    steps: list[AgentStep]
    safety_note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class AgentPlanner:
    """Uses a real PyTorch classifier and bounded rules to make plans."""

    def __init__(
        self,
        intent_model: PyTorchIntentModel,
        output_dir: Path,
    ):
        self.intent_model = intent_model
        self.output_dir = output_dir

    def _fallback_intent(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in ("月份", "销售额", "excel", "统计图")):
            return "sales_analysis"
        if any(word in lower for word in ("回答", "哪个", "哪里", "有没有")):
            return "document_qa"
        if any(word in lower for word in ("总结", "摘要", "概括")):
            return "document_summary"
        return "general_chat"

    def plan(
        self,
        task_text: str,
        *,
        has_excel: bool,
        excel_path: Path | None,
    ) -> AgentPlan:
        task_text = task_text.strip()
        if not task_text:
            raise AgentError("请输入任务内容。")

        prediction: IntentPrediction = self.intent_model.predict(task_text)
        intent = prediction.label
        if prediction.confidence < 0.42:
            intent = self._fallback_intent(task_text)

        if has_excel and excel_path is not None:
            if any(
                keyword in task_text
                for keyword in ("销售", "金额", "月份", "统计图", "报表", "Excel")
            ):
                intent = "sales_analysis"

        if intent == "sales_analysis":
            if not has_excel or excel_path is None:
                raise AgentError(
                    "该任务需要 Excel 文件。请先在“自动任务”页上传销售数据。"
                )
            steps = [
                AgentStep(1, "校验工作簿", "load_excel", "读取并校验 Excel 数据"),
                AgentStep(2, "识别业务字段", "detect_columns", "识别日期与销售额列"),
                AgentStep(3, "计算月度汇总", "aggregate_monthly", "按月份汇总销售额"),
                AgentStep(4, "生成统计图", "render_chart", "生成月度销售趋势图"),
                AgentStep(5, "生成报告", "build_report", "生成报告与下载文件"),
            ]
            title = "销售Excel自动分析"
        elif intent in {"document_summary", "document_qa"}:
            title = "文档理解与分析"
            steps = [
                AgentStep(1, "校验文件", "validate_document", "检查格式与大小"),
                AgentStep(2, "提取内容", "extract_document", "读取文档文本"),
                AgentStep(3, "检索相关片段", "retrieve_context", "定位相关内容"),
                AgentStep(4, "生成回答", "llm_document_answer", "调用本地模型生成结果"),
            ]
        else:
            title = "通用办公问答"
            steps = [
                AgentStep(1, "理解问题", "pytorch_intent", "使用 PyTorch 分类意图"),
                AgentStep(2, "生成回答", "llm_chat", "调用已配置的语言模型"),
            ]

        return AgentPlan(
            title=title,
            intent=intent,
            confidence=prediction.confidence,
            classifier=(
                "PyTorch IntentClassifier"
                if prediction.confidence >= 0.42
                else "PyTorch + 安全关键词回退"
            ),
            steps=steps,
            safety_note=(
                "仅允许调用白名单工具；不会执行模型生成的系统命令、"
                "动态 Python 代码或任意文件操作。"
            ),
        )


ProgressCallback = Callable[[int, str, str], None]


class AgentExecutor:
    def __init__(
        self,
        output_dir: Path,
        llm_client: LLMClient | None = None,
    ):
        self.output_dir = output_dir
        self.llm_client = llm_client

    def _ai_commentary(self, metrics: dict[str, object]) -> str:
        if not self.llm_client:
            return ""
        prompt = (
            "请根据以下真实统计结果，用不超过180字写销售分析结论。"
            "不要编造数据。\n"
            f"销售额合计={metrics['total_sales']:.2f}，"
            f"月均={metrics['average_monthly']:.2f}，"
            f"峰值月份={metrics['peak_month']}，"
            f"峰值={metrics['peak_sales']:.2f}，"
            f"最低月份={metrics['low_month']}，"
            f"最低={metrics['low_sales']:.2f}。"
        )
        try:
            return self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=260,
            )
        except Exception:
            return "本地模型分析不可用；以下结论基于经过校验的确定性统计结果。"

    def execute_sales(
        self,
        excel_path: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> ExcelAnalysisResult:
        analyzer = ExcelSalesAnalyzer(self.output_dir)
        sheet_name, frame = analyzer.load_best_sheet(excel_path)
        if progress:
            progress(1, "completed", f"已读取工作表：{sheet_name}")
        date_column, sales_column, working = analyzer.detect_columns(frame)
        if progress:
            progress(
                2,
                "completed",
                f"日期列：{date_column}；销售额列：{sales_column}",
            )
        monthly, metrics = analyzer.aggregate_monthly(
            working, date_column, sales_column
        )
        if progress:
            progress(3, "completed", f"已汇总 {len(monthly)} 个月")
        ai_commentary = self._ai_commentary(metrics)
        chart_path = analyzer.create_chart(monthly)
        if progress:
            progress(4, "completed", f"图表已生成：{chart_path.name}")
        csv_path = self.output_dir / f"sales_monthly_{excel_path.stem}.csv"
        monthly.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report = analyzer.build_report(
            file_name=excel_path.name,
            sheet_name=sheet_name,
            monthly=monthly,
            metrics=metrics,
            ai_commentary=ai_commentary,
        )
        if progress:
            progress(5, "completed", "报告和下载文件已生成")
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
