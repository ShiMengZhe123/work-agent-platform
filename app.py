"""WorkMate AI - enterprise office task assistant."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from workmate.agent import AgentError, AgentExecutor, AgentPlanner
from workmate.config import get_settings
from workmate.db import WorkMateDatabase
from workmate.document_service import DocumentAnswer, DocumentService
from workmate.documents import (
    DocumentError,
    ExtractedDocument,
    extract_document,
    extract_key_phrases,
    save_upload,
)
from workmate.intent_model import PyTorchIntentModel
from workmate.llm import LLMClient, LLMConfig, LLMError
from workmate.ui_theme import apply_theme, render_hero


st.set_page_config(
    page_title="WorkMate AI · 智能办公任务助手",
    page_icon=":material/robot_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = get_settings()
apply_theme()


@st.cache_resource(show_spinner=False)
def get_database(db_path: str) -> WorkMateDatabase:
    return WorkMateDatabase(Path(db_path))


@st.cache_resource(show_spinner=False)
def get_intent_model(model_path: str) -> PyTorchIntentModel:
    return PyTorchIntentModel(Path(model_path))


db = get_database(str(settings.database_path))
with st.spinner("正在准备 PyTorch 任务规划模型…"):
    intent_model = get_intent_model(str(settings.intent_model_path))


def ensure_session_state() -> None:
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = db.create_conversation()
    st.session_state.conversation_id = db.ensure_conversation(
        st.session_state.conversation_id
    )
    defaults = {
        "analyzed_document": None,
        "analyzed_file_path": None,
        "file_signature": None,
        "document_summary": None,
        "document_answers": [],
        "task_excel_path": None,
        "task_excel_name": "",
        "last_task_result": None,
        "last_task_plan": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ensure_session_state()


def sidebar_model_client() -> tuple[LLMClient, str]:
    st.sidebar.markdown("### 模型接口")
    provider_index = 0 if settings.llm_provider == "ollama" else 1
    provider = st.sidebar.selectbox(
        "提供方",
        options=["ollama", "openai"],
        index=provider_index,
        format_func=lambda value: (
            "本地 Ollama" if value == "ollama" else "OpenAI 兼容接口"
        ),
        key="sidebar_provider",
    )
    if provider == "ollama":
        base_url = settings.ollama_base_url
        model = settings.ollama_model
        api_key = ""
        with st.sidebar.expander("连接设置", expanded=False):
            base_url = st.text_input(
                "Ollama 地址", value=base_url, key="sidebar_ollama_url"
            ).rstrip("/")
            model = st.text_input(
                "模型名称", value=model, key="sidebar_ollama_model"
            ).strip()
        label = f"Ollama · {model}"
    else:
        base_url = settings.openai_base_url
        model = settings.openai_model
        api_key = settings.openai_api_key
        with st.sidebar.expander("连接设置", expanded=True):
            base_url = st.text_input(
                "Base URL", value=base_url, key="sidebar_openai_url"
            ).rstrip("/")
            model = st.text_input(
                "模型名称", value=model, key="sidebar_openai_model"
            ).strip()
            api_key = st.text_input(
                "API Key",
                value=api_key,
                type="password",
                key="sidebar_openai_key",
                help="仅保存在当前 Streamlit 会话，不写入代码或数据库。",
            ).strip()
        label = f"OpenAI · {model}"
    client = LLMClient(
        LLMConfig(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=settings.llm_timeout,
        )
    )
    return client, label


client, provider_label = sidebar_model_client()
model_ok, model_message = client.health()

st.sidebar.space("medium")
st.sidebar.markdown("### 运行状态")
if model_ok:
    st.sidebar.success(model_message, icon=":material/check_circle:")
else:
    st.sidebar.error(model_message, icon=":material/error:")

model_metadata = intent_model.metadata()
st.sidebar.metric(
    "PyTorch 意图模型",
    f"{float(model_metadata['validation_accuracy']):.0%}",
    help="训练时留出的验证集准确率，用于任务意图分类与工具选择。",
)
st.sidebar.caption(
    f"{model_metadata['framework']} · 词表 {model_metadata['vocabulary_size']}"
)

st.sidebar.space("medium")
st.sidebar.markdown("### 当前对话")
conversations = db.list_conversations(limit=20)
conversation_ids = [row["id"] for row in conversations]
if st.session_state.conversation_id not in conversation_ids:
    conversation_ids.insert(0, st.session_state.conversation_id)


def conversation_label(conversation_id: str) -> str:
    row = next(
        (item for item in conversations if item["id"] == conversation_id),
        None,
    )
    if not row:
        return "当前对话"
    return f"{row['title'][:18]} · {row['updated_at'][5:16].replace('T', ' ')}"


selected_conversation = st.sidebar.selectbox(
    "会话",
    options=conversation_ids,
    index=conversation_ids.index(st.session_state.conversation_id),
    format_func=conversation_label,
    label_visibility="collapsed",
)
if selected_conversation != st.session_state.conversation_id:
    st.session_state.conversation_id = selected_conversation
    st.rerun()

if st.sidebar.button(
    "新建对话",
    icon=":material/add_comment:",
    width="stretch",
):
    st.session_state.conversation_id = db.create_conversation()
    st.rerun()

st.sidebar.caption(
    "数据仅保存在本机 SQLite；默认模型文件位于 D 盘。"
)

render_hero(provider_label, model_ok)

if not model_ok:
    st.warning(
        "当前模型接口未连接。聊天和 AI 文档总结会提示错误；"
        "任务规划、Excel 统计、图表和数据库记录仍可独立运行。"
    )


def render_chat_tab() -> None:
    st.markdown("#### 智能对话")
    st.caption(
        "多轮上下文保存在 SQLite；同一次会话会携带最近消息。"
    )
    messages = db.get_messages(st.session_state.conversation_id, limit=80)
    if not messages:
        st.info(
            "从一个实际问题开始，例如：请给我一个适合技术周会的汇报提纲。"
        )
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["created_at"].replace("T", " "))

    prompt = st.chat_input(
        "输入问题，Enter 发送…",
        key="chat_prompt",
        submit_mode="disable",
    )
    if prompt:
        db.add_message(
            st.session_state.conversation_id,
            "user",
            prompt,
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("WorkMate 正在思考…")
            try:
                history = db.get_messages(
                    st.session_state.conversation_id, limit=16
                )
                llm_messages = [
                    {"role": row["role"], "content": row["content"]}
                    for row in history
                    if row["role"] in {"user", "assistant"}
                ]
                answer = client.chat(llm_messages, temperature=0.35)
                placeholder.markdown(answer)
                db.add_message(
                    st.session_state.conversation_id,
                    "assistant",
                    answer,
                )
                st.rerun()
            except LLMError as exc:
                placeholder.empty()
                st.error(str(exc))
            except Exception as exc:
                placeholder.empty()
                st.error(f"对话处理失败：{exc}")


def render_file_tab() -> None:
    st.markdown("#### 智能文档分析")
    st.caption(
        f"支持 PDF、DOCX、XLSX、XLS、TXT；单文件上限 {settings.max_upload_mb} MB。"
    )
    uploaded = st.file_uploader(
        "上传办公文件",
        type=["pdf", "docx", "xlsx", "xls", "txt"],
        key="analysis_uploader",
        help="文件在本机解析，不会上传到第三方存储。",
    )
    if uploaded is not None:
        signature = (uploaded.name, int(uploaded.size))
        if st.session_state.file_signature != signature:
            try:
                with st.spinner("正在读取文件内容…"):
                    upload_dir = settings.data_dir / "uploads"
                    path = save_upload(uploaded, upload_dir)
                    document = extract_document(path, settings.max_upload_mb)
                    st.session_state.analyzed_file_path = path
                    st.session_state.analyzed_document = document
                    st.session_state.file_signature = signature
                    st.session_state.document_summary = None
                    st.session_state.document_answers = []
                    if document.suffix in {".xlsx", ".xls"}:
                        st.session_state.task_excel_path = path
                        st.session_state.task_excel_name = path.name
            except DocumentError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"文件解析失败：{exc}")

    document: ExtractedDocument | None = st.session_state.analyzed_document
    if document is None:
        with st.container(border=True):
            st.markdown("**可分析内容**")
            st.write("PDF 文本、Word 段落与表格、Excel 多工作表、TXT 文本。")
            st.caption("扫描版 PDF 若没有文字层，会明确提示需要 OCR。")
        return

    info_columns = st.columns(4)
    info_columns[0].metric("文件", document.name)
    info_columns[1].metric("格式", document.suffix.upper().lstrip("."))
    info_columns[2].metric("字符数", f"{document.characters:,}")
    info_columns[3].metric(
        "工作表" if document.dataframes else "处理状态",
        len(document.dataframes) if document.dataframes else "已读取",
    )

    if document.dataframes:
        sheet = st.selectbox(
            "Excel 工作表预览",
            options=list(document.dataframes.keys()),
            key="analysis_sheet",
        )
        st.dataframe(
            document.dataframes[sheet].head(100),
            height=300,
            alt=f"{document.name} 的 {sheet} 工作表预览",
        )
    else:
        with st.expander("查看提取文本", expanded=False):
            st.text_area(
                "文本内容",
                document.text[:20_000],
                height=260,
                disabled=True,
                label_visibility="collapsed",
            )

    action_column, keyword_column = st.columns([1, 2])
    with action_column:
        summarize_clicked = st.button(
            "生成智能摘要",
            type="primary",
            icon=":material/summarize:",
            width="stretch",
        )
    with keyword_column:
        keywords = extract_key_phrases(document.text, 8)
        st.caption("本地关键词提示：" + " · ".join(keywords))

    if summarize_clicked:
        service = DocumentService(client if model_ok else None)
        with st.spinner("正在提取关键信息并生成摘要…"):
            result = service.summarize(document)
        st.session_state.document_summary = result

    summary: DocumentAnswer | None = st.session_state.document_summary
    if summary:
        st.markdown("##### 文档摘要")
        st.markdown(summary.content)
        st.caption(summary.source_note)

    st.markdown("##### 针对文件提问")
    with st.form("document_question_form", clear_on_submit=False):
        question = st.text_input(
            "问题",
            placeholder="例如：合同金额是多少？项目截止日期是什么时候？",
        )
        submitted = st.form_submit_button(
            "提问",
            type="primary",
        )
    if submitted:
        service = DocumentService(client if model_ok else None)
        with st.spinner("正在检索文档并生成回答…"):
            answer = service.answer(document, question)
        st.session_state.document_answers.append(
            {"question": question, "answer": answer}
        )

    for item in reversed(st.session_state.document_answers):
        answer = item["answer"]
        with st.container(border=True):
            st.markdown(f"**你：** {item['question']}")
            st.markdown(f"**WorkMate：** {answer.content}")
            st.caption(answer.source_note)

    text_bytes = document.text.encode("utf-8")
    st.download_button(
        "下载提取文本",
        data=text_bytes,
        file_name=f"{Path(document.name).stem}_extracted.txt",
        mime="text/plain",
    )


def render_task_result(result) -> None:
    metrics = result.metrics
    st.markdown("### 执行结果")
    cols = st.columns(4)
    cols[0].metric("销售额合计", f"{metrics['total_sales']:,.0f}")
    cols[1].metric("月均销售额", f"{metrics['average_monthly']:,.0f}")
    cols[2].metric("峰值月份", metrics["peak_month"])
    cols[3].metric("记录数", metrics["records"])

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### 月度汇总数据")
        st.dataframe(
            result.monthly,
            hide_index=True,
            alt="按月份汇总的销售额数据表",
        )
    with right:
        st.markdown("##### 销售趋势")
        st.image(
            str(result.chart_path),
            width="stretch",
            alt="按月份展示销售额变化的柱状图和趋势线",
        )

    report_path = settings.outputs_dir / f"{result.chart_path.stem}_report.md"
    report_path.write_text(result.report_markdown, encoding="utf-8")
    with st.expander("查看完整分析报告", expanded=True):
        st.markdown(result.report_markdown)

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "下载报告 Markdown",
        data=result.report_markdown.encode("utf-8"),
        file_name=report_path.name,
        mime="text/markdown",
        width="stretch",
    )
    download_columns[1].download_button(
        "下载图表 PNG",
        data=result.chart_path.read_bytes(),
        file_name=result.chart_path.name,
        mime="image/png",
        width="stretch",
    )
    download_columns[2].download_button(
        "下载月度 CSV",
        data=result.csv_path.read_bytes(),
        file_name=result.csv_path.name,
        mime="text/csv",
        width="stretch",
    )


def render_auto_task_tab() -> None:
    st.markdown("#### 智能任务执行")
    st.caption(
        "自然语言 → PyTorch 意图识别 → 白名单计划 → 工具执行 → 图表/报告交付。"
    )

    upload_column, demo_column = st.columns([2, 1])
    with upload_column:
        task_upload = st.file_uploader(
            "上传用于分析的 Excel",
            type=["xlsx", "xls"],
            key="task_uploader",
        )
    with demo_column:
        st.markdown("##### 没有数据？")
        demo_path = settings.data_dir / "demo" / "demo_sales_2026.xlsx"
        if demo_path.exists() and st.button(
            "使用演示 Excel",
            icon=":material/table_view:",
            width="stretch",
        ):
            st.session_state.task_excel_path = demo_path
            st.session_state.task_excel_name = demo_path.name
            st.success("已选择演示数据")

    if task_upload is not None:
        signature = (task_upload.name, int(task_upload.size))
        if st.session_state.get("task_upload_signature") != signature:
            try:
                path = save_upload(task_upload, settings.data_dir / "uploads")
                st.session_state.task_excel_path = path
                st.session_state.task_excel_name = path.name
                st.session_state.task_upload_signature = signature
            except Exception as exc:
                st.error(str(exc))

    excel_path = st.session_state.task_excel_path
    if excel_path:
        st.info(f"当前数据文件：{st.session_state.task_excel_name}")
    else:
        st.warning("请上传 Excel，或点击“使用演示 Excel”。")

    task_text = st.text_area(
        "用自然语言描述任务",
        value=(
            "分析上传的销售Excel，计算各月份销售额，"
            "生成统计图和总结报告。"
        ),
        height=110,
    )
    use_ai_commentary = st.checkbox(
        "使用本地大模型补充分析结论",
        value=True,
        help="统计数字和图表始终由确定性工具计算，大模型只解释真实结果。",
    )
    execute_clicked = st.button(
        "生成计划并执行",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )

    planner = AgentPlanner(intent_model, settings.outputs_dir)
    if execute_clicked:
        if not excel_path:
            st.error("任务需要 Excel 文件。")
        else:
            try:
                plan = planner.plan(
                    task_text,
                    has_excel=True,
                    excel_path=Path(excel_path),
                )
                st.session_state.last_task_plan = plan
            except AgentError as exc:
                st.error(str(exc))

    plan = st.session_state.last_task_plan
    if plan:
        top_columns = st.columns([1, 1, 2])
        top_columns[0].metric(
            "PyTorch 意图",
            plan.intent,
            f"置信度 {plan.confidence:.0%}",
        )
        top_columns[1].metric("规划器", plan.classifier)
        with top_columns[2]:
            st.badge(plan.title, icon=":material/account_tree:", color="blue")
            st.caption(plan.safety_note)

        with st.expander("查看执行计划", expanded=True):
            for step in plan.steps:
                st.markdown(
                    f"**{step.step_no}. {step.name}**  "
                    f"`{step.tool}`  \n{step.description}"
                )

        if execute_clicked and excel_path and plan.intent == "sales_analysis":
            task_id = db.create_task(
                conversation_id=st.session_state.conversation_id,
                title=plan.title,
                user_task=task_text,
                intent=plan.intent,
                confidence=plan.confidence,
                plan_json=plan.to_json(),
            )
            executor = AgentExecutor(
                settings.outputs_dir,
                client if use_ai_commentary and model_ok else None,
            )
            step_map = {step.step_no: step for step in plan.steps}

            def progress(step_no: int, status: str, detail: str) -> None:
                step = step_map.get(step_no)
                db.add_task_step(
                    task_id,
                    step_no,
                    step.tool if step else "unknown",
                    status,
                    detail,
                )
                if step:
                    execution_status.write(f"✅ **{step.name}** · {detail}")

            try:
                with st.status(
                    "Agent 正在执行白名单工具…",
                    type="step",
                    expanded=True,
                ) as execution_status:
                    execution_status.write(
                        f"PyTorch 已识别任务：{plan.intent} "
                        f"（置信度 {plan.confidence:.0%}）"
                    )
                    result = executor.execute_sales(
                        Path(excel_path),
                        progress=progress,
                    )
                    report_path = (
                        settings.outputs_dir
                        / f"{result.chart_path.stem}_report.md"
                    )
                    report_path.write_text(
                        result.report_markdown,
                        encoding="utf-8",
                    )
                    db.finish_task(
                        task_id,
                        status="completed",
                        report_markdown=result.report_markdown,
                        report_path=str(report_path),
                        chart_path=str(result.chart_path),
                        csv_path=str(result.csv_path),
                    )
                    execution_status.update(
                        label="任务执行完成",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.last_task_result = result
            except Exception as exc:
                db.finish_task(task_id, status="failed")
                st.error(f"任务执行失败：{exc}")
        elif execute_clicked and plan.intent != "sales_analysis":
            st.info(
                "当前自动任务页重点演示 Excel 数据分析。"
                "该意图可在“文件分析”中使用文档工具完成。"
            )

    if st.session_state.last_task_result is not None:
        render_task_result(st.session_state.last_task_result)


def render_history_tab() -> None:
    st.markdown("#### 历史记录")
    conversation_tab, task_tab = st.tabs(["对话历史", "任务历史"])
    with conversation_tab:
        histories = db.list_conversations(limit=30)
        if not histories:
            st.info("暂无对话记录。")
        for conversation in histories:
            with st.expander(
                f"{conversation['title']} · "
                f"{conversation['message_count']} 条消息"
            ):
                st.caption(
                    f"创建：{conversation['created_at'].replace('T', ' ')}  "
                    f"更新：{conversation['updated_at'].replace('T', ' ')}"
                )
                for message in db.get_messages(conversation["id"], limit=100):
                    role = "你" if message["role"] == "user" else "WorkMate"
                    st.markdown(f"**{role}：** {message['content']}")

    with task_tab:
        tasks = db.list_tasks(limit=30)
        if not tasks:
            st.info("暂无自动任务记录。")
        for task in tasks:
            status_text = {
                "completed": "已完成",
                "failed": "失败",
                "running": "执行中",
            }.get(task["status"], task["status"])
            with st.expander(
                f"{task['title']} · {status_text} · "
                f"{task['created_at'].replace('T', ' ')}"
            ):
                st.write(
                    f"意图：`{task['intent']}`  "
                    f"置信度：{task['confidence']:.0%}"
                )
                st.caption(task["user_task"])
                for step in db.get_task_steps(task["id"]):
                    st.write(
                        f"{step['step_no']}. `{step['tool_name']}` · "
                        f"{step['status']} · {step['detail'] or ''}"
                    )
                if task.get("report_markdown"):
                    st.markdown(task["report_markdown"])


chat_tab, file_tab, task_tab, history_tab = st.tabs(
    ["智能聊天", "文件分析", "自动任务", "历史记录"]
)

with chat_tab:
    render_chat_tab()
with file_tab:
    render_file_tab()
with task_tab:
    render_auto_task_tab()
with history_tab:
    render_history_tab()

st.caption(
    "WorkMate AI · Python / PyTorch / Streamlit / SQLite · "
    f"本地运行时间 {datetime.now():%Y-%m-%d %H:%M}"
)
