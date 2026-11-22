"""Native Streamlit presentation helpers."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Theme colors are defined in .streamlit/config.toml."""


def render_hero(provider_label: str, model_ok: bool) -> None:
    with st.container(border=True):
        badge_text = "模型在线" if model_ok else "模型待连接"
        badge_icon = ":material/check_circle:" if model_ok else ":material/error:"
        badge_color = "green" if model_ok else "orange"
        st.badge(
            badge_text,
            icon=badge_icon,
            color=badge_color,
        )
        st.title("WorkMate AI", icon=":material/robot_2:")
        st.subheader("智能办公任务助手")
        st.caption(
            "多轮对话、文档理解、PyTorch 任务规划与白名单工具执行"
            f" · 当前模型：{provider_label}"
        )
        columns = st.columns(4)
        columns[0].metric("智能聊天", "多轮上下文")
        columns[1].metric("文件分析", "4 类格式")
        columns[2].metric("自动任务", "5 步工具链")
        columns[3].metric("历史留痕", "SQLite")
