from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_with_four_core_workflows():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=40).run()
    assert not app.exception
    tab_labels = {tab.label for tab in app.tabs}
    assert {"智能聊天", "文件分析", "自动任务", "历史记录"}.issubset(tab_labels)
    assert len(app.chat_input) >= 1
    assert len(app.file_uploader) >= 2
