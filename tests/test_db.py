from pathlib import Path

from workmate.db import WorkMateDatabase


def test_conversation_and_task_history(tmp_path: Path):
    database = WorkMateDatabase(tmp_path / "test.db")
    conversation_id = database.create_conversation()
    database.add_message(conversation_id, "user", "请分析销售数据")
    database.add_message(conversation_id, "assistant", "已收到")
    messages = database.get_messages(conversation_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]

    task_id = database.create_task(
        conversation_id=conversation_id,
        title="销售分析",
        user_task="分析Excel",
        intent="sales_analysis",
        confidence=0.96,
        plan_json="{}",
    )
    database.add_task_step(
        task_id, 1, "load_excel", "completed", "读取成功"
    )
    database.finish_task(task_id, status="completed")
    tasks = database.list_tasks()
    assert tasks[0]["status"] == "completed"
    assert database.get_task_steps(task_id)[0]["detail"] == "读取成功"
