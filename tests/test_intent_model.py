from pathlib import Path

from workmate.intent_model import PyTorchIntentModel


def test_pytorch_intent_model_trains_and_predicts(tmp_path: Path):
    model_path = tmp_path / "intent.pt"
    model = PyTorchIntentModel(model_path, force_train=True)
    assert model_path.exists()
    prediction = model.predict(
        "分析上传的销售Excel，计算各月份销售额，生成统计图和总结报告。"
    )
    assert prediction.label == "sales_analysis"
    assert 0 < prediction.confidence <= 1

    reloaded = PyTorchIntentModel(model_path)
    second = reloaded.predict("总结这份PDF的主要结论")
    assert second.label == "document_summary"
    assert reloaded.metadata()["framework"].startswith("PyTorch")


def test_intent_model_has_four_safe_classes(tmp_path: Path):
    model = PyTorchIntentModel(tmp_path / "intent.pt", force_train=True)
    assert set(model.predict("你好").probabilities) == {
        "general_chat",
        "document_summary",
        "document_qa",
        "sales_analysis",
    }
