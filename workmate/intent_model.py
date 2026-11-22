"""Small, real PyTorch model used to classify office-task intent."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


LABELS = [
    "general_chat",
    "document_summary",
    "document_qa",
    "sales_analysis",
]

TRAIN_EXAMPLES = [
    ("你好，介绍一下你能做什么", "general_chat"),
    ("帮我写一封会议通知", "general_chat"),
    ("怎样提高团队沟通效率", "general_chat"),
    ("给我一个周报模板", "general_chat"),
    ("解释一下什么是智能体", "general_chat"),
    ("你能做什么办公任务", "general_chat"),
    ("帮我拟定项目复盘提纲", "general_chat"),
    ("给我三条客户沟通建议", "general_chat"),
    ("把下面内容润色得更正式", "general_chat"),
    ("为团队生成一个季度目标模板", "general_chat"),
    ("告诉我怎么做风险复盘", "general_chat"),
    ("提供一个会议议程框架", "general_chat"),
    ("总结这份文档", "document_summary"),
    ("阅读上传的PDF并提炼关键信息", "document_summary"),
    ("分析这个Word文件的主要内容", "document_summary"),
    ("把上传的TXT总结成要点", "document_summary"),
    ("这份材料讲了什么", "document_summary"),
    ("生成文档摘要和关键词", "document_summary"),
    ("提取附件中的重点结论", "document_summary"),
    ("概述上传文件", "document_summary"),
    ("帮我把附件压缩成五条摘要", "document_summary"),
    ("分析文档的主要观点和行动项", "document_summary"),
    ("快速浏览材料并生成关键信息", "document_summary"),
    ("提炼这份报告的结论", "document_summary"),
    ("针对文件回答问题", "document_qa"),
    ("根据文档告诉我合同金额", "document_qa"),
    ("上传的PDF里截止日期是哪天", "document_qa"),
    ("这份Word中谁负责验收", "document_qa"),
    ("查一下文档里的项目范围", "document_qa"),
    ("文件里有没有提到风险", "document_qa"),
    ("从附件中找出联系人", "document_qa"),
    ("根据材料回答问题", "document_qa"),
    ("文档规定的付款时间是什么", "document_qa"),
    ("根据合同找出违约责任", "document_qa"),
    ("附件里是否说明了交付地点", "document_qa"),
    ("从报告里查询项目负责人", "document_qa"),
    ("分析上传的销售Excel", "sales_analysis"),
    ("计算各月份销售额", "sales_analysis"),
    ("生成销售统计图和总结报告", "sales_analysis"),
    ("分析Excel中的销售额趋势", "sales_analysis"),
    ("按月份汇总营收数据", "sales_analysis"),
    ("统计表格里的销售金额", "sales_analysis"),
    ("找出销售最高的月份", "sales_analysis"),
    ("读取Excel并计算月度销售合计", "sales_analysis"),
    ("对销售明细做数据分析和图表", "sales_analysis"),
    ("计算订单金额并按月生成报表", "sales_analysis"),
    ("分析营收表并输出结论", "sales_analysis"),
    ("把Excel数据画成趋势图", "sales_analysis"),
    ("按区域汇总订单金额", "sales_analysis"),
    ("统计每个销售员的业绩", "sales_analysis"),
    ("读取表格并计算收入合计", "sales_analysis"),
    ("分析销售趋势并输出图表", "sales_analysis"),
]


def _normalize(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^\u4e00-\u9fffa-z0-9]+", "", text)


def tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    chars = list(normalized)
    bigrams = [
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    ]
    latin_words = re.findall(r"[a-z0-9]+", text.lower())
    return chars + bigrams + latin_words


def build_vocabulary(samples: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for sample in samples:
        for token in tokenize(sample):
            counts[token] = counts.get(token, 0) + 1
    tokens = [
        token
        for token, count in counts.items()
        if count >= 1 and len(token) <= 8
    ]
    return sorted(tokens)


def vectorize(text: str, vocabulary: list[str]) -> torch.Tensor:
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    vector = torch.zeros(len(vocabulary), dtype=torch.float32)
    for token in tokenize(text):
        index = token_to_index.get(token)
        if index is not None:
            vector[index] += 1.0
    norm = vector.norm()
    if norm > 0:
        vector /= norm
    return vector


class IntentClassifier(nn.Module):
    def __init__(self, input_size: int, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str = "pytorch"


class PyTorchIntentModel:
    def __init__(
        self,
        model_path: Path,
        *,
        force_train: bool = False,
    ):
        self.model_path = model_path
        self.vocabulary: list[str] = []
        self.model: IntentClassifier | None = None
        self.validation_accuracy = 0.0
        if force_train or not model_path.exists():
            self.train()
        else:
            self.load()

    def _build_training_tensors(
        self, rows: list[tuple[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_index = {label: index for index, label in enumerate(LABELS)}
        features = torch.stack(
            [vectorize(text, self.vocabulary) for text, _ in rows]
        )
        targets = torch.tensor(
            [label_index[label] for _, label in rows],
            dtype=torch.long,
        )
        return features, targets

    def train(self) -> dict[str, float]:
        torch.manual_seed(2026)
        random.seed(2026)
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

        grouped: dict[str, list[tuple[str, str]]] = {}
        for row in TRAIN_EXAMPLES:
            grouped.setdefault(row[1], []).append(row)
        validation_rows = []
        train_rows = []
        for label in LABELS:
            rows = grouped[label]
            validation_rows.extend(rows[:2])
            train_rows.extend(rows[2:])
        random.shuffle(train_rows)
        self.vocabulary = build_vocabulary([text for text, _ in train_rows])

        features, targets = self._build_training_tensors(train_rows)
        self.model = IntentClassifier(len(self.vocabulary), len(LABELS))
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.025, weight_decay=0.002
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        final_loss = 0.0
        for _ in range(260):
            optimizer.zero_grad()
            logits = self.model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

        self.model.eval()
        with torch.no_grad():
            validation_features, validation_targets = (
                self._build_training_tensors(validation_rows)
            )
            predictions = self.model(validation_features).argmax(dim=1)
            self.validation_accuracy = float(
                (predictions == validation_targets).float().mean()
            )
        self.save(loss=final_loss)
        return {
            "loss": final_loss,
            "validation_accuracy": self.validation_accuracy,
            "vocabulary_size": float(len(self.vocabulary)),
        }

    def save(self, *, loss: float = 0.0) -> None:
        if self.model is None:
            raise RuntimeError("模型尚未创建")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "vocabulary": self.vocabulary,
                "labels": LABELS,
                "validation_accuracy": self.validation_accuracy,
                "training_loss": loss,
                "framework": f"PyTorch {torch.__version__}",
            },
            self.model_path,
        )

    def load(self) -> None:
        checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=True
        )
        labels = checkpoint.get("labels", [])
        if labels != LABELS:
            raise RuntimeError("PyTorch 意图模型标签与代码不兼容")
        self.vocabulary = list(checkpoint["vocabulary"])
        self.validation_accuracy = float(
            checkpoint.get("validation_accuracy", 0.0)
        )
        self.model = IntentClassifier(len(self.vocabulary), len(LABELS))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, text: str) -> IntentPrediction:
        if self.model is None:
            self.load()
        assert self.model is not None
        with torch.no_grad():
            logits = self.model(vectorize(text, self.vocabulary).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
        best_index = int(probabilities.argmax().item())
        probability_map = {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(LABELS)
        }
        return IntentPrediction(
            label=LABELS[best_index],
            confidence=probability_map[LABELS[best_index]],
            probabilities=probability_map,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "framework": f"PyTorch {torch.__version__}",
            "validation_accuracy": self.validation_accuracy,
            "vocabulary_size": len(self.vocabulary),
            "labels": LABELS,
        }
