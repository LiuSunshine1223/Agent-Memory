"""High-precision extraction for discrete user memories.

The extractor deliberately prefers missing a vague memory over storing a wrong one.
It only reads the user's original sentence; model-generated answers never become facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class MemoryType(str, Enum):
    PROFILE = "profile"
    GOAL = "goal"
    PREFERENCE = "preference"
    RESEARCH = "research"
    PROJECT = "project"


@dataclass(frozen=True)
class ExtractedMemory:
    memory_type: MemoryType
    slot: str
    value: str
    evidence: str = ""


class DiscreteMemoryExtractor:
    """Extract a small, explainable set of long-term fact slots."""

    COLLECTION_SLOTS = frozenset({"preference.like", "preference.dislike"})

    @classmethod
    def cardinality_for_slot(cls, slot: str) -> str:
        return "collection" if slot in cls.COLLECTION_SLOTS else "scalar"

    @classmethod
    def has_non_user_context(cls, text: str) -> bool:
        """Conservatively exclude quotations, reported speech and examples.

        This rule extractor does not try to resolve the speaker of a quote.
        A separate, unquoted first-person assertion can be saved instead.
        """
        return bool(
            any(mark in text for mark in ('"', '“', '”', '‘', '’', '「', '」', '『', '』', '`'))
            or re.search(r"(?:他说|她说|对方说|朋友说|同学说|有人说|转述|引用|台词|例句|说[：:])", text)
            or any(marker in text for marker in
                   cls._UNCERTAIN_CONTEXT_MARKERS + cls._DOCUMENT_CONTEXT_MARKERS)
        )

    _RULES = (
        (
            MemoryType.PROFILE,
            "profile.name",
            re.compile(r"(?:我叫|我的名字是)\s*([^\s，。！？,!?]{1,30})"),
        ),
        (
            MemoryType.PROFILE,
            "profile.school",
            re.compile(r"(?:我在)\s*([^，。！？,!?]{2,50}?)(?:读书|上学|就读)"),
        ),
        (
            MemoryType.PROFILE,
            "profile.school",
            re.compile(r"(?:我就读于)\s*([^，。！？,!?]{2,50})"),
        ),
        (
            MemoryType.GOAL,
            "career.goal",
            re.compile(r"(?:我的(?:求职)?目标是|我的目标[：:]?)\s*([^，。！？,!?]{2,120})"),
        ),
        (
            MemoryType.GOAL,
            "career.plan",
            re.compile(r"(?:我的计划是|我计划|我打算)\s*([^，。！？,!?]{2,120})"),
        ),
        (
            MemoryType.PREFERENCE,
            "preference.like",
            re.compile(r"(?:我喜欢|我的偏好是)\s*([^，。！？,!?]{1,120})"),
        ),
        (
            MemoryType.PREFERENCE,
            "preference.dislike",
            re.compile(r"(?:我不喜欢|我讨厌)\s*([^，。！？,!?]{1,120})"),
        ),
        (
            MemoryType.RESEARCH,
            "research.topic",
            re.compile(
                r"(?:我的研究方向是|我的论文(?:方向|工作|主题)是|我目前的论文工作是)"
                r"\s*([^，。！？,!?]{2,120})"
            ),
        ),
        (
            MemoryType.PROJECT,
            "project.focus",
            re.compile(
                r"(?:我的项目(?:方向|重点|核心)是|这个项目以)\s*"
                r"([^，。！？,!?]{2,120})"
            ),
        ),
    )

    _FORGET_SLOT_KEYWORDS = (
        ("名字", ("profile.name",)),
        ("姓名", ("profile.name",)),
        ("学校", ("profile.school",)),
        ("目标", ("career.goal",)),
        ("计划", ("career.plan",)),
        ("偏好", ("preference.like", "preference.dislike")),
        ("不喜欢", ("preference.dislike",)),
        ("喜欢", ("preference.like",)),
        ("研究方向", ("research.topic",)),
        ("论文", ("research.topic",)),
        ("项目", ("project.focus",)),
    )

    _NON_ASSERTIVE_MARKERS = (
        "我想知道",
        "我想问",
        "如果",
        "假如",
        "假设",
        "比如",
        "例如",
        "是否",
        "是不是",
        "什么",
        "谁",
        "怎么",
        "如何",
        "为什么",
        "不确定",
        "不是",
        "并非",
        "还没决定",
        "尚未决定",
    )

    _UNCERTAIN_CONTEXT_MARKERS = (
        "如果", "假如", "假设", "比如", "例如", "不确定",
        "还没决定", "尚未决定", "举个例子", "举例",
    )
    _DOCUMENT_CONTEXT_MARKERS = (
        "根据文档", "根据资料", "文档里", "文档中", "文档写着",
        "文档写道", "资料里", "资料中", "材料里", "材料中",
        "原文写着", "原文写道",
    )
    _QUESTION_MARKERS = (
        "我想知道", "我想问", "是否", "是不是", "什么", "谁",
        "怎么", "如何", "为什么", "你呢", "你觉得", "你认为",
        "能否", "可否", "请介绍", "介绍一下", "解释一下",
        "说说", "请推荐", "推荐一下", "帮我分析",
    )
    _ENGLISH_QUESTION_PATTERN = re.compile(
        r"\b(?:what|who|how|which|when|where|why|can you|could you|do you|are you)\b",
        re.IGNORECASE,
    )

    _FORGET_ACTIONS = ("忘记", "删除", "清除", "移除", "不要再记")
    _NEGATED_FORGET_ACTIONS = (
        "不要忘记",
        "别忘记",
        "不要删除",
        "别删除",
        "不用删除",
        "无需删除",
        "不要清除",
        "别清除",
        "不要移除",
        "别移除",
    )
    _NON_MEMORY_DELETE_TARGETS = (
        "文件",
        "文档",
        "代码",
        "配置",
        "数据库",
        "测试数据",
        "申请记录",
    )
    _FORGET_MEMORY_TARGETS = (
        "记忆",
        "你记住",
        "记住的内容",
    )
    _FORGET_SLOT_TARGETS = (
        "我的名字",
        "我的姓名",
        "我的学校",
        "我的目标",
        "我的计划",
        "我的偏好",
        "我喜欢",
        "我不喜欢",
        "我的研究方向",
        "我的论文方向",
        "我的论文工作",
        "我的项目方向",
        "我的项目重点",
        "我的项目核心",
    )
    _RECALL_SLOT_PATTERNS = (
        (
            "profile.name",
            (
                "我叫什么",
                "我的名字是什么",
                "我的姓名是什么",
                "你记得我的名字吗",
            ),
        ),
        (
            "profile.school",
            (
                "我在哪所学校",
                "我就读于哪",
                "我的学校是什么",
                "你记得我的学校吗",
            ),
        ),
        (
            "career.goal",
            (
                "我的目标是什么",
                "我的求职目标是什么",
                "我的目标是啥",
                "我之前的目标是什么",
                "你记得我的目标吗",
            ),
        ),
        (
            "career.plan",
            (
                "我的计划是什么",
                "我接下来准备先做什么",
                "我打算做什么",
                "你记得我的计划吗",
            ),
        ),
        (
            "preference.dislike",
            (
                "我不喜欢什么",
                "我讨厌什么",
                "你记得我不喜欢什么吗",
            ),
        ),
        (
            "preference.like",
            (
                "我喜欢什么",
                "我喜欢做哪",
                "你记得我喜欢什么吗",
            ),
        ),
        (
            "research.topic",
            (
                "我的研究方向是什么",
                "我的论文方向是什么",
                "我的论文研究什么方向",
                "你记得我的研究方向吗",
            ),
        ),
        (
            "project.focus",
            (
                "我的项目核心是什么",
                "这个实习项目的核心是什么",
                "你记得我的项目核心吗",
            ),
        ),
    )
    _LIST_MEMORY_PATTERNS = (
        "我之前说过什么",
        "我刚才说过什么",
        "我上次说过什么",
        "之前我说了什么",
        "你还记得我吗",
        "你记得我说过",
        "你记住了什么",
        "我的记忆",
    )

    def extract(self, text: str) -> list[ExtractedMemory]:
        sentence = self._normalize(text)
        if not sentence:
            return []
        if self.has_non_user_context(sentence):
            return []

        facts = []
        seen_slots = set()
        seen_values = set()
        for clause in self._clauses(sentence):
            if self._is_question_clause(clause) or any(
                marker in clause for marker in self._NON_ASSERTIVE_MARKERS
            ):
                continue
            for memory_type, slot, pattern in self._RULES:
                match = pattern.search(clause)
                if not match or (slot in seen_slots and slot not in self.COLLECTION_SLOTS):
                    continue
                value = self._clean_value(match.group(1))
                if not value or (slot, value) in seen_values:
                    continue
                facts.append(ExtractedMemory(memory_type, slot, value))
                seen_slots.add(slot)
                seen_values.add((slot, value))
        return facts

    @classmethod
    def question_text(cls, text: str) -> str:
        """Select question clauses without treating earlier assertions as questions."""
        return "".join(
            clause for clause in cls._clauses(cls._normalize(text))
            if cls._is_question_clause(clause)
        ).strip()

    @staticmethod
    def _clauses(text: str) -> list[str]:
        # Keep punctuation so a question remains a question after splitting.
        return [part.strip() for part in re.split(r"(?<=[，,。！？!?；;])", text) if part.strip()]

    @classmethod
    def _is_question_clause(cls, clause: str) -> bool:
        return bool(
            any(mark in clause for mark in ("?", "？"))
            or any(marker in clause for marker in cls._QUESTION_MARKERS)
            or re.search(r"(?:吗|呢)[，,。！？!?；;]*$", clause)
            or cls._ENGLISH_QUESTION_PATTERN.search(clause)
        )

    def is_forget_request(self, text: str) -> bool:
        sentence = self._normalize(text)
        if not sentence or self.has_non_user_context(sentence) or self.question_text(sentence):
            return False
        if re.search(r"(?:不要|别|不用|无需|不必|请勿|不想|没让你|没有让你).{0,12}(?:忘记|删除|清除|移除)", sentence):
            return False
        if any(
            phrase in sentence for phrase in self._NEGATED_FORGET_ACTIONS
        ):
            return False
        if any(
            target in sentence for target in self._NON_MEMORY_DELETE_TARGETS
        ):
            return False
        has_action = any(word in sentence for word in self._FORGET_ACTIONS)
        has_target = any(
            target in sentence
            for target in self._FORGET_MEMORY_TARGETS + self._FORGET_SLOT_TARGETS
        )
        return has_action and has_target

    def forget_value(self, text: str) -> str | None:
        """An explicit preference value narrows deletion; None means whole slot.

        Only exact values are removed. Unmatched wording removes nothing;
        the memory page provides an unambiguous per-item delete operation.
        """
        match = re.search(r"我(?:不)?喜欢\s*([^，。！？,!?；;]*)", text)
        if not match:
            return None
        value = re.sub(r"(?:的)?(?:那条)?(?:偏好)?记忆$", "", match.group(1)).strip()
        return value or None

    def infer_forget_slots(self, text: str) -> list[str]:
        sentence = self._normalize(text)
        if not self.is_forget_request(sentence):
            return []
        if any(
            phrase in sentence
            for phrase in ("全部记忆", "所有记忆", "关于我的记忆", "你记住的内容")
        ):
            return ["*"]

        slots = []
        for keyword, candidates in self._FORGET_SLOT_KEYWORDS:
            if keyword == "喜欢" and "不喜欢" in sentence:
                continue
            if keyword in sentence:
                slots.extend(candidates)
        return list(dict.fromkeys(slots))

    def infer_recall_slots(self, text: str) -> list[str]:
        """Map explicit personal questions to structured memory slots."""

        sentence = self._normalize(text).lower()
        if not sentence:
            return []

        slots = []
        for slot, patterns in self._RECALL_SLOT_PATTERNS:
            if any(pattern in sentence for pattern in patterns):
                slots.append(slot)
        return slots

    def is_list_memory_request(self, text: str) -> bool:
        """Return whether the user explicitly asks for the whole memory list."""

        sentence = self._normalize(text).lower()
        return bool(sentence) and any(
            pattern in sentence for pattern in self._LIST_MEMORY_PATTERNS
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _clean_value(value: str) -> str:
        cleaned = value.strip(" \t\r\n：:，,。.!！?？")
        if not cleaned or len(cleaned) > 120:
            return ""
        return cleaned
