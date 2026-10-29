"""应用层长期记忆写入的确定性授权策略。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from memory.extractor import DiscreteMemoryExtractor


@dataclass(frozen=True)
class WritePolicyDecision:
    """说明提取出的事实是否可以进入有效长期记忆。"""

    action: str
    reason_code: str
    policy_version: str
    explicit_consent: bool
    candidate_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class ConsentAwareWritePolicy:
    """仅对明确的保存请求直接写入；其他事实先作为候选等待确认。

    提取器判断哪些内容像可长期保存的事实；本策略独立判断是否允许保存：
    明确拒绝保存时忽略写入，明确要求保存时直接提交，普通陈述生成候选。
    """

    VERSION = "consent-gated-v3"
    _SAVE_ACTIONS = ("记住", "记下来", "保存", "写入", "加入")

    _OPT_OUT_MARKERS = (
        "不保存",
        "不存",
        "不记住",
        "不记下来",
        "不用记",
        "就当我没说",
        "不必存",
        "不用存",
        "不要存",
        "无需存",
        "别存",
        "不要记住",
        "别记住",
        "不用记住",
        "无需记住",
        "不要保存",
        "别保存",
        "不用保存",
        "无需保存",
        "do not remember",
        "don't remember",
        "do not save",
        "don't save",
    )
    _NON_CONSENT_MARKERS = (
        "没让你",
        "没有让你",
        "未让你",
        "没叫你",
        "没有叫你",
        "未叫你",
        "没要求你",
        "没有要求你",
        "未要求你",
        "没同意",
        "没有同意",
        "未同意",
        "不允许",
        "不准",
        "不许",
    )
    _EXPLICIT_SAVE_MARKERS = (
        "请记住",
        "帮我记住",
        "请帮我记住",
        "麻烦记住",
        "务必记住",
        "记下来",
        "请保存",
        "帮我保存",
        "保存到记忆",
        "保存为记忆",
        "保存到长期记忆",
        "保存为长期记忆",
        "写入记忆",
        "写入长期记忆",
        "加入记忆",
        "加入长期记忆",
        "please remember",
        "remember that",
        "save to memory",
    )
    _EXPLICIT_SAVE_PREFIXES = (
        "记住，",
        "记住,",
        "记住：",
        "记住:",
        "记住 ",
        "记住我的",
        "记住我",
    )
    _NEGATED_SAVE_PATTERN = re.compile(
        r"(?:不要|别|不用|无需|不必|请勿|不想|不需要).{0,12}(?:" + "|".join(_SAVE_ACTIONS) + r")"
        r"|(?:do not|don't|never).{0,20}(?:remember|save)",
        re.IGNORECASE,
    )

    def decide(self, text: str, *, candidate_count: int) -> WritePolicyDecision:
        if isinstance(candidate_count, bool) or candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative integer")
        normalized = " ".join(str(text or "").strip().lower().split())
        if candidate_count == 0 or DiscreteMemoryExtractor.has_non_user_context(normalized):
            return WritePolicyDecision(
                action="ignore",
                reason_code="no_extractable_fact",
                policy_version=self.VERSION,
                explicit_consent=False,
                candidate_count=0,
            )
        has_memory_verb = any(verb in normalized for verb in self._SAVE_ACTIONS)
        if (
            any(marker in normalized for marker in self._OPT_OUT_MARKERS)
            or self._NEGATED_SAVE_PATTERN.search(normalized)
            or (
                has_memory_verb
                and any(
                    marker in normalized
                    for marker in self._NON_CONSENT_MARKERS
                )
            )
        ):
            return WritePolicyDecision(
                action="ignore",
                reason_code="explicit_opt_out",
                policy_version=self.VERSION,
                explicit_consent=False,
                candidate_count=candidate_count,
            )
        save_question = any(
            any(verb in clause for verb in self._SAVE_ACTIONS)
            and DiscreteMemoryExtractor.question_text(clause)
            for clause in DiscreteMemoryExtractor._clauses(normalized)
        )
        if not save_question and (any(marker in normalized for marker in self._EXPLICIT_SAVE_MARKERS) or (
            normalized.startswith(self._EXPLICIT_SAVE_PREFIXES)
        )):
            return WritePolicyDecision(
                action="commit",
                reason_code="explicit_user_consent",
                policy_version=self.VERSION,
                explicit_consent=True,
                candidate_count=candidate_count,
            )
        return WritePolicyDecision(
            action="propose",
            reason_code="confirmation_required",
            policy_version=self.VERSION,
            explicit_consent=False,
            candidate_count=candidate_count,
        )


__all__ = ["ConsentAwareWritePolicy", "WritePolicyDecision"]
