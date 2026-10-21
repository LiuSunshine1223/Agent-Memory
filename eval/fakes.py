"""评测脚本自检用的替身模型。

这些替身只用来确认评测流程本身能跑通、打分逻辑正确，
不代表系统的真实表现。正式结果必须用真实嵌入模型和真实 LLM 运行。
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

_SKILL_PATTERN_CACHE: dict[str, re.Pattern] = {}


def skill_pattern(skill: str) -> re.Pattern:
    """匹配完整技能名，避免 Git 命中 GitHub 这类情况。"""
    pattern = _SKILL_PATTERN_CACHE.get(skill)
    if pattern is None:
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        _SKILL_PATTERN_CACHE[skill] = pattern
    return pattern


class HashEmbedder:
    """确定性的字符 n-gram 哈希向量，接口与 models.embedding.EmbeddingModel 相同。

    它没有语义能力：语义检索类指标在替身模式下没有意义，报告里会标出来。
    """

    def __init__(self, dimension: int = 512):
        self.dimension = int(dimension)

    def get_embeddings(self, texts):
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            normalized = "".join(str(text).lower().split())
            grams = [normalized[i:i + n] for n in (1, 2, 3) for i in range(len(normalized) - n + 1)]
            for gram in grams:
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class StubChatLLM:
    """对话用替身：立即返回固定回答，用于检查延迟评测流程。"""

    def ask(self, prompt, *, system=None):
        return "（替身模型回答）"

    def ask_json(self, prompt, *, schema=None, system=None):
        return '{"is_user_fact": false, "memories": []}'


class OracleJobFitLLM:
    """求职匹配用的“标准答案”替身。

    它从提示词里读出岗位要求和简历，按技能词表给出完全正确的结果。
    用途：验证打分脚本在完美输出下能得到 100%，在格式错误时能正确报错。
    """

    def __init__(self, skills, break_every: int = 0):
        self.skills = list(skills)
        self.break_every = int(break_every)
        self.calls = 0

    def ask(self, prompt):
        return self.ask_json(prompt)

    def ask_json(self, prompt):
        self.calls += 1
        if self.break_every and self.calls % self.break_every == 0:
            return "这不是 JSON"
        marker = "输入数据："
        payload = json.loads(prompt[prompt.rindex(marker) + len(marker):])
        jd_text = payload["job_description"]
        sources = payload["candidate_sources"]
        jd_skills = [s for s in self.skills if skill_pattern(s).search(jd_text)]

        matches, gaps, evidence = [], [], []
        for skill in jd_skills:
            quote = None
            source_id = None
            for source in sources:
                for line in source["content"].splitlines():
                    if skill_pattern(skill).search(line):
                        quote, source_id = line.strip(), source["source_id"]
                        break
                if quote:
                    break
            if quote:
                evidence_id = f"E{len(evidence) + 1}"
                evidence.append({"evidence_id": evidence_id, "source_id": source_id, "quote": quote})
                matches.append({"requirement": skill, "assessment": f"简历中提到 {skill}", "evidence_ids": [evidence_id]})
            else:
                gaps.append({"gap_id": f"G{len(gaps) + 1}", "requirement": skill, "reason": "简历中没有相关经历", "priority": "medium"})
        plan = [
            {"action": f"补充 {gap['requirement']} 项目经验", "gap_ids": [gap["gap_id"]], "priority": "medium", "success_criteria": "完成一个小项目"}
            for gap in gaps
        ]
        return json.dumps(
            {"matches": matches, "gaps": gaps, "preparation_plan": plan, "evidence": evidence},
            ensure_ascii=False,
        )
