"""用模板构造 DiscreteMemory 的评测用例。

标签在模板构造时确定；需抽查文本语义，不能代替真实数据标注。用法（在项目根目录运行）：

    python eval/generate_cases.py            # 生成到 eval/cases/
    python eval/generate_cases.py --seed 7   # 换一个随机种子

模板分成 dev 和 heldout 两部分；它们都是公开的合成诊断集。
开发与评测需分开，最终泛化结论还需要新的未查看数据。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CASES_VERSION = "v1"
EVAL_DIR = Path(__file__).resolve().parent

SLOTS = [
    "profile.name",
    "profile.school",
    "career.goal",
    "career.plan",
    "preference.like",
    "preference.dislike",
    "research.topic",
    "project.focus",
]

# ---------------------------------------------------------------------------
# 填充值：全部是虚构的示例，不含真实个人信息
# ---------------------------------------------------------------------------
FILLERS = {
    "zh": {
        "profile.name": ["李明", "王芳", "陈思远", "张伟", "刘洋", "赵雨桐"],
        "profile.school": ["浙江大学", "南京大学", "新加坡国立大学", "武汉大学", "中山大学", "南洋理工大学"],
        "career.goal": ["找一份RAG实习", "进入大模型公司做算法", "拿到数据分析岗位的offer", "成为后端工程师", "去新加坡做机器学习工程师", "找一份推荐系统实习"],
        "career.plan": ["下个月开始投递实习", "先把论文投出去", "每周刷十道算法题", "暑假前完成项目重构", "先学完Kubernetes", "整理一份英文简历"],
        "preference.like": ["远程办公", "小团队", "做底层系统", "写技术文档", "研究型岗位", "弹性工作时间"],
        "preference.dislike": ["频繁出差", "销售类岗位", "无效加班", "纯外包项目", "大量重复标注", "过长的通勤"],
        "research.topic": ["LLM记忆", "检索增强生成", "智能体记忆更新", "长上下文建模", "知识编辑", "模型可解释性"],
        "project.focus": ["离散长期记忆", "岗位匹配分析", "记忆写入治理", "向量检索优化", "文档生命周期管理", "评测框架"],
    },
    "en": {
        "profile.name": ["Wei Ling", "Arjun Kumar", "Sarah Tan", "Daniel Lim", "Nur Aisyah", "Kevin Ong"],
        "profile.school": ["NUS", "NTU", "SMU", "SUTD", "the University of Melbourne", "Imperial College London"],
        "career.goal": ["land a data analyst role", "join an AI startup as an ML engineer", "get a backend internship", "become a product manager", "work on LLM infrastructure", "find a research internship in NLP"],
        "career.plan": ["apply to ten companies next month", "finish my thesis first", "practice coding interviews every week", "learn Kubernetes before summer", "rewrite my resume in English", "build a portfolio project"],
        "preference.like": ["remote work", "small teams", "systems programming", "writing documentation", "research-oriented roles", "flexible hours"],
        "preference.dislike": ["frequent travel", "sales roles", "unpaid overtime", "outsourcing projects", "repetitive annotation work", "long commutes"],
        "research.topic": ["LLM memory", "retrieval-augmented generation", "agent memory updates", "long-context modeling", "knowledge editing", "model interpretability"],
        "project.focus": ["discrete long-term memory", "job-fit analysis", "memory write governance", "vector search optimization", "document lifecycle management", "evaluation tooling"],
    },
}

# ---------------------------------------------------------------------------
# 抽取模板：每个槽位 2 条 dev + 2 条 heldout
# 中文 dev 对应现有规则的说法，heldout 是规则没有覆盖的同义说法
# ---------------------------------------------------------------------------
EXTRACTION_TEMPLATES = {
    "zh": {
        "profile.name": {"dev": ["我叫{v}", "我的名字是{v}"], "heldout": ["大家都叫我{v}", "你可以叫我{v}"]},
        "profile.school": {"dev": ["我在{v}读书", "我就读于{v}"], "heldout": ["我目前是{v}的研究生", "我现在在{v}念书"]},
        "career.goal": {"dev": ["我的目标是{v}", "我的求职目标是{v}"], "heldout": ["我希望{v}", "我最想做的事是{v}"]},
        "career.plan": {"dev": ["我的计划是{v}", "我打算{v}"], "heldout": ["接下来我准备{v}", "我下一步要{v}"]},
        "preference.like": {"dev": ["我喜欢{v}", "我的偏好是{v}"], "heldout": ["我比较偏好{v}", "我更倾向于{v}"]},
        "preference.dislike": {"dev": ["我不喜欢{v}", "我讨厌{v}"], "heldout": ["我很讨厌{v}", "我没法接受{v}"]},
        "research.topic": {"dev": ["我的研究方向是{v}", "我的论文方向是{v}"], "heldout": ["我主要研究{v}", "我做的研究是关于{v}的"]},
        "project.focus": {"dev": ["我的项目重点是{v}", "我的项目核心是{v}"], "heldout": ["这个项目主要做{v}", "项目的重点在于{v}"]},
    },
    "en": {
        "profile.name": {"dev": ["My name is {v}", "I'm {v}"], "heldout": ["Call me {v}", "Everyone calls me {v}"]},
        "profile.school": {"dev": ["I study at {v}", "I'm a student at {v}"], "heldout": ["I'm currently doing my master's at {v}", "I go to {v}"]},
        "career.goal": {"dev": ["My goal is to {v}", "My career goal is to {v}"], "heldout": ["I want to {v}", "I'm hoping to {v}"]},
        "career.plan": {"dev": ["My plan is to {v}", "I plan to {v}"], "heldout": ["I'm going to {v}", "Next, I will {v}"]},
        "preference.like": {"dev": ["I like {v}", "I prefer {v}"], "heldout": ["I really enjoy {v}", "I'm a big fan of {v}"]},
        "preference.dislike": {"dev": ["I don't like {v}", "I hate {v}"], "heldout": ["I'm not a fan of {v}", "I can't stand {v}"]},
        "research.topic": {"dev": ["My research topic is {v}", "My research focuses on {v}"], "heldout": ["I'm researching {v}", "My thesis is about {v}"]},
        "project.focus": {"dev": ["My project focuses on {v}", "The core of my project is {v}"], "heldout": ["My project is mainly about {v}", "I'm building a project on {v}"]},
    },
}

SENTENCE_END = {"zh": ["", "。"], "en": ["", "."]}

COMPOUND_TEMPLATES = {
    "zh": ["我叫{name}，我的目标是{goal}。", "我在{school}读书，我的研究方向是{topic}。", "我的计划是{plan}，我不喜欢{dislike}。"],
    "en": ["My name is {name}, and my goal is to {goal}.", "I study at {school}, and my research focuses on {topic}.", "My plan is to {plan}, and I don't like {dislike}."],
}
COMPOUND_SLOTS = [
    {"name": "profile.name", "goal": "career.goal"},
    {"school": "profile.school", "topic": "research.topic"},
    {"plan": "career.plan", "dislike": "preference.dislike"},
]

# 负例：都不应该抽出任何事实（提问、假设、第三人称、否定、未决定、知识问题）
NEGATIVE_TEMPLATES = {
    "zh": [
        ("question", "我的目标是什么？"),
        ("question", "你知道我在哪所学校读书吗？"),
        ("question", "我的研究方向是不是{topic}？"),
        ("hypothetical", "如果我叫{name}，你会怎么称呼我"),
        ("hypothetical", "假如我的目标是{goal}，该怎么准备"),
        ("third_person", "我朋友叫{name}"),
        ("third_person", "我同学的目标是{goal}"),
        ("third_person", "我导师的研究方向是{topic}"),
        ("negation", "我不是在{school}读书"),
        ("negation", "我的目标不是{goal}"),
        ("undecided", "我还没决定要不要{plan}"),
        ("knowledge", "RAG和微调有什么区别"),
        ("knowledge", "介绍一下向量数据库"),
    ],
    "en": [
        ("question", "What is my goal?"),
        ("question", "Do you remember where I study?"),
        ("question", "Is my research topic {topic}?"),
        ("hypothetical", "If I wanted to {goal}, I would need more experience."),
        ("hypothetical", "Suppose my goal were to {goal}. How should I prepare?"),
        ("third_person", "My friend's name is {name}"),
        ("third_person", "My classmate's goal is to {goal}"),
        ("third_person", "My advisor's research focuses on {topic}"),
        ("negation", "I'm not studying at {school}"),
        ("negation", "My goal is not to {goal}"),
        ("undecided", "I haven't decided whether to {plan}"),
        ("knowledge", "What is the difference between RAG and fine-tuning?"),
        ("knowledge", "Explain how a vector database works."),
    ],
}
NEGATIVE_KEYS = {
    "name": "profile.name",
    "school": "profile.school",
    "goal": "career.goal",
    "plan": "career.plan",
    "topic": "research.topic",
    "dislike": "preference.dislike",
}

# ---------------------------------------------------------------------------
# 回忆问题路由：问题应该被识别为对应槽位
# ---------------------------------------------------------------------------
RECALL_QUESTIONS = {
    "zh": {
        "profile.name": ("我叫什么？", "你还记得怎么称呼我吗？"),
        "profile.school": ("我在哪所学校？", "我是哪个大学的？"),
        "career.goal": ("我的目标是什么？", "我当前的求职方向是什么？"),
        "career.plan": ("我的计划是什么？", "我之后准备干什么？"),
        "preference.like": ("我喜欢什么？", "我偏好什么样的工作？"),
        "preference.dislike": ("我不喜欢什么？", "我比较反感什么？"),
        "research.topic": ("我的研究方向是什么？", "我平时研究哪个领域？"),
        "project.focus": ("我的项目核心是什么？", "我的项目主要在做什么？"),
    },
    "en": {
        "profile.name": ("What's my name?", "Do you remember what to call me?"),
        "profile.school": ("Which school do I go to?", "Where am I studying?"),
        "career.goal": ("What is my goal?", "What kind of job am I aiming for?"),
        "career.plan": ("What is my plan?", "What am I going to do next?"),
        "preference.like": ("What do I like?", "What kind of work do I prefer?"),
        "preference.dislike": ("What don't I like?", "What do I want to avoid?"),
        "research.topic": ("What is my research topic?", "What field do I research?"),
        "project.focus": ("What is my project about?", "What does my project focus on?"),
    },
}

# ---------------------------------------------------------------------------
# 同意判定：包装一句可抽取的事实，标签是期望的 (action, reason_code)
# ---------------------------------------------------------------------------
CONSENT_WRAPPERS = {
    "zh": [
        ("explicit", "dev", "请记住，{fact}", "commit", "explicit_user_consent"),
        ("explicit", "dev", "帮我记住：{fact}", "commit", "explicit_user_consent"),
        ("explicit", "dev", "{fact}，请保存到长期记忆", "commit", "explicit_user_consent"),
        ("explicit", "dev", "记住，{fact}", "commit", "explicit_user_consent"),
        ("plain", "dev", "{fact}", "propose", "confirmation_required"),
        ("plain", "dev", "顺便说一下，{fact}", "propose", "confirmation_required"),
        ("opt_out", "dev", "{fact}，不要记住这个", "ignore", "explicit_opt_out"),
        ("opt_out", "dev", "别保存：{fact}", "ignore", "explicit_opt_out"),
        ("opt_out", "dev", "{fact}，这个不用记住", "ignore", "explicit_opt_out"),
        ("opt_out", "dev", "我没让你记住，{fact}", "ignore", "explicit_opt_out"),
        ("explicit", "heldout", "别忘了，{fact}", "commit", "explicit_user_consent"),
        ("opt_out", "heldout", "{fact}，这条就当我没说", "ignore", "explicit_opt_out"),
        ("opt_out", "heldout", "{fact}，只是随口一提，不必存", "ignore", "explicit_opt_out"),
    ],
    "en": [
        ("explicit", "dev", "Please remember that {fact_lc}", "commit", "explicit_user_consent"),
        ("explicit", "dev", "Remember this: {fact}", "commit", "explicit_user_consent"),
        ("explicit", "dev", "Save this to memory: {fact}", "commit", "explicit_user_consent"),
        ("explicit", "dev", "{fact}. Please save it.", "commit", "explicit_user_consent"),
        ("plain", "dev", "{fact}", "propose", "confirmation_required"),
        ("plain", "dev", "By the way, {fact_lc}", "propose", "confirmation_required"),
        ("opt_out", "dev", "{fact}, but don't remember it.", "ignore", "explicit_opt_out"),
        ("opt_out", "dev", "Do not save this: {fact}", "ignore", "explicit_opt_out"),
        ("opt_out", "dev", "{fact}. You don't need to store this.", "ignore", "explicit_opt_out"),
        ("opt_out", "dev", "Don't remember this, but {fact_lc}", "ignore", "explicit_opt_out"),
        ("explicit", "heldout", "Don't forget that {fact_lc}", "commit", "explicit_user_consent"),
        ("opt_out", "heldout", "{fact}. Just mentioning it, no need to note it down.", "ignore", "explicit_opt_out"),
        ("opt_out", "heldout", "{fact}. Forget I said that.", "ignore", "explicit_opt_out"),
    ],
}

# ---------------------------------------------------------------------------
# 更新与删除场景
# ---------------------------------------------------------------------------
SLOT_QUERY_ZH = {
    "profile.name": "我叫什么",
    "profile.school": "我的学校是什么",
    "career.goal": "我的目标是什么",
    "career.plan": "我的计划是什么",
    "preference.like": "我喜欢什么",
    "preference.dislike": "我不喜欢什么",
    "research.topic": "我的研究方向是什么",
    "project.focus": "我的项目核心是什么",
}
SEMANTIC_QUERIES = {
    "zh": {
        "profile.name": "怎么称呼我",
        "profile.school": "我是哪个学校的学生",
        "career.goal": "我想找什么样的工作",
        "career.plan": "我接下来有什么安排",
        "preference.like": "我有什么偏好",
        "preference.dislike": "我反感什么",
        "research.topic": "我研究什么课题",
        "project.focus": "我的项目主要做什么",
    },
    "en": {
        "profile.name": "What should you call me?",
        "profile.school": "Which university do I attend?",
        "career.goal": "What kind of job am I looking for?",
        "career.plan": "What am I planning to do next?",
        "preference.like": "What do I prefer?",
        "preference.dislike": "What do I want to avoid?",
        "research.topic": "What do I research?",
        "project.focus": "What is my project about?",
    },
}
LIFECYCLE_TYPES_ALL_SLOTS = ["update", "delete", "purge"]
LIFECYCLE_TYPES_SAMPLED = ["update_delete", "restart", "index_loss", "candidate_confirm", "candidate_reject"]

# ---------------------------------------------------------------------------
# 求职匹配：虚构简历和岗位描述，匹配项和缺口在构造时已知
# 技能名选用彼此不包含的词，便于自动判断
# ---------------------------------------------------------------------------
SKILLS = [
    "Python", "MySQL", "Docker", "Kubernetes", "PyTorch", "FastAPI", "Spark", "Tableau",
    "Redis", "Kafka", "TensorFlow", "Airflow", "PostgreSQL", "React", "Linux", "Git",
]
JOBFIT_TEXT = {
    "zh": {
        "names": ["林晓", "周然", "许诺", "沈一舟"],
        "schools": ["某理工大学", "某师范大学", "某财经大学", "某工业大学"],
        "titles": ["机器学习工程师实习生", "数据分析实习生", "后端开发实习生", "AI 应用工程师实习生"],
        "tasks": ["一个数据清洗流程", "一个推荐服务的后端接口", "实验结果的可视化看板", "模型训练与评估脚本", "日志采集与分析流程", "一个内部知识库检索服务"],
        "requirements": ["熟练使用{s}", "有{s}项目经验", "熟悉{s}", "了解{s}的基本原理"],
        "project_line": "- 使用{s}完成{task}。",
        "resume": "姓名：{name}\n教育背景：{school}，计算机科学硕士\n项目经历：\n{projects}\n技能：{skills}",
        "skill_sep": "、",
        "jd": "岗位名称：{title}\n工作职责：参与团队相关系统的开发与维护。\n任职要求：\n{requirements}",
    },
    "en": {
        "names": ["Alex Chen", "Priya Nair", "Marcus Lee", "Siti Rahman"],
        "schools": ["a technical university", "a state university", "a business school", "an engineering college"],
        "titles": ["Machine Learning Engineer Intern", "Data Analyst Intern", "Backend Engineer Intern", "AI Application Engineer Intern"],
        "tasks": ["a data cleaning pipeline", "a backend API for a recommendation service", "a dashboard for experiment results", "model training and evaluation scripts", "a log collection and analysis pipeline", "an internal knowledge-base search service"],
        "requirements": ["Proficient in {s}", "Hands-on experience with {s}", "Familiar with {s}", "Working knowledge of {s}"],
        "project_line": "- Built {task} using {s}.",
        "resume": "Name: {name}\nEducation: MSc in Computer Science, {school}\nProjects:\n{projects}\nSkills: {skills}",
        "skill_sep": ", ",
        "jd": "Position: {title}\nResponsibilities: Develop and maintain the team's internal systems.\nRequirements:\n{requirements}",
    },
}

LATENCY_MESSAGES = [
    ("zh", "什么是RAG？"),
    ("zh", "怎么准备机器学习岗位的面试？"),
    ("zh", "请记住，我的目标是找一份RAG实习"),
    ("zh", "我的目标是什么"),
    ("zh", "新加坡的科技公司一般看重什么能力？"),
    ("zh", "我之前说过什么"),
    ("en", "What is RAG?"),
    ("en", "How should I prepare for a machine learning interview?"),
    ("en", "Please remember that my goal is to land a data analyst role"),
    ("en", "What is my goal?"),
    ("en", "What skills do tech companies in Singapore usually look for?"),
    ("en", "What have I told you before?"),
]


def _fill(template: str, value: str) -> str:
    return template.replace("{v}", value)


def _lower_first(text: str) -> str:
    """英文句子放到句中时首字母小写，但代词 I / I'm 保持大写。"""
    if text.startswith(("I ", "I'")):
        return text
    return text[:1].lower() + text[1:]


def build_extraction(rng: random.Random) -> list[dict]:
    cases = []
    counter = 0

    def add(lang, kind, split, text, expected, template, category=None):
        nonlocal counter
        counter += 1
        cases.append({
            "id": f"ext-{counter:04d}",
            "lang": lang,
            "kind": kind,
            "split": split,
            "category": category or kind,
            "text": text,
            "expected": expected,
            "template": template,
        })

    # 单句、单个事实
    for lang in ("zh", "en"):
        for slot in SLOTS:
            for split in ("dev", "heldout"):
                for template in EXTRACTION_TEMPLATES[lang][slot][split]:
                    for value in rng.sample(FILLERS[lang][slot], 3):
                        end = rng.choice(SENTENCE_END[lang])
                        add(lang, "single", split, _fill(template, value) + end,
                            [{"slot": slot, "value": value}], template)

    # 中英混合：中文句式配英文值，英文句式配中文值（只用 dev 句式）
    for slot in SLOTS:
        for template in EXTRACTION_TEMPLATES["zh"][slot]["dev"][:1]:
            for value in rng.sample(FILLERS["en"][slot], 2):
                add("mix", "single", "dev", _fill(template, value),
                    [{"slot": slot, "value": value}], template)
        for template in EXTRACTION_TEMPLATES["en"][slot]["dev"][:1]:
            for value in rng.sample(FILLERS["zh"][slot], 2):
                add("mix", "single", "dev", _fill(template, value),
                    [{"slot": slot, "value": value}], template)

    # 一句话里有两个事实
    for lang in ("zh", "en"):
        for template, slot_map in zip(COMPOUND_TEMPLATES[lang], COMPOUND_SLOTS):
            for _ in range(3):
                values = {key: rng.choice(FILLERS[lang][slot]) for key, slot in slot_map.items()}
                text = template.format(**values)
                expected = [{"slot": slot_map[key], "value": values[key]} for key in slot_map]
                add(lang, "compound", "dev", text, expected, template)

    # 负例
    for lang in ("zh", "en"):
        for category, template in NEGATIVE_TEMPLATES[lang]:
            repeats = 3 if "{" in template else 1
            seen = set()
            for _ in range(repeats):
                values = {key: rng.choice(FILLERS[lang][slot]) for key, slot in NEGATIVE_KEYS.items()}
                text = template.format(**values)
                if text in seen:
                    continue
                seen.add(text)
                add(lang, "negative", "dev", text, [], template, category=category)
    return cases


def build_recall_routing() -> list[dict]:
    cases = []
    for lang in ("zh", "en"):
        for slot in SLOTS:
            dev_q, heldout_q = RECALL_QUESTIONS[lang][slot]
            for split, question in (("dev", dev_q), ("heldout", heldout_q)):
                cases.append({
                    "id": f"route-{len(cases) + 1:04d}",
                    "lang": lang,
                    "split": split,
                    "text": question,
                    "expected_slots": [slot],
                })
    return cases


def build_consent(rng: random.Random) -> list[dict]:
    cases = []
    for lang in ("zh", "en"):
        facts = []
        for slot in rng.sample(SLOTS, 6):
            template = EXTRACTION_TEMPLATES[lang][slot]["dev"][0]
            value = rng.choice(FILLERS[lang][slot])
            facts.append((slot, value, _fill(template, value)))
        for kind, split, wrapper, action, reason in CONSENT_WRAPPERS[lang]:
            for slot, value, fact in facts:
                fact_lc = _lower_first(fact) if lang == "en" else fact
                text = wrapper.format(fact=fact, fact_lc=fact_lc)
                cases.append({
                    "id": f"consent-{len(cases) + 1:04d}",
                    "lang": lang,
                    "kind": kind,
                    "split": split,
                    "wrapper": wrapper,
                    "text": text,
                    "fact": {"slot": slot, "value": value},
                    "expected_action": action,
                    "expected_reason": reason,
                })
    return cases


def build_lifecycle(rng: random.Random) -> list[dict]:
    scenarios = []
    for lang in ("zh", "en"):
        plan = [(kind, SLOTS) for kind in LIFECYCLE_TYPES_ALL_SLOTS]
        plan += [(kind, rng.sample(SLOTS, 4)) for kind in LIFECYCLE_TYPES_SAMPLED]
        for kind, slots in plan:
            for slot in slots:
                old, new = rng.sample(FILLERS[lang][slot], 2)
                others = [item for item in SLOTS if item != slot]
                distractors = [
                    {"slot": other, "value": rng.choice(FILLERS[lang][other])}
                    for other in rng.sample(others, 3)
                ]
                template = EXTRACTION_TEMPLATES[lang][slot]["dev"][0]
                # 更新有两条代码路径：记忆管理接口的编辑 (edit)，
                # 以及对话写入/候选确认走的 remember。两条都要测。
                update_path = "edit" if len(scenarios) % 2 == 0 else "remember"
                scenarios.append({
                    "id": f"life-{len(scenarios) + 1:04d}",
                    "lang": lang,
                    "type": kind,
                    "slot": slot,
                    "old": old,
                    "new": new,
                    "update_path": update_path,
                    "distractors": distractors,
                    "slot_query": SLOT_QUERY_ZH[slot],
                    "semantic_query": SEMANTIC_QUERIES[lang][slot],
                    # 候选流程用：一句可被抽取的用户原话
                    "propose_text": _fill(template, new),
                })
    return scenarios


def build_jobfit(rng: random.Random, per_lang: int = 20) -> list[dict]:
    cases = []
    for lang in ("zh", "en"):
        text = JOBFIT_TEXT[lang]
        for _ in range(per_lang):
            jd_skills = rng.sample(SKILLS, 5)
            matched = rng.sample(jd_skills, rng.randint(2, 4))
            extras = rng.sample([s for s in SKILLS if s not in jd_skills], rng.randint(1, 2))
            resume_skills = matched + extras
            rng.shuffle(resume_skills)
            tasks = rng.sample(text["tasks"], len(resume_skills))
            projects = "\n".join(
                text["project_line"].format(s=skill, task=task)
                for skill, task in zip(resume_skills, tasks)
            )
            resume = text["resume"].format(
                name=rng.choice(text["names"]),
                school=rng.choice(text["schools"]),
                projects=projects,
                skills=text["skill_sep"].join(resume_skills),
            )
            requirements = "\n".join(
                f"{index}. " + rng.choice(text["requirements"]).format(s=skill)
                for index, skill in enumerate(jd_skills, start=1)
            )
            jd = text["jd"].format(title=rng.choice(text["titles"]), requirements=requirements)
            cases.append({
                "id": f"jobfit-{len(cases) + 1:04d}",
                "lang": lang,
                "jd_text": jd,
                "resume": {"source_id": "resume", "title": "Resume" if lang == "en" else "简历", "kind": "resume", "content": resume},
                "jd_skills": jd_skills,
                "expected_matches": sorted(matched),
                "expected_gaps": sorted(set(jd_skills) - set(matched)),
            })
    return cases


def build_latency() -> list[dict]:
    return [
        {"id": f"lat-{index:02d}", "lang": lang, "text": text}
        for index, (lang, text) in enumerate(LATENCY_MESSAGES, start=1)
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 DiscreteMemory 评测用例")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "cases")
    parser.add_argument("--jobfit-per-lang", type=int, default=20)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    datasets = {
        "extraction": build_extraction(rng),
        "recall_routing": build_recall_routing(),
        "consent": build_consent(rng),
        "lifecycle": build_lifecycle(rng),
        "jobfit": build_jobfit(rng, args.jobfit_per_lang),
        "latency_messages": build_latency(),
    }
    for name, rows in datasets.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)
    manifest = {
        "cases_version": CASES_VERSION,
        "seed": args.seed,
        "counts": {name: len(rows) for name, rows in datasets.items()},
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, count in manifest["counts"].items():
        print(f"{name:18s} {count:4d} 条")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
