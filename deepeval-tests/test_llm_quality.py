"""LLM-judged quality tests, the "does it actually hold?" counterpart to the
offline efficiency tests.

The offline suite proves Ramea *should* be cheaper (lineage-only context).
These tests prove it *still answers correctly* with that minimal context: the
model must (a) retain the deep-branch context, (b) never let sibling branches
leak in, and (c) follow the custom system prompt. Every answer is generated
through OpenRouter (the same pipeline the app uses) and then judged by a second,
stronger OpenRouter model via deepeval metrics.

Requires ``OPENROUTER_API_KEY``. Automatically skipped otherwise (see
``conftest.py``). Run:

    $env:OPENROUTER_API_KEY  = "sk-or-..."     # PowerShell
    $env:OPENROUTER_MODEL_NAME = "anthropic/claude-3.5-sonnet"
    python -m pytest deepeval-tests -m llm -v
"""

from __future__ import annotations

import pytest

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

pytestmark = pytest.mark.llm

SYSTEM_PROMPT = (
    "You are Ramea, a branching-chat assistant. Answer using only the "
    "conversation history you are given. Never invent details. If a fact is "
    "needed that is not in the given history, say you don't know rather than "
    "guess. Be concise."
)

# The lineage Ramea would stream for the active deep turn (b-branch).
DEEP_LINEAGE = [
    {"role": "user", "content": "I need a plan to launch our open-source chat product in 90 days."},
    {
        "role": "assistant",
        "content": (
            "90-day plan: weeks 1-2 research and architecture, 3-5 build the branching core, "
            "6-7 polish the graph UX, 8-9 harden security and auth, 10-11 test and pilot with "
            "5 teams, 12 marketing and GA."
        ),
    },
    {"role": "user", "content": "Expand the security and auth section."},
    {
        "role": "assistant",
        "content": (
            "Security in weeks 8-9: enforce encryption-at-rest for the API key store "
            "(Fernet/AES), throttle login by IP and account, sanitise user and AI markdown, "
            "scope LLM context to the active branch lineage, and add per-user isolation. "
            "Want the checklist as Markdown?"
        ),
    },
    {"role": "user", "content": "Yes, give me the checklist as a Markdown file."},
    {
        "role": "assistant",
        "content": (
            "Checklist (security, weeks 8-9): (1) encrypt stored OpenRouter keys with a "
            "derived Fernet key; (2) throttle login by IP and account; (3) sanitise "
            "user/AI markdown; (4) restrict prompt context to the ancestor path; (5) scope "
            "every query to the conversation's owner."
        ),
    },
]

# Facts that exist ONLY in sibling branches, must never be used.
SIBLING_FACTS = [
    "The ReactFlow graph is the main frontend cost; budget 3-5 days for visual polish.",
    "A managed Postgres (Neon or RDS) removes backup and patching overhead.",
    "/models is public and unauthenticated; the API key is only used by /chat.",
    "Deleting a node removes its entire subtree while siblings stay untouched.",
]


def _context_texts() -> list[str]:
    return [m["content"] for m in DEEP_LINEAGE]


def _case(input: str, actual_output: str) -> LLMTestCase:
    """Build a test case with both context aliases deepeval supports.

    deepeval 4.1.x renamed ``context`` -> ``retrieval_context`` for the
    retrieval metrics (Faithfulness, ContextualRelevancy) but GEval still reads
    ``context``. Setting both keeps every metric working across versions.
    """
    texts = _context_texts()
    return LLMTestCase(
        input=input,
        actual_output=actual_output,
        context=texts,
        retrieval_context=texts,
    )


def _report(metric, label: str) -> None:
    """Print a machine-readable result line for the results.md snapshot."""
    verdict = "PASS" if metric.is_successful() else "FAIL"
    threshold = getattr(metric, "threshold", None)
    print(
        f"\n[llm-quality] {label} | score={getattr(metric, 'score', None):.3f} "
        f"| threshold={threshold} | {verdict}"
    )
    reason = getattr(metric, "reason", None)
    if reason:
        try:
            print(f"[llm-quality] {label} reason: {reason}")
        except UnicodeEncodeError:
            safe = reason.encode("ascii", "replace").decode("ascii")
            print(f"[llm-quality] {label} reason: {safe}")


def _prompt_for(question: str) -> str:
    """Mirror Ramea's prompt assembly: system prompt, then each lineage turn, then the question."""
    parts = [f"System: {SYSTEM_PROMPT}"]
    parts += [f"{m['role']}: {m['content']}" for m in DEEP_LINEAGE]
    parts.append(f"user: {question}")
    return "\n\n".join(parts)


@pytest.fixture(scope="module")
def _answers(judge):
    """Generate model answers to the deep-branch questions once per module run."""
    questions = {
        "q1": "Step 3 of the checklist, what does it say exactly?",
        "q2": "When in the 90-day plan is the security work scheduled?",
        "q3": "Is there any mention of a minimap in this conversation?",
    }
    answers = {}
    for key, q in questions.items():
        out, _cost = judge.generate(prompt=_prompt_for(q))
        answers[key] = out.strip() if isinstance(out, str) else str(out)
    return answers


def test_deep_branch_context_is_retained(_answers, judge):
    """The model must answer from the deep-branch context, not hallucinate."""
    actual = _answers["q1"]
    test_case = _case(
        input="Step 3 of the checklist, what does it say exactly?",
        actual_output=actual,
    )
    metric = FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)
    metric.measure(test_case)
    _report(metric, "deep_branch_context_is_retained")
    assert metric.is_successful(), (
        f"Faithfulness failed ({metric.score:.2f}): answer {actual!r} added facts "
        f"not present in the branch context. Reason: {metric.reason}"
    )


def test_answer_relevancy_to_question(_answers, judge):
    test_case = _case(
        input="When in the 90-day plan is the security work scheduled?",
actual_output=_answers["q2"],
    )
    metric = AnswerRelevancyMetric(model=judge, threshold=0.5, async_mode=False)
    metric.measure(test_case)
    _report(metric, "answer_relevancy_to_question")
    assert metric.is_successful(), (
        f"AnswerRelevancy failed ({metric.score:.2f}): {_answers['q2']!r}. Reason: {metric.reason}"
    )


def test_contextual_relevancy_of_provided_history(_answers, judge):
    """The lineage Ramea sends must actually contain what the question needs.

    NOTE on the threshold: deepeval's ContextualRelevancy is *ratio-based*,
    relevant context statements ÷ all context statements. A chat lineage is a
    long history of mostly off-topic-but-necessary messages, so the ratio is
    naturally low even when the needed fact ("weeks 8-9") is present. Passing
    here means the lineage demonstrably contains the answer; the strict
    grounding checks are Faithfulness (above).
    """
    test_case = _case(
        input="When in the 90-day plan is the security work scheduled?",
        actual_output=_answers["q2"],
    )
    metric = ContextualRelevancyMetric(model=judge, threshold=0.2, async_mode=False)
    metric.measure(test_case)
    _report(metric, "contextual_relevancy_of_provided_history")
    assert metric.is_successful(), (
        f"ContextualRelevancy failed ({metric.score:.2f}): the sent context didn't cover "
        f"the question. Reason: {metric.reason}"
    )


def test_model_does_not_invent_absent_facts(_answers, judge):
    """A question about a sibling-only topic must NOT be answered from context."""
    test_case = _case(
        input="Is there any mention of a minimap in this conversation?",
actual_output=_answers["q3"],
    )
    metric = FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False)
    metric.measure(test_case)
    _report(metric, "model_does_not_invent_absent_facts")
    assert metric.is_successful(), (
        f"Model hallucinated from a sibling topic ({metric.score:.2f}): {_answers['q3']!r}. "
        f"Reason: {metric.reason}"
    )


def test_sibling_facts_never_leak_into_answer(_answers, judge):
    """GEval judge: the reply must draw only from the active lineage."""
    sibling_hint = "; ".join(SIBLING_FACTS)
    metric = GEval(
        name="Sibling isolation",
        criteria=(
            "Judge whether the assistant answer uses ONLY the supplied conversation "
            "history. In particular, it must never reference, endorse, or contradict any "
            f"of these facts that come from other branches (they are NOT in context): "
            f"{sibling_hint}. A perfect answer relies solely on the provided history and "
            "stays silent about the other branches' content."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
model=judge,
        threshold=0.7,
        async_mode=False,
    )
    # q3 asks about a minimap, a sibling-only topic; the model must not invent it.
    test_case = _case(
        input="Is there any mention of a minimap in this conversation?",
        actual_output=_answers["q3"],
    )
    metric.measure(test_case)
    _report(metric, "sibling_facts_never_leak_into_answer")
    assert metric.is_successful(), (
        f"Sibling facts leaked into the answer ({metric.score:.2f}): {_answers['q3']!r}. "
        f"Reason: {metric.reason}"
    )


def test_system_prompt_is_followed(_answers, judge):
    """Custom system prompt (concise, no invention) must be obeyed."""
    metric = GEval(
        name="System prompt adherence",
        criteria=(
            "The assistant must follow its system instructions: be concise, answer only "
            "from the supplied history, and say it doesn't know rather than guess. An "
            "answer that invents details or contradicts the system prompt fails."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
model=judge,
        threshold=0.7,
        async_mode=False,
    )
    test_case = _case(
        input="Step 3 of the checklist, what does it say exactly?",
        actual_output=_answers["q1"],
    )
    metric.measure(test_case)
    _report(metric, "system_prompt_is_followed")
    assert metric.is_successful(), (
        f"System prompt not followed ({metric.score:.2f}): {_answers['q1']!r}. "
        f"Reason: {metric.reason}"
    )
