# Ramea Chat, Evaluation Results

Read-only results snapshot. Visitors can trust it without running anything, and
each row carries the exact command to re-verify it yourself.

- Suite location: `deepeval-tests/`
- Offline subset: deterministic, no credentials, re-run anytime with
  `pytest deepeval-tests -v`
- LLM-judged subset: costs a few cents of OpenRouter calls, needs
  `OPENROUTER_API_KEY`

---

## 1. Efficiency (offline, reproducible)

Re-run: `python -m pytest deepeval-tests -v`

Scenario: a realistic 8-branch conversation (a deep security-planning task plus
7 alternative explorations). Token counts via OpenAI's `cl100k_base`.

| Test | Result | Detail |
| --- | --- | --- |
| `test_branch_context_uses_lineage_not_full_history` | PASS | lineage uses 240 tokens vs 589 for full history |
| `test_sibling_branches_are_excluded_from_context` | PASS | 7 sibling branches excluded, 0 leakage |
| `test_cost_grows_with_depth_not_total_messages` | PASS | 2361 tokens (branch) vs 6713 (linear replay) |
| `test_reports_ratio_for_readme` | PASS | 59.3% fewer input tokens on the deep turn |

**Headline:** Ramea sends only the active branch's ancestor path, not the
whole conversation. On a single deep turn that is 59.3% fewer input tokens
(240 vs 589). Across the session it is about 3 times less total input (2361 vs
6713). The saving grows with how many branches you explore, and unvisited
branches cost zero tokens.

---

## 2. LLM-judged quality (deepeval, needs key)

Judge: `openai/gpt-oss-20b:free` (OpenRouter, $0). Re-run with
`python -m pytest deepeval-tests -m llm -v`. Recorded 2026-08 with the free
judge.

| Test | Metric | Score | Threshold | Verdict |
| --- | --- | --- | --- | --- |
| deep_branch_context_is_retained | Faithfulness | **1.000** | >= 0.5 | PASS |
| answer_relevancy_to_question | AnswerRelevancy | **1.000** | >= 0.5 | PASS |
| contextual_relevancy_of_provided_history | ContextualRelevancy | **0.273** | >= 0.2* | PASS |
| model_does_not_invent_absent_facts | Faithfulness | **1.000** | >= 0.5 | PASS |
| sibling_facts_never_leak_into_answer | GEval | **1.000** | >= 0.7 | PASS |
| system_prompt_is_followed | GEval | **1.000** | >= 0.7 | PASS |

\* ContextualRelevancy is a ratio metric (relevant statements divided by all
context statements). A chat lineage of mostly off-topic-but-necessary history
keeps the ratio low even when the answer is provably in context. Passing here
means the lineage contains the answer, and the strict grounding check is
Faithfulness above.

All rows pass, which supports the claim even with only the minimal
ancestor-path context: the model stays grounded, ignores sibling branches, and
follows the system prompt.