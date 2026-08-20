# Ramea Chat, Evaluation and Efficiency Suite

> Just want the proof without running anything? [`results.md`](results.md) is
> the committed snapshot with the pass/fail table and headline numbers. This
> README explains how the suite works and how to re-run it.

The two files have distinct jobs:

- `README.md` is the instructions, what each test proves and how to re-run it.
- `results.md` is the evidence, the last recorded verdicts.

This is the quality and cost layer of the project, written so that anyone who
sees the repo can understand how Ramea makes good answers cheaper. It is
split into two halves on purpose:

| Half | What it proves | Needs a key? |
| --- | --- | --- |
| `test_efficiency.py` | Ramea sends only the active branch's ancestor path, so it costs a fraction of a linear chat. Deterministic, offline. | **No**, runs anywhere. |
| `test_llm_quality.py` | With that minimal context the model still retains the branch, ignores siblings, and follows the system prompt. Judged by a real model via deepeval. | **Yes**, needs `OPENROUTER_API_KEY`. |

The pair is meant to be read together. Efficiency shows why it is cheap,
quality shows that cheap is still correct.

## Why lineage and not history

In a classic linear chat, every new turn re-sends the entire conversation into
the prompt. As the transcript grows, each turn costs more, and most of those
tokens are old, off-topic content.

Ramea stores every reply as a node in a tree. When you ask something on a
branch, it streams only the ancestor path (root to active message) as context.
The exact path is `get_message_ancestor_path`, a recursive CTE shown in the
main README. Other branches, however many, cost zero tokens until you visit
them.

### Self-contained

This suite runs with only the files in this folder. The product source is
private, so the LLM tests hard-code a committed snapshot of the exact context
Ramea sends (the active branch's ancestor path) and reproduce the prompt
assembly inline. Every claim below is re-runnable in one command without
access to the application code.

## Reproducible numbers, run it yourself

```bash
pip install -r requirements.txt
python -m pytest deepeval-tests -v
```

Observed on a realistic 8-branch conversation (a deep security task plus 7
alternatives), measured with OpenAI's `cl100k_base` tokenizer:

```
[Efficiency] full-history context = 589 tokens; Ramea lineage = 240 tokens;
             59.3% fewer input tokens on this turn.
[Efficiency] all-turns: naive cumulative = 6713; lineage-sum = 2361.
```

- 59.3% fewer input tokens on the deepest active turn.
- About 3 times less total input over the whole session (2361 vs 6713 tokens).
- The saving grows with how much else you explore. Every extra branch you do
  not revisit is free forever.

## Running the LLM-judged quality tests

These call OpenRouter to generate answers through the same pipeline the app
uses, then judge them with a stronger model via deepeval metrics (Faithfulness,
Answer Relevancy, Contextual Relevancy, GEval). They need a key and are skipped
automatically without one.

### 1. Put your key where the app can see it

The tests read the same env var as the app's `/chat` fallback:

**PowerShell:**

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:OPENROUTER_MODEL_NAME = "openai/gpt-oss-20b:free"
```

**bash / zsh:**

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_MODEL_NAME="openai/gpt-oss-20b:free"
```

- `OPENROUTER_API_KEY` is your [OpenRouter](https://openrouter.ai) key.
- `OPENROUTER_MODEL_NAME` is the judge model. It defaults to
  `openai/gpt-oss-20b:free` (free, $0, so you can run the suite without
  spending credits). Set any paid flagship when you want a stronger verdict.
- `OPENROUTER_BASE_URL` optionally overrides the endpoint (defaults to the
  OpenRouter API).

(Or add the same `OPENROUTER_API_KEY` line to the repo's `.env`.)

### 2. Run just the LLM-judged tests

```bash
python -m pytest deepeval-tests -m llm -v
```

Each test generates a real answer and judges it, printing a per-test score and
reason. A passing run reports `6 passed`.

### What each LLM test proves

| Evaluated test | Judgement | Proves |
| --- | --- | --- |
| `test_deep_branch_context_is_retained` | Faithfulness | the model answers a follow-up from the deep branch without inventing facts |
| `test_answer_relevancy_to_question` | AnswerRelevancy | the answer is on-topic |
| `test_contextual_relevancy_of_provided_history` | ContextualRelevancy | the lineage actually contains what the question needs |
| `test_model_does_not_invent_absent_facts` | Faithfulness | a question about a sibling-only topic is not hallucinated |
| `test_sibling_facts_never_leak_into_answer` | GEval | none of the other branches' content leaks in |
| `test_system_prompt_is_followed` | GEval | the custom system prompt is obeyed (concise, no guessing) |

Together they answer the "is it worth saving tokens?" objection head-on: the
minimal context the app sends is still enough, and answers stay grounded.