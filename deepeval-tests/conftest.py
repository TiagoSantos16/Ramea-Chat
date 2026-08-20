"""Shared fixtures for the Ramea LLM evaluation suite (deepeval).

The suite lives in a top-level ``deepeval-tests/`` directory so it is obviously
separate from the fast backend unit tests, and so that any visitor can inspect:

- what token/cost efficiency Ramea claims,
- how those claims are measured (offline + LLM-judged),
- exactly how to run the LLM-judged subset themselves.

Judge wiring
------------
Every LLM metric takes an explicit ``model=`` (an ``OpenRouterModel``) so we
judge with the exact credentials the product runs on. ``OPENROUTER_MODEL_NAME``
picks the judge model; the default here is a strong, **free** judge
(``openai/gpt-oss-20b:free`` on OpenRouter, $0, no credits). Override with a
paid flagship (e.g. ``anthropic/claude-3.5-sonnet``) whenever you want a
top-tier verdict.

Keyless friendliness
--------------------
All deepeval (LLM-judged) tests carry the ``llm`` marker and are automatically
skipped when ``OPENROUTER_API_KEY`` is absent. This keeps ``pytest`` / CI green
out-of-the-box while making the suite trivially runnable the moment a key is set:

    # PowerShell
    $env:OPENROUTER_API_KEY = "sk-or-..."
    $env:OPENROUTER_MODEL_NAME = "openai/gpt-oss-20b:free"   # judge (default, free)
    python -m pytest deepeval-tests -m llm -v

    # bash
    export OPENROUTER_API_KEY="sk-or-..."
    export OPENROUTER_MODEL_NAME="openai/gpt-oss-20b:free"
    python -m pytest deepeval-tests -m llm -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``efficiency`` importable from this folder without turning the folder
# into a package (which would shadow the installed ``deepeval`` library).
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from deepeval.models import OpenRouterModel  # noqa: E402

DEFAULT_JUDGE = "openai/gpt-oss-20b:free"


def openrouter_judge() -> OpenRouterModel:
    return OpenRouterModel(
        model=os.environ.get("OPENROUTER_MODEL_NAME", DEFAULT_JUDGE),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def has_api_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.fixture(scope="session")
def judge() -> OpenRouterModel:
    return openrouter_judge()


def pytest_collection_modifyitems(config, items):
    # Lax about the key: offline efficiency tests run regardless; any ``llm``
    # (LLM-judged) test must be skipped when there is no key so a default run
    # stays green.
    if has_api_key():
        return
    skip = pytest.mark.skip(
        reason="OPENROUTER_API_KEY not set; set it to run the LLM-judged deepeval tests"
    )
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)