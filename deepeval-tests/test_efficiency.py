"""Offline efficiency tests: deterministic, no LLM, no API key.

These prove the project's core cost thesis numerically:

    Ramea sends only the *active branch's ancestor path*, so input-token
    cost scales with conversation *depth*, not with the total number of messages
    a user has produced across all branches. Unvisited branches never cost a
    token again.

No network, no judge, just honest token counts (``cl100k_base``). They run in
plain ``pytest deepeval-tests`` with no credentials, so the efficiency story is
verifiable by anyone who clones the repo.

Marker: ``efficiency`` (these are safe/fast; no ``llm`` marker, no key needed).
"""

from __future__ import annotations

import pytest

from efficiency import BranchTree, Msg, tokens_for

pytestmark = pytest.mark.efficiency



def build_conversation() -> BranchTree:
    """A realistic branching conversation.

    One root user turn, then a deep main thread plus three alternative branches,
    the exact pattern Ramea is designed for (compare alternatives, go deep
    on one, keep the rest warm).
    """
    root = Msg("I need a plan to launch our open-source chat product in 90 days.")
    a1 = Msg(
        "Here's a 90-day plan: weeks 1-2 research and architecture, 3-5 build the branching "
        "core, 6-7 polish the graph UX, 8-9 harden security and auth, 10-11 test and pilot "
        "with 5 teams, 12 marketing and GA. Which section should I expand?"
    )
    b1 = Msg("Expand the security and auth section.")
    b2 = Msg(
        "Security in weeks 8-9: enforce encryption-at-rest for the API key store (Fernet/AES), "
        "throttle login by IP and account, sanitise user and AI markdown, scope LLM context "
        "strictly to the active branch lineage, and add per-user isolation. Want the full "
        "checklist as a Markdown file?"
    )
    b3 = Msg("Yes, give me the checklist as a Markdown file.")
    b4 = Msg(
        "Checklist (security, weeks 8-9): (1) encrypt stored OpenRouter keys with a derived "
        "Fernet key; (2) throttle login by IP and account; (3) sanitise user/AI markdown; "
        "(4) restrict prompt context to the ancestor path; (5) scope every query to the "
        "conversation's owner. Tick each as done."
    )
    c1 = Msg("How much effort does the branching graph add to the frontend?")
    c2 = Msg(
        "The ReactFlow graph is the main frontend cost: layout, minimap, node actions "
        "and branch rendering. Budget an extra 3-5 days for visual polish and edge cases "
        "such as deleting a node's whole subtree."
    )
    d1 = Msg("Which hosting do you recommend for Postgres?")
    d2 = Msg(
        "A managed Postgres (e.g. Neon or a dedicated RDS instance) removes backup and "
        "patching overhead. Keep the connection through a pooler and tune the pool to "
        "your concurrency. The app uses asyncpg and Alembic on top."
    )
    e1 = Msg("Should we gate the public /models endpoint?")
    e2 = Msg(
        "No, /models is a read-only public catalog and is intentionally unauthenticated. "
        "The API key is only ever used by /chat. Keep it keyless to avoid rate-limited "
        "homepage calls."
    )
    f1 = Msg("How do you handle deleted branches?")
    f2 = Msg(
        "Deleting a node removes its entire subtree via a cascading delete, while sibling "
        "branches stay untouched. Users get a confirm step so a destructive action is never "
        "accidental."
    )
    g1 = Msg("Is there a way to compare two branch answers side by side?")
    g2 = Msg(
        "The graph view shows every branch simultaneously; to compare answers visually, open "
        "the two leaves' panels next to each other. A future nicety could be an explicit "
        "diff mode between sibling replies."
    )
    h1 = Msg("What does temporary messages mean exactly?")
    h2 = Msg(
        "Temporary messages carry an expiry (1h, 24h, or end-of-session) and are pruned "
        "automatically. This keeps throwaway reasoning from bloating a conversation and "
        "from being stored longer than needed."
    )
    i1 = Msg("Show me a past-deployment circa: custom domains?")
    i2 = Msg(
        "Custom domains are outside v1. The CORS allow-list is configurable via CORS_ORIGINS, "
        "so a reverse proxy or custom origin is trivial to wire in later without code changes."
    )

    root.children = [a1]
    a1.children = [b1, c1, d1, e1, f1, g1, h1, i1]
    b1.children = [b2]
    b2.children = [b3]
    b3.children = [b4]
    c1.children = [c2]
    d1.children = [d2]
    e1.children = [e2]
    f1.children = [f2]
    g1.children = [g2]
    h1.children = [h2]
    i1.children = [i2]
    return BranchTree([root])


def _deep_leaf(tree: BranchTree) -> Msg:
    return tree.deepest_leaf()


def test_branch_context_uses_lineage_not_full_history():
    tree = build_conversation()
    deep = _deep_leaf(tree)

    naive = tokens_for(tree.turns())  # a linear chat replays everything
    branch = tokens_for(tree.lineage(deep))  # Ramea: ancestor path only

    assert branch > 0
    assert branch < naive
    assert branch <= naive * 0.55, f"branch {branch} not clearly under full-history {naive}"


def test_sibling_branches_are_excluded_from_context():
    """Untouched sibling branches must never leak into the prompt."""
    tree = build_conversation()
    lineage = tree.lineage(_deep_leaf(tree))
    texts = " ".join(m.text.lower() for m in lineage)

    assert "checklist" in texts  # the deep b-branch IS the active path
    assert "reactflow" not in texts, "sibling c branch leaked into context"
    assert "managed postgres" not in texts, "sibling d branch leaked into context"
    assert "models endpoint" not in texts, "sibling e branch leaked into context"
    assert "deleting a node removes" not in texts, "sibling f branch leaked into context"
    assert "side by side" not in texts, "sibling g branch leaked into context"
    assert "temporary messages carry" not in texts, "sibling h branch leaked into context"
    assert "custom domains" not in texts, "sibling i branch leaked into context"


def test_cost_grows_with_depth_not_total_messages():
    """Over all turns, lineage-only contexts cost far less than replaying history."""
    tree = build_conversation()
    turns = tree.turns()

    # Classic linear client: on turn i it re-sends messages 0..i. Sum over all turns.
    cumulative_history_tokens = sum(
        tokens_for(turns[: i + 1]) for i in range(len(turns))
    )
    # Ramea: each turn streams only its own ancestor path.
    lineage_tokens = sum(tokens_for(tree.lineage(m)) for m in turns)

    assert lineage_tokens <= cumulative_history_tokens * 0.60


def test_reports_ratio_for_readme(capsys):
    """Print the concrete number so the README can quote a reproducible saving."""
    tree = build_conversation()
    deep = _deep_leaf(tree)
    naive = tokens_for(tree.turns())
    branch = tokens_for(tree.lineage(deep))
    saved = 100 - (branch / naive * 100)
    print(
        f"\n[Efficiency] full-history context = {naive} tokens; "
        f"Ramea lineage = {branch} tokens; "
        f"{saved:.1f}% fewer input tokens on this turn."
    )
    print(
        f"[Efficiency] all-turns: naive cumulative = "
        f"{sum(tokens_for(tree.turns()[: i + 1]) for i in range(len(tree.turns())))}; "
        f"lineage-sum = "
        f"{sum(tokens_for(tree.lineage(m)) for m in tree.turns())}."
    )