"""Deterministic, API-key-free model of Ramea's core cost mechanism.

The single most important property of Ramea is that **context grows with
the depth of the active branch, not with the size of the whole tree.**

This module reproduces that decision *without* calling any LLM. It builds a
realistic branching conversation as a tree of messages, then compares two ways
of building the prompt sent to a model on every new turn:

.. _naive:
   replay the *entire* conversation every turn (a classic linear chat client,
   or an eager "stuff all history into context" approach).
.. _branch:
   replay *only the ancestor path* that leads to the active message, exactly
   what Ramea sends via its ``get_message_ancestor_path`` lineage CTE (the
   recursive CTE shown in the main README).

Token counts use OpenAI's ``cl100k_base`` tokenizer so the numbers reflect how
an OpenRouter-hosted model bills input tokens. No network is used; the results
are surfaced by the tests so a reviewer can quote the exact saving in the README.

The encouraging result is that the gap grows **super-linearly** with the
number/size of *other* branches: the more the user explores, the more Ramea
saves, because untouched branches never cost a token again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

ENCODER_NAME = "cl100k_base"


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return max(1, len(text) // 4)  # clearly-labelled fallback: ~4 chars/token
    return len(tiktoken.get_encoding(ENCODER_NAME).encode(text))


@dataclass(eq=False)
class Msg:
    text: str
    children: list["Msg"] = field(default_factory=list)

    def __hash__(self) -> int:
        return id(self)


class BranchTree:
    def __init__(self, msgs: list[Msg]):
        self.roots = msgs

    def _parent_map(self) -> dict[Msg, Msg]:
        parents: dict[Msg, Msg] = {}

        def walk(m: Msg, parent: Msg | None) -> None:
            if parent is not None:
                parents[m] = parent
            for c in m.children:
                walk(c, m)

        for root in self.roots:
            walk(root, None)
        return parents

    def lineage(self, leaf: Msg) -> list[Msg]:
        """Ancestor path root→leaf, in chat order (what Ramea streams)."""
        parents = self._parent_map()
        path: list[Msg] = []
        node: Msg | None = leaf
        while node is not None:
            path.append(node)
            node = parents.get(node)
        path.reverse()
        return path

    def deepest_leaf(self) -> Msg:
        """The leaf on the longest root->leaf path (the current active turn)."""
        best: Msg | None = None
        best_len = -1
        for node in self.turns():
            if not node.children:
                lineage_len = len(self.lineage(node))
                if lineage_len > best_len:
                    best_len = lineage_len
                    best = node
        assert best is not None
        return best

    def turns(self) -> list[Msg]:
        """Every user/assistant node in chat order."""
        out: list[Msg] = []

        def visit(m: Msg) -> None:
            out.append(m)
            for c in m.children:
                visit(c)

        for root in self.roots:
            visit(root)
        return out


def tokens_for(messages: list[Msg]) -> int:
    return sum(count_tokens(m.text) for m in messages)