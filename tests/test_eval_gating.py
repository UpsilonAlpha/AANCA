"""An update that skips its training step must not skip its evaluation.

Four runs came back with 38 or 39 held-out rows instead of 40, and `g17_long`'s
pre-registered `settled` pair did not exist, because the evaluation gate sat
inside the training loop **after every `continue`**: a crop that failed to load
took that update's held-out row with it
(``docs/claims/a-skipped-update-silently-skips-its-evaluation.md``).

The bug was structural -- a statement in the wrong place -- so these tests are
structural too. They read `scripts/train_growth.py` as a syntax tree and assert
where things sit, which is a weaker kind of test than driving the loop and the
right kind for this fault: the loop lives inside `main()` behind a dataset, a
device and ten thousand updates, and a behavioural test would have to fake all
three to observe one `if` being reachable.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "train_growth.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(io.open(SOURCE, encoding="utf-8").read())


def _training_loop(tree: ast.Module) -> ast.For:
    """The `for update in range(start_update + 1, args.updates + 1)` loop."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if isinstance(node.target, ast.Name) and node.target.id == "update":
            return node
    raise AssertionError("the training loop is gone; this file needs rereading")


def _eval_gate(loop: ast.For) -> ast.If:
    """The `if update % args.eval_every == 0 or update == args.updates` gate."""
    for node in loop.body:
        if isinstance(node, ast.If) and "eval_every" in ast.unparse(node.test):
            if "args.updates" in ast.unparse(node.test):
                return node
    raise AssertionError(
        "the held-out evaluation gate is no longer a direct statement of the "
        "training loop, which is exactly the bug this file is about"
    )


def test_the_evaluation_gate_is_not_inside_any_try(tree):
    """**The fix.** The gate must be a direct child of the loop, not of a try.

    If it sits inside the `try` that the skip paths raise out of, a skipped
    update loses its evaluation again -- the same fault in a new shape.
    """
    loop = _training_loop(tree)
    gate = _eval_gate(loop)
    assert gate in loop.body, "the evaluation gate is nested, not a loop statement"
    for node in ast.walk(loop):
        if isinstance(node, (ast.Try, ast.For, ast.While)) and node is not loop:
            assert gate not in getattr(node, "body", []), (
                "the evaluation gate is inside a try/loop and can be skipped"
            )


def test_nothing_in_the_training_loop_uses_a_bare_continue(tree):
    """`continue` is what skipped the evaluation; `_SkipUpdate` is what does not.

    A `continue` jumps past every remaining statement of the loop body, the
    evaluation gate included. Raising out of the body's `try` skips the
    training step and leaves the gate reachable, which is the whole difference.
    """
    loop = _training_loop(tree)
    offenders = [
        node.lineno
        for node in ast.walk(loop)
        if isinstance(node, ast.Continue)
    ]
    assert not offenders, (
        f"bare `continue` at {offenders} in the training loop: it will skip "
        f"that update's held-out evaluation. Raise _SkipUpdate instead."
    )


def test_the_skip_paths_raise_the_sentinel(tree):
    """Every skip is the sentinel, so one `except` covers all of them."""
    loop = _training_loop(tree)
    raises = [
        ast.unparse(node.exc)
        for node in ast.walk(loop)
        if isinstance(node, ast.Raise) and node.exc is not None
    ]
    assert raises, "the training loop no longer skips anything; unexpected"
    assert all(r.startswith("_SkipUpdate") for r in raises), raises
    # The four recorded skip paths: a fresh crop that will not load, a crop
    # shorter than the window, a non-finite loss, a non-finite gradient.
    assert len(raises) == 4, (
        f"{len(raises)} skip paths, expected 4. A new one is fine -- update "
        f"this count and the list in the _SkipUpdate docstring together."
    )


def test_the_sentinel_is_defined_and_documented(tree):
    """It is load-bearing control flow, so it says why it exists."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_SkipUpdate":
            assert ast.get_docstring(node), "_SkipUpdate needs its reason"
            return
    raise AssertionError("_SkipUpdate is not defined at module level")


def test_a_pool_miss_falls_back_instead_of_spending_the_update(tree):
    """The second half: a dropped pool entry draws a fresh crop.

    Both pool failures -- the crop will not load, and the crop did not
    reproduce its residue count -- used to `continue`, so a pool miss cost a
    training step *and* that update's evaluation. They now leave `resume` as
    `None` and fall through to the `if resume is None` fresh draw, which is
    what the update would have done had the pool not been consulted.
    """
    loop = _training_loop(tree)
    # Anywhere in the loop's subtree, not just its direct children: the body
    # now sits inside the `try` the skip paths raise out of.
    fallbacks = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "resume is None"
    ]
    assert len(fallbacks) == 1, (
        "the `if resume is None` fresh-draw fallback is gone; a pool miss "
        "spends the update again"
    )
    source = io.open(SOURCE, encoding="utf-8").read()
    pool_block = source[source.index("args.pool_probability > 0"):source.index("if resume is None")]
    assert "resume = None" in pool_block, (
        "a pool failure no longer clears `resume`, so it cannot fall through"
    )
    assert "continue" not in pool_block
