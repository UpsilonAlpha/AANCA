#!/usr/bin/env python3
"""Fixture tests for argmap.py: `python test_argmap.py` (stdlib unittest).

Each test writes a tiny map into a temporary directory, runs the validator
with `--json` and asserts on the report. Generic, like the validator: copy it
alongside argmap.py.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).resolve().parent / "argmap.py"


def note(type_, status, body="", evidence=None, extra=""):
    lines = ["---", f"type: {type_}", f"status: {status}"]
    if evidence:
        lines.append(f"evidence: {evidence}")
    if extra:
        lines.append(extra)
    lines += ["---", "", "# heading", "", body]
    return "\n".join(lines) + "\n"


THESIS = note("thesis", "open", "## Requires\n- [[c]]\n")
CLAIM = note("claim", "supported", "## Supported by\n- [[o]]\n")
OBS = note("observation", "supported", evidence="experimental")


class Map:
    """A temporary map directory; `files` maps stem -> text (or bytes)."""

    def __init__(self, files, config=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for stem, text in files.items():
            path = self.dir / f"{stem}.md"
            if isinstance(text, bytes):
                path.write_bytes(text)
            else:
                path.write_text(text, encoding="utf-8")
        if config is not None:
            (self.dir / "argmap.json").write_text(json.dumps(config),
                                                  encoding="utf-8")

    def run(self, *flags):
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR), str(self.dir), "--json", *flags],
            capture_output=True, text=True)
        report = json.loads(proc.stdout) if proc.stdout.strip() else None
        return proc.returncode, report

    def close(self):
        self.tmp.cleanup()


def check(report, number):
    return next(c for c in report["checks"] if c["check"] == str(number))


def problems(report, number, kind="findings"):
    return [f"{i['node']}: {i['problem']}" for i in check(report, number)[kind]]


class ArgmapTest(unittest.TestCase):

    def report(self, files, config=None, *flags):
        m = Map(files, config)
        self.addCleanup(m.close)
        return m.run(*flags)

    # --- baseline ---------------------------------------------------------

    def test_clean_map_exits_zero(self):
        code, r = self.report({"t": THESIS, "c": CLAIM, "o": OBS})
        self.assertEqual(code, 0)
        self.assertEqual(r["findings"], 0)
        self.assertEqual(r["counts"]["edges"]["supports"], 1)

    def test_no_thesis_is_exit_2_with_json_error(self):
        code, r = self.report({"c": CLAIM, "o": OBS})
        self.assertEqual(code, 2)
        self.assertEqual(r["exit"], 2)
        self.assertIn("thesis", r["error"])

    # --- parsing ----------------------------------------------------------

    def test_byte_order_mark_is_tolerated(self):
        code, r = self.report({"t": THESIS, "c": CLAIM,
                               "o": b"\xef\xbb\xbf" + OBS.encode()})
        self.assertEqual(code, 0, problems(r, 0))

    def test_numbered_list_items_are_edges(self):
        claim = note("claim", "supported", "## Supported by\n1. [[o]]\n")
        code, r = self.report({"t": THESIS, "c": claim, "o": OBS})
        self.assertEqual(code, 0)
        self.assertEqual(r["counts"]["edges"]["supports"], 1)

    def test_prose_after_an_edge_list_is_not_flagged(self):
        claim = note("claim", "supported",
                     "## Supported by\n- [[o]]\n\nProse that mentions [[t]].\n")
        code, r = self.report({"t": THESIS, "c": claim, "o": OBS})
        self.assertEqual(code, 0, problems(r, 0))

    def test_edge_section_with_links_but_no_list_item_is_flagged(self):
        claim = note("claim", "supported",
                     "## Supported by\nSee [[o]] for the number.\n")
        code, r = self.report({"t": THESIS, "c": claim, "o": OBS})
        self.assertEqual(code, 1)
        self.assertTrue(any("no list item" in p for p in problems(r, 0)))

    def test_headings_inside_code_fences_are_ignored(self):
        claim = note("claim", "supported",
                     "## Supported by\n- [[o]]\n\n```\n## Refuted by\n"
                     "- [[ghost]]\n```\n")
        code, r = self.report({"t": THESIS, "c": claim, "o": OBS})
        self.assertEqual(code, 0, problems(r, "5a"))

    def test_misspelled_edge_heading_is_flagged(self):
        claim = note("claim", "supported", "## Supporting\n- [[o]]\n")
        _, r = self.report({"t": THESIS, "c": claim, "o": OBS})
        self.assertTrue(any("looks like an edge heading" in p
                            for p in problems(r, 0)))

    def test_unknown_key_and_bad_status_are_flagged(self):
        bad = note("claim", "suported", extra="entrys: [D1]")
        _, r = self.report({"t": THESIS, "c": CLAIM, "o": OBS, "x": bad})
        found = " | ".join(problems(r, 0))
        self.assertIn("unknown frontmatter key", found)
        self.assertIn("'suported'", found)

    def test_ids_differing_only_in_case_are_flagged(self):
        m = Map({"t": THESIS, "c": CLAIM, "o": OBS})
        self.addCleanup(m.close)
        try:
            (m.dir / "O.md").write_text(OBS, encoding="utf-8")
        except OSError:
            self.skipTest("filesystem cannot hold both")
        if len(list(m.dir.glob("*.md"))) < 4:
            self.skipTest("case-insensitive filesystem: one file overwrote "
                          "the other")
        _, r = m.run()
        self.assertTrue(any("only in case" in p for p in problems(r, 0)))

    # --- checks 1, 2, 2b --------------------------------------------------

    def test_empty_support_set(self):
        claim = note("claim", "supported")
        _, r = self.report({"t": THESIS, "c": claim})
        self.assertTrue(any("empty support set" in p for p in problems(r, 1)))

    def test_ours_only_support(self):
        obs = note("observation", "supported", evidence="ours")
        _, r = self.report({"t": THESIS, "c": CLAIM, "o": obs})
        self.assertTrue(any("our own calculation" in p
                            for p in problems(r, 1)))

    def test_standing_refutation(self):
        refuter = note("observation", "supported", "## Refutes\n- [[c]]\n",
                       evidence="experimental")
        _, r = self.report({"t": THESIS, "c": CLAIM, "o": OBS, "r": refuter})
        self.assertEqual(len(check(r, 2)["findings"]), 1)

    def test_void_without_replacement(self):
        dead = note("claim", "void", "## Supports\n- [[t]]\n")
        _, r = self.report({"t": THESIS, "c": CLAIM, "o": OBS, "d": dead})
        self.assertTrue(any("no `superseded-by`" in p
                            for p in problems(r, "2b")))

    # --- 3b: `requires` propagates from every held node -------------------

    def test_supported_node_requiring_an_unheld_premise(self):
        claim = note("claim", "supported",
                     "## Supported by\n- [[o]]\n## Requires\n- [[p]]\n")
        premise = note("claim", "open", "## Supported by\n- [[o]]\n")
        _, r = self.report({"t": THESIS, "c": claim, "o": OBS, "p": premise})
        self.assertTrue(any("requires p" in p and "not held" in p
                            for p in problems(r, "3b")))

    def test_premise_outside_the_thesis_chain_is_still_checked(self):
        # the observation is not under the thesis's `requires` at all
        obs = note("observation", "supported",
                   "## Supports\n- [[c]]\n## Requires\n- [[s]]\n",
                   evidence="experimental")
        structure = note("claim", "unsupported", "## Supports\n- [[t]]\n")
        _, r = self.report({"t": THESIS, "c": note("claim", "supported"),
                            "o": obs, "s": structure})
        self.assertTrue(any(p.startswith("o:") and "requires s" in p
                            for p in problems(r, "3b")))

    def test_transitive_void_premise(self):
        a = note("claim", "supported",
                 "## Supported by\n- [[o]]\n## Requires\n- [[b]]\n")
        b = note("claim", "supported",
                 "## Supported by\n- [[o]]\n## Requires\n- [[v]]\n")
        v = note("claim", "void", "## Superseded by\n- [[b]]\n")
        _, r = self.report({"t": THESIS, "c": CLAIM, "o": OBS,
                            "a": a, "b": b, "v": v})
        found = problems(r, "3b")
        self.assertTrue(any(p.startswith("a:") and "through b" in p
                            and "void" in p for p in found), found)
        # the direct `requires` of a void node is check 2b's, not 3b's
        self.assertFalse(any(p.startswith("b:") for p in found), found)
        self.assertTrue(any(p.startswith("b:") for p in problems(r, "2b")))

    def test_question_premise_is_a_note(self):
        claim = note("claim", "supported",
                     "## Supported by\n- [[o]]\n## Requires\n- [[q]]\n")
        q = note("question", "open")
        code, r = self.report({"t": THESIS, "c": claim, "o": OBS, "q": q})
        self.assertEqual(check(r, "3b")["findings"], [])
        self.assertEqual(len(check(r, "3b")["notes"]), 1)

    # --- check 4 ----------------------------------------------------------

    def retracted_map(self, extra):
        claim = note("claim", "supported", "## Supported by\n- [[o]]\n",
                     extra=extra)
        files = {"t": THESIS, "c": claim, "o": OBS}
        m = Map(files, {"retracted_sources": "retractions.txt",
                        "source_id_pattern": "D\\d+"})
        (m.dir / "retractions.txt").write_text(
            "| id | why |\n|---|---|\n| D7 | measured away |\n",
            encoding="utf-8")
        self.addCleanup(m.close)
        return m.run()

    def test_unacknowledged_overturned_citation(self):
        _, r = self.retracted_map("entries: [D7]")
        self.assertEqual(len(check(r, 4)["findings"]), 1)

    def test_acknowledged_overturned_citation(self):
        _, r = self.retracted_map("entries: [D7, D8]\nchecked: [D7, D8]")
        self.assertEqual(check(r, 4)["findings"], [])
        self.assertEqual(check(r, 4)["notes"], [])

    def test_acknowledgement_of_an_uncited_entry_is_a_note(self):
        _, r = self.retracted_map("entries: [D8]\nchecked: [D7]")
        self.assertEqual(check(r, 4)["findings"], [])
        self.assertTrue(any("not cited" in p
                            for p in problems(r, 4, "notes")))

    # --- check 5 ----------------------------------------------------------

    def test_broken_link_and_orphan(self):
        claim = note("claim", "supported",
                     "## Supported by\n- [[o]]\n- [[missing]]\n")
        orphan = note("claim", "open")
        _, r = self.report({"t": THESIS, "c": claim, "o": OBS, "x": orphan})
        self.assertEqual(len(check(r, "5a")["findings"]), 1)
        self.assertEqual(problems(r, "5b"),
                         ["x: claim, no edge path to the thesis"])


if __name__ == "__main__":
    unittest.main()
