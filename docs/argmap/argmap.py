#!/usr/bin/env python3
"""argmap - validate a directory of argument-map notes.

Project-agnostic. Standard library only. Run it as:

    python argmap.py [directory] [--json] [--strict]

It discovers the thesis from frontmatter (`type: thesis`), builds the edge
graph from the body sections, and prints ten numbered sections:

    0   note format         every note parses; frontmatter keys, types,
                            statuses and evidence classes are valid
    0b  one thesis          a map has exactly one root
    1   support beneath     what a `supported` claim actually rests on
        supported claims    (empty, no observation, `ours` only, withheld only)
    2   standing refutation a `supported` node with a `supported` refuter
    2b  retractions         `void` nodes with no replacement, and live nodes
                            still resting on a `void` one
    3   thesis premises     anything the thesis requires, transitively, with
                            no observation beneath it
    3b  held premises       anything a `supported` node requires,
                            transitively, that is not itself held
    4   overturned sources  nodes citing a source the project retracted
    5a  broken wikilinks
    5b  unreachable nodes

Optional per-project settings live in `argmap.json` beside the notes;
everything in it is optional and every check that depends on it degrades to
"skipped" rather than failing.

See SPEC.md for the file format, GUIDE.md for a full description of the
system, and AGENT.md for how to work in one.

Options:
    --json      emit the whole report as JSON on stdout (nothing else), for a
                tool or an agent that wants to act on findings rather than
                read them. Exit status is unchanged.
    --strict    count notes (the advisory lines the text report prints under a
                section) as findings, so they affect the exit status.

Exit status: 0 when nothing is flagged, 1 when something is, 2 on a usage or
structural error (not a directory, no notes, no thesis).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# the vocabulary: four edge types, four node types, five statuses,
# three evidence classes
# --------------------------------------------------------------------------

#  heading (lowercased, no trailing colon) -> (relation, direction)
#  "out" = this note asserts the edge; "in" = this note mirrors an edge
#  another note asserts. Both are parsed; the graph is one directed edge set.
HEADINGS = {
    "supports": ("supports", "out"),
    "refutes": ("refutes", "out"),
    "requires": ("requires", "out"),
    "superseded by": ("superseded-by", "out"),
    "superseded-by": ("superseded-by", "out"),
    "supported by": ("supports", "in"),
    "refuted by": ("refutes", "in"),
    "required by": ("requires", "in"),
    "supersedes": ("superseded-by", "in"),
}

#  A heading that is not in HEADINGS but contains one of these stems is almost
#  certainly a misspelled edge heading, and a misspelled edge heading silently
#  drops every edge under it. Flagged rather than ignored.
EDGE_STEMS = ("support", "refut", "requir", "supersed", "supercede", "superced")

RELATIONS = ("supports", "refutes", "requires", "superseded-by")
NODE_TYPES = ("thesis", "claim", "observation", "question")
STATUSES = ("supported", "open", "unsupported", "withheld", "void")
EVIDENCE_CLASSES = ("experimental", "literature", "ours")

# A node counts as asserted-and-standing: load-bearing for checks 1, 2, 2b
# and 4. This is the one status that carries a commitment.
SUPPORTED = ("supported",)

# Evidence exists and is deliberately unread (a blind prediction, a held-back
# dataset). Not load-bearing, and deliberately not reported as missing: see
# checks 1 and 3.
WITHHELD = "withheld"

# Held once, now retracted. Must carry a `superseded-by` edge, and must not
# still be bearing weight: see check 2b.
VOID = "void"

DEFAULTS = {
    # file listing sources this project has overturned; checked by check 4
    "retracted_sources": None,
    # headings in that file to ignore (substring match, lowercased)
    "retracted_sources_ignore_headings": ["already marked"],
    # how a cited source id looks, e.g. "D\\d+" for a decision log, or
    # "[0-9]{4}-[0-9]{2}-[0-9]{2}" for dated journal entries
    "source_id_pattern": r"[A-Za-z]+\d+",
    # frontmatter key holding the citations
    "source_field": "entries",
    # frontmatter key listing overturned sources whose superseding source the
    # author has read. Check 4 stops flagging a citation once it is listed
    # here, so the list is an assertion: "I read the correction."
    "acknowledged_field": "checked",
    # which nodes must carry a non-empty `source_field`; checked by check 0.
    # "auto" = "supported" when a journal is configured, else "none";
    # "all" = every node; "supported" = every load-bearing node; "none" = off.
    "require_sources": "auto",
    # free-text pointer, printed in the header so a reader knows the journal
    "journal": None,
}

CONFIG_KEYS = tuple(DEFAULTS)


# --------------------------------------------------------------------------
# reporting primitives (defined early: parsing reports through them)
# --------------------------------------------------------------------------

class Item:
    """One line of the report: what is wrong, and what to do about it."""

    def __init__(self, node, problem: str, fix: str = ""):
        self.node = node
        self.problem = problem
        self.fix = fix

    def text(self) -> str:
        return f"{self.node}: {self.problem}" if self.node else self.problem

    def as_dict(self) -> dict:
        return {"node": self.node, "problem": self.problem, "fix": self.fix}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

WIKILINK = re.compile(r"\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]")
INLINE_LIST = re.compile(r"^\[(.*)\]$")
#  A list item: `- `, `* `, `+ `, or a numbered `1.` / `1)`. Wikilinks on these
#  lines under an edge heading are edges.
LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])(?:\s|$)")
FENCE = re.compile(r"^(```|~~~)")


def read_note(path: Path) -> str:
    """Read a file as UTF-8, tolerating a byte-order mark - some Windows
    editors write one, and it would otherwise hide the `---` frontmatter."""
    return path.read_text(encoding="utf-8-sig")


def parse_frontmatter(text: str):
    """Return (frontmatter, body, warnings).

    Supports the small YAML subset the format uses: `key: scalar`,
    `key: [a, b]`, and a `-` block list under a bare `key:`. Anything outside
    that subset is **reported** in `warnings` rather than skipped silently, so
    frontmatter the parser does not understand fails loudly instead of quietly
    producing a node with no status.
    """
    warnings = []
    if not text.startswith("---"):
        return {}, text, ["no `---` frontmatter block at the top of the file"]
    lines = text.splitlines()
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return {}, text, [
            "frontmatter opens with `---` but is never closed by a second "
            "`---` on its own line"
        ]
    meta = {}
    key = None
    for lineno, raw in enumerate(lines[1:end], start=2):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue  # a comment; intentionally ignored
        if line.lstrip().startswith("-"):
            if key is None:
                warnings.append(
                    f"line {lineno}: list item `{line.strip()}` with no key "
                    f"above it")
                continue
            if not isinstance(meta.get(key), list):
                warnings.append(
                    f"line {lineno}: list item under `{key}:`, which already "
                    f"holds a scalar; write either `{key}: [a, b]` or a bare "
                    f"`{key}:` followed by `- a` lines, not both")
                continue
            meta[key].append(line.lstrip()[1:].strip().strip("\"'"))
            continue
        if ":" not in line:
            warnings.append(
                f"line {lineno}: `{line.strip()}` is neither `key: value` nor "
                f"a `- item` list entry; the parser understands only "
                f"`key: scalar`, `key: [a, b]` and `- item` blocks")
            continue
        if line[0].isspace():
            warnings.append(
                f"line {lineno}: indented key `{line.strip()}`; nested "
                f"frontmatter is not supported - flatten it to a top-level "
                f"`key: value`")
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            warnings.append(f"line {lineno}: empty key in `{line.strip()}`")
            continue
        if key in meta:
            warnings.append(f"line {lineno}: duplicate key `{key}`")
        if not value:
            meta[key] = []
            continue
        m = INLINE_LIST.match(value)
        if m:
            inner = m.group(1).strip()
            meta[key] = [
                v.strip().strip("\"'") for v in inner.split(",") if v.strip()
            ] if inner else []
        else:
            meta[key] = value.strip("\"'")
    return meta, "\n".join(lines[end + 1:]), warnings


def parse_edges_directed(body: str):
    """Return ([(relation, direction, target_slug)], warnings).

    `warnings` names headings that look like misspelled edge headings, because
    an unrecognised heading silently discards every wikilink beneath it - the
    quietest way to lose an edge.
    """
    out = []
    warnings = []
    current = None
    fence = None
    # per edge section: its heading, how many list items it held, and how
    # many wikilinks sat on non-list lines
    section = None

    def close(sec):
        # Prose with wikilinks after an edge list is normal and is prose. An
        # edge section holding wikilinks but *no* list item at all is an
        # author who meant edges and wrote them in a form the parser skips.
        if sec and sec["items"] == 0 and sec["stray"]:
            warnings.append(
                f"`{sec['heading']}` holds {sec['stray']} wikilink(s) but no "
                f"list item, so none of them are edges; write each as a "
                f"`- [[target]]` (or `1. [[target]]`) line")

    for line in body.splitlines():
        stripped = line.strip()
        # Nothing inside a fenced code block is a heading or an edge.
        m = FENCE.match(stripped)
        if fence:
            if m and m.group(1) == fence:
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        if stripped.startswith("#"):
            close(section)
            section = None
            level = len(stripped) - len(stripped.lstrip("#"))
            name = stripped.lstrip("#").strip().rstrip(":").lower()
            current = HEADINGS.get(name)
            if current:
                section = {"heading": stripped, "items": 0, "stray": 0}
            # Only a short level-2-or-deeper heading is a plausible misspelled
            # edge heading. The node's own `#` title and any prose heading
            # ("## Why the refutation does not bite") are left alone.
            if (current is None and level >= 2 and len(name.split()) <= 3
                    and any(stem in name for stem in EDGE_STEMS)):
                warnings.append(
                    f"heading `{stripped}` looks like an edge heading but is "
                    f"not one, so any links under it are not edges; the edge "
                    f"headings are "
                    f"{', '.join('## ' + h for h in HEADINGS)}")
            continue
        if current is None:
            continue
        targets = WIKILINK.findall(stripped)
        if not LIST_ITEM.match(stripped):
            section["stray"] += len(targets)
            continue
        section["items"] += 1
        for target in targets:
            out.append((current[0], current[1], target.strip()))
    close(section)
    if fence:
        warnings.append(
            f"a {fence} code fence is never closed, so everything after it "
            f"is ignored")
    return out, warnings


class Node:
    def __init__(self, path: Path, meta: dict, body: str):
        self.path = path
        self.slug = path.stem
        self.meta = meta
        self.body = body
        self.type = str(meta.get("type", "")).strip().lower()
        self.status = str(meta.get("status", "")).strip().lower()
        self.evidence = str(meta.get("evidence", "")).strip().lower() or None

    def sources(self, field: str):
        value = self.meta.get(field, [])
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(v).strip() for v in value if str(v).strip()]


def load_config(directory: Path):
    config = dict(DEFAULTS)
    warnings = []
    path = directory / "argmap.json"
    if path.exists():
        try:
            given = json.loads(read_note(path))
        except (OSError, ValueError) as exc:
            warnings.append(f"could not read {path.name}: {exc}")
        else:
            if not isinstance(given, dict):
                warnings.append(f"{path.name}: expected a JSON object")
            else:
                for k in given:
                    if k not in CONFIG_KEYS:
                        warnings.append(
                            f"{path.name}: unknown key `{k}`; the keys are "
                            f"{', '.join(CONFIG_KEYS)}")
                config.update(given)
    return config, warnings


def load_nodes(directory: Path, config: dict):
    """Return (nodes, rejected, problems).

    A note whose `type` is invalid is **rejected**: it is reported and kept out
    of the graph, so it contributes no edges and no reachability. It stays in
    `rejected` so a wikilink pointing at it can say what happened rather than
    report the file as missing.
    """
    nodes = {}
    rejected = []
    problems = []
    skip = {"readme", "spec", "agent", "guide", "index"}
    source_field = config.get("source_field", "entries")
    ack_field = config.get("acknowledged_field", "checked")
    allowed_keys = {"type", "status", "evidence", source_field, ack_field}

    require = str(config.get("require_sources", "auto")).strip().lower()
    if require == "auto":
        require = "supported" if config.get("journal") else "none"
    if require not in ("none", "supported", "all"):
        problems.append(Item(
            None, f"argmap.json: require_sources is {require!r}",
            'use one of "auto", "none", "supported", "all"'))
        require = "none"

    for path in sorted(directory.glob("*.md")):
        if path.stem.lower() in skip:
            continue
        text = read_note(path)
        meta, body, fm_warnings = parse_frontmatter(text)
        for warning in fm_warnings:
            if meta or "no `---` frontmatter" not in warning:
                problems.append(Item(
                    path.stem, f"frontmatter: {warning}",
                    "fix the frontmatter; a key the parser drops is a field "
                    "the checks cannot see"))
        if not meta:
            problems.append(Item(
                path.stem,
                "no frontmatter the parser could read; not treated as a node, "
                "so none of the checks see it",
                "add a `---` block with at least `type:` and `status:`, or "
                "move the file out of the map directory"))
            continue

        node = Node(path, meta, body)
        unknown = sorted(set(meta) - allowed_keys)
        if unknown:
            problems.append(Item(
                node.slug, f"unknown frontmatter key(s): {', '.join(unknown)}",
                f"the frontmatter holds only "
                f"{', '.join(sorted(allowed_keys))}; a misspelled key is read "
                f"as a missing field - put prose in the body"))

        bad_type = node.type not in NODE_TYPES
        if bad_type:
            problems.append(Item(
                node.slug,
                f"type {node.type or '(missing)'!r} is not one of "
                f"{', '.join(NODE_TYPES)}, so the note is excluded from the "
                f"graph entirely",
                f"set `type:` to one of {', '.join(NODE_TYPES)}"))

        if node.status not in STATUSES:
            problems.append(Item(
                node.slug,
                f"status {node.status or '(missing)'!r} is not one of "
                f"{', '.join(STATUSES)}; an unrecognised status is not "
                f"load-bearing, so checks 1, 2, 2b, 3b and 4 would not examine "
                f"this node at all",
                f"set `status:` to one of {', '.join(STATUSES)}"))

        if node.type == "observation" and node.evidence not in EVIDENCE_CLASSES:
            problems.append(Item(
                node.slug,
                f"observation has evidence {node.evidence!r}, not one of "
                f"{', '.join(EVIDENCE_CLASSES)}",
                "add `evidence:` - `experimental` for a deposited structure or "
                "a measurement from one, `literature` for a published result, "
                "`ours` for our own calculation"))
        if node.type != "observation" and node.evidence:
            problems.append(Item(
                node.slug, f"evidence class on a {node.type or 'node'}",
                "remove `evidence:`; it belongs on observations, which are the "
                "only nodes that may be cited as evidence"))

        if require != "none" and not bad_type:
            if (require == "all" or node.status in SUPPORTED) \
                    and not node.sources(source_field):
                problems.append(Item(
                    node.slug,
                    f"`{source_field}` is empty or missing on a "
                    f"{node.status or 'statusless'} node",
                    f"list the journal entries this node came from in "
                    f"`{source_field}`; a node with no provenance cannot be "
                    f"traced back, and check 4 cannot tell whether its sources "
                    f"were overturned"))

        for warning in parse_edges_directed(body)[1]:
            problems.append(Item(
                node.slug, warning,
                "rename the heading to an edge heading, or move the links into "
                "prose if they are not edges"))

        if bad_type:
            rejected.append(node)
            continue
        # Ids are filenames, so an exact duplicate cannot exist; but two ids
        # that differ only in case collide on a case-insensitive filesystem
        # and in editors that resolve wikilinks case-insensitively.
        clash = next((s for s in nodes if s.lower() == node.slug.lower()), None)
        if clash:
            problems.append(Item(
                node.slug, f"node id differs from {clash} only in case",
                "rename one of the files; on Windows and macOS, and in most "
                "wikilink editors, the two are the same note"))
        nodes[node.slug] = node
    return nodes, rejected, problems


# --------------------------------------------------------------------------
# retracted-source index (optional, project-configured)
# --------------------------------------------------------------------------

def load_retracted(path: Path, pattern: str, ignore):
    """Extract source ids from a retractions file.

    Generic over format: it reads ids out of the first cell of any table row
    and out of any `- **ID**` bullet, which is how such an index is normally
    written. Headings matching `ignore` are skipped.
    """
    rx = re.compile(pattern)
    bullet = re.compile(r"^\s*[-*]\s+\**\s*(" + pattern + r")")
    found = set()
    skipping = False
    for line in read_note(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            skipping = any(token in heading for token in ignore)
            continue
        if skipping:
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not cells or set(cells[0]) <= set("-: "):
                continue
            found.update(rx.findall(cells[0]))
            continue
        m = bullet.match(stripped)
        if m:
            found.add(m.group(1))
    return found


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.edges = set()      # (src, rel, dst)
        # (holder, src, rel, dst, missing) - `missing` is the end that does not
        # exist, which under a mirror heading is the *source* of the edge, not
        # the node whose file the wikilink sits in.
        self.broken = []
        for slug, node in nodes.items():
            for rel, direction, target in parse_edges_directed(node.body)[0]:
                src, dst = (slug, target) if direction == "out" else (target, slug)
                if target not in nodes:
                    self.broken.append((slug, src, rel, dst, target))
                    continue
                self.edges.add((src, rel, dst))

    def inbound(self, slug: str, rel: str):
        return sorted(s for (s, r, d) in self.edges if d == slug and r == rel)

    def outbound(self, slug: str, rel: str):
        return sorted(d for (s, r, d) in self.edges if s == slug and r == rel)

    def connected_to(self, roots):
        """Undirected reachability: a node participates in the argument if it
        is linked to the thesis by any chain of any of the four edge types."""
        adjacency = {s: set() for s in self.nodes}
        for src, _rel, dst in self.edges:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
        seen = set(roots)
        stack = list(roots)
        while stack:
            for neighbour in adjacency.get(stack.pop(), ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen


class Support:
    """What a node rests on, following inbound `supports` edges transitively.

    One hop is not enough: a claim correctly supported by evidence *through an
    intermediate claim* has no observation directly under it, and a one-hop
    check calls that unsupported. So the walk descends through intermediate
    claims and stops at observations, which are where the argument touches the
    world.

    It descends through an intermediate **only when that intermediate is
    itself held** - a `claim` or `thesis` with `status: supported`. Everything
    else terminates the chain and is recorded separately, because a chain
    through a node nobody holds carries nothing:

    * `void`      - retracted; its presence in the chain is check 2b's finding
    * `withheld`  - the evidence exists and is deliberately unread, which is
                    not the same as absent, so it is reported apart
    * `question`  - asserts nothing, so it cannot pass an assertion on
    * anything else (`open`, `unsupported`, an unrecognised status) - stated,
                    not settled

    This keeps the walk at least as strong as the one-hop version it replaces:
    everything one hop flagged is still flagged, and a legitimate chain through
    a supported intermediate is no longer miscounted as unsupported.
    """

    def __init__(self, graph: Graph, nodes, slug: str):
        self.direct = graph.inbound(slug, "supports")
        self.observations = set()
        self.via_void = set()
        self.via_withheld = set()
        self.via_question = set()
        self.via_unheld = set()
        self.all = set()
        stack = [slug]
        while stack:
            for s in graph.inbound(stack.pop(), "supports"):
                if s in self.all:
                    continue
                self.all.add(s)
                node = nodes[s]
                if node.status == VOID:
                    self.via_void.add(s)
                elif node.status == WITHHELD:
                    self.via_withheld.add(s)
                elif node.type == "observation":
                    self.observations.add(s)
                elif node.type == "question":
                    self.via_question.add(s)
                elif node.status in SUPPORTED:
                    stack.append(s)
                else:
                    self.via_unheld.add(s)
        self.classes = {nodes[s].evidence for s in self.observations
                        if nodes[s].evidence}

    def blocked(self) -> str:
        """Why a chain stopped short of an observation, for a report line."""
        parts = []
        for label, group in (("not themselves held", self.via_unheld),
                             ("open questions", self.via_question),
                             ("void", self.via_void),
                             ("withheld", self.via_withheld)):
            if group:
                parts.append(f"{label}: {', '.join(sorted(group))}")
        return "; ".join(parts)


def premise_closure(graph: Graph, nodes, roots):
    """Everything the thesis requires, transitively: slug -> the node that
    requires it. A premise of a premise is a premise."""
    found = {}
    stack = list(roots)
    while stack:
        current = stack.pop()
        for premise in graph.outbound(current, "requires"):
            if premise in found or premise in roots:
                continue
            found[premise] = current
            stack.append(premise)
    return found


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

class Report:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.findings = 0
        self.note_count = 0
        self.skipped = []
        self.sections = []

    def section(self, number, title, items, notes=None,
                clean="nothing flagged", note_header=""):
        notes = notes or []
        self.findings += len(items)
        self.note_count += len(notes)
        if self.strict:
            self.findings += len(notes)
        self.sections.append({
            "check": str(number),
            "title": title,
            "status": "flagged" if items or (self.strict and notes) else "clean",
            "findings": [i.as_dict() for i in items],
            "notes": [i.as_dict() for i in notes],
        })
        print(f"\n[{number}] {title}")
        if not items:
            print(f"    {clean}")
        for item in items:
            print(f"    - {item.text()}")
            if item.fix:
                print(f"      fix: {item.fix}")
        if notes:
            suffix = " (counted: --strict)" if self.strict else ""
            print(f"    note: {note_header or 'advisory'}{suffix}:")
            for item in notes:
                print(f"      . {item.text()}")

    def skip(self, number, title, why):
        self.sections.append({
            "check": str(number), "title": title, "status": "skipped",
            "why": why, "findings": [], "notes": [],
        })
        print(f"\n[{number}] {title}")
        print(f"    skipped: {why}")
        self.skipped.append(f"{number} {title}")


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = [a for a in argv[1:] if a.startswith("-")]
    for flag in flags:
        if flag not in ("--json", "--strict", "-h", "--help"):
            print(f"unknown option: {flag}", file=sys.stderr)
            print("usage: argmap.py [directory] [--json] [--strict]",
                  file=sys.stderr)
            return 2
    if "-h" in flags or "--help" in flags:
        print(__doc__)
        return 0
    if len(args) > 1:
        print("usage: argmap.py [directory] [--json] [--strict]",
              file=sys.stderr)
        return 2

    as_json = "--json" in flags
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    if as_json:
        # the text report must not reach stdout in --json mode; stderr is
        # still shown, and also captured so a structural error reaches the
        # JSON consumer instead of leaving stdout empty
        import io
        sys.stdout = io.StringIO()
        captured = io.StringIO()

        class Tee:
            def write(self, s):
                captured.write(s)
                return real_stderr.write(s)

            def flush(self):
                real_stderr.flush()

        sys.stderr = Tee()
    try:
        code = run(args[0] if args else ".", as_json, "--strict" in flags,
                   real_stdout)
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr
    if as_json and code == 2:
        json.dump({"error": captured.getvalue().strip(), "exit": 2},
                  real_stdout, indent=2)
        real_stdout.write("\n")
    return code


def run(where, as_json, strict, real_stdout):
    directory = Path(where).resolve()
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    config, config_warnings = load_config(directory)
    for warning in config_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    nodes, rejected, problems = load_nodes(directory, config)
    if not nodes:
        print(f"no argument-map notes found in {directory}", file=sys.stderr)
        if rejected:
            print(f"({len(rejected)} note(s) rejected for an invalid `type`)",
                  file=sys.stderr)
        return 2

    graph = Graph(nodes)
    theses = [s for s, n in nodes.items() if n.type == "thesis"]

    counts = {t: sum(1 for n in nodes.values() if n.type == t)
              for t in NODE_TYPES}
    statuses = {s: sum(1 for n in nodes.values() if n.status == s)
                for s in STATUSES}
    evidence = {c: sum(1 for n in nodes.values() if n.evidence == c)
                for c in EVIDENCE_CLASSES}

    print(f"argmap: {directory}")
    if config.get("journal"):
        print(f"journal: {config['journal']}  (the graph holds what we believe; "
              f"the journal holds what happened)")
    print("nodes:   " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("status:  " + ", ".join(f"{k} {v}" for k, v in statuses.items()))
    print("evidence:" + " " + ", ".join(f"{k} {v}" for k, v in evidence.items()))
    print("edges:   " + ", ".join(
        f"{r} {sum(1 for e in graph.edges if e[1] == r)}" for r in RELATIONS))

    report = Report(strict=strict)

    report.section("0", "Note format", problems,
                   clean="all notes parse; types, statuses, evidence classes "
                         "and frontmatter keys valid")

    if not theses:
        print("\nERROR: no node has `type: thesis`; nothing to check against.",
              file=sys.stderr)
        print("Give the map a root: one node with `type: thesis`, stating an "
              "assertion rather than a question.", file=sys.stderr)
        return 2
    if len(theses) > 1:
        report.section("0b", "More than one thesis",
                       [Item(s, "a map has one root",
                             "keep one `type: thesis` and demote the rest to "
                             "`type: claim`") for s in sorted(theses)])

    # --- 1 -----------------------------------------------------------------
    # What a `supported` claim rests on, following `supports` transitively.
    items, notes = [], []
    for slug, node in sorted(nodes.items()):
        if node.type not in ("claim", "thesis") or node.status not in SUPPORTED:
            continue
        support = Support(graph, nodes, slug)
        if not support.all:
            items.append(Item(
                slug, "marked supported with an empty support set",
                "add a `## Supported by` here, or a `## Supports` on the "
                "observation that evidences it - or set `status: unsupported` "
                "and say in the body that it is an assertion"))
        elif not support.observations and support.via_withheld and not (
                support.via_unheld or support.via_question or support.via_void):
            items.append(Item(
                slug,
                f"marked supported, but the only support that reaches evidence "
                f"ends at a withheld node "
                f"({', '.join(sorted(support.via_withheld))})",
                "withheld evidence is deliberately unread, so it cannot carry "
                "a commitment: set `status: open` until the evidence is read"))
        elif not support.observations:
            items.append(Item(
                slug,
                f"marked supported with no observation anywhere beneath it; "
                f"every chain stops at a node that cannot pass support on "
                f"({support.blocked()})",
                "an argument has to touch the world somewhere: add the "
                "observation the chain rests on, promote the intermediate if "
                "it is in fact held, or set `status: unsupported`"))
        elif support.classes == {"ours"}:
            items.append(Item(
                slug,
                f"marked supported, and every observation beneath it is our "
                f"own calculation ({', '.join(sorted(support.observations))})",
                "a calculation performed on an input is evidence about the "
                "calculation, not about the input: find an `experimental` or "
                "`literature` observation, or set `status: open`"))
        elif support.classes == {"literature"}:
            notes.append(Item(
                slug,
                f"every observation beneath it is `literature`, read rather "
                f"than re-derived ({', '.join(sorted(support.observations))})"))
    report.section(
        1, "Support beneath supported claims", items, notes,
        clean="every supported claim reaches an observation, and not only our "
              "own",
        note_header="the hierarchy is `experimental` > `literature` > `ours`; "
                    "these rest on the middle rank, which is legitimate but "
                    "weaker than a measurement")

    # --- 2 -----------------------------------------------------------------
    items = []
    for slug, node in sorted(nodes.items()):
        if node.status not in SUPPORTED:
            continue
        for refuter in graph.inbound(slug, "refutes"):
            if nodes[refuter].status in SUPPORTED:
                ev = nodes[refuter].evidence
                items.append(Item(
                    slug,
                    f"still marked supported, but {refuter} "
                    f"({nodes[refuter].type}{', ' + ev if ev else ''}, "
                    f"supported) refutes it",
                    "demote this claim, refute the refuter with evidence, or "
                    "record in the body why the refutation does not bite - do "
                    "not delete the edge and do not flip the status"))
    report.section(2, "Supported claims with a standing refutation", items)

    # --- 2b ----------------------------------------------------------------
    # A retraction has two halves: the void node must name its replacement,
    # and nothing live may still be resting on it.
    items = []
    for slug, node in sorted(nodes.items()):
        if node.status == VOID and not graph.outbound(slug, "superseded-by"):
            items.append(Item(
                slug,
                "status is void with no `superseded-by` edge, so the record "
                "says what was abandoned but not what replaced it",
                "add a `## Superseded by` section pointing at the node that "
                "replaced it; if nothing replaced it, the status is "
                "`unsupported` or `open`, not `void`"))
    for slug, node in sorted(nodes.items()):
        if node.status not in SUPPORTED:
            continue
        support = Support(graph, nodes, slug)
        for dead in sorted(support.via_void):
            items.append(Item(
                slug,
                f"marked supported, but its support runs through {dead}, "
                f"which is void",
                f"follow {dead}'s `superseded-by` edge and rest this claim on "
                f"the replacement, or demote this claim - a conclusion "
                f"outliving its premise is the failure the map exists to "
                f"catch"))
        for dead in sorted(d for d in graph.outbound(slug, "requires")
                           if nodes[d].status == VOID):
            items.append(Item(
                slug,
                f"marked supported, but it requires {dead}, which is void",
                f"point the `requires` edge at {dead}'s replacement, or drop "
                f"the premise and say in the body why it is no longer needed"))
    report.section("2b", "Retracted nodes, and live nodes resting on them",
                   items,
                   clean="every void node names its replacement, and nothing "
                         "supported rests on one")

    # --- 3 -----------------------------------------------------------------
    # Transitive: a premise of a premise is a premise, and a premise is
    # unsupported when nothing beneath it is an observation.
    items, notes = [], []
    premises = premise_closure(graph, nodes, theses)
    for premise, requirer in sorted(premises.items()):
        node = nodes[premise]
        if node.type == "question":
            notes.append(Item(
                premise, f"an open question gating {requirer} "
                         f"(status {node.status or 'unset'})"))
            continue
        if node.status == WITHHELD:
            notes.append(Item(
                premise,
                f"required by {requirer}, support deliberately withheld - the "
                f"evidence exists and is unread by design, not missing"))
            continue
        if node.status == VOID:
            items.append(Item(
                premise,
                f"required by {requirer}, and void - the thesis rests on a "
                f"premise the project has retracted",
                f"repoint {requirer}'s `requires` edge at this node's "
                f"replacement (see check 2b), or drop the premise and say in "
                f"{requirer}'s body why the thesis no longer needs it"))
            continue
        support = Support(graph, nodes, premise)
        if not support.all:
            items.append(Item(
                premise,
                f"required by {requirer}, support set empty "
                f"(status {node.status or 'unset'})",
                "the thesis is not available while this premise is an "
                "assertion: evidence it, or drop the `requires` edge and say "
                "in the body why the thesis does not need it"))
        elif not support.observations:
            items.append(Item(
                premise,
                f"required by {requirer}, no observation anywhere beneath it "
                f"(status {node.status or 'unset'}; chains stop at "
                f"{support.blocked()})",
                "trace the chain down to a measurement, a published result or "
                "a named calculation, and add that observation"))
    report.section(3, "Unsupported premises of the thesis", items, notes,
                   clean=f"every premise the thesis requires, transitively "
                         f"({len(premises)}), reaches an observation",
                   note_header="premises not expected to carry support")

    # --- 3b ----------------------------------------------------------------
    # `requires` propagates from every held node, not only the thesis: a
    # supported claim - or a supported observation computed on a structure we
    # built - is no more available than its premises. Check 3 already covers
    # the thesis, and check 2b a direct `requires` of a void node.
    items, notes = [], []
    for slug, node in sorted(nodes.items()):
        if node.type == "thesis" or node.status not in SUPPORTED:
            continue
        direct = set(graph.outbound(slug, "requires"))
        for premise, via in sorted(premise_closure(graph, nodes, [slug]).items()):
            p = nodes[premise]
            chain = "" if via == slug else f" (through {via})"
            if p.type == "question":
                notes.append(Item(
                    slug, f"gated by the open question {premise}{chain}"))
            elif p.status == WITHHELD:
                notes.append(Item(
                    slug, f"requires {premise}{chain}, which is withheld"))
            elif p.status == VOID:
                if premise in direct:
                    continue  # check 2b's finding
                items.append(Item(
                    slug,
                    f"marked supported, but it requires {premise}{chain}, "
                    f"which is void",
                    f"repoint the `requires` chain at {premise}'s replacement, "
                    f"or demote this node"))
            elif p.status not in SUPPORTED:
                items.append(Item(
                    slug,
                    f"marked supported, but it requires {premise}{chain}, "
                    f"which is not held (status {p.status or 'unset'})",
                    f"a node is no more available than its premises: settle "
                    f"{premise}, demote this node, or drop the `requires` edge "
                    f"and say in the body why it is not needed"))
    report.section(
        "3b", "Supported nodes resting on premises that are not held",
        items, notes,
        clean="every premise of every supported node is itself held",
        note_header="premises that gate a supported node by design")

    # --- 4 -----------------------------------------------------------------
    source_file = config.get("retracted_sources")
    field = config.get("source_field", "entries")
    if not source_file:
        report.skip(4, "Citations of overturned sources",
                    "no `retracted_sources` configured in argmap.json")
    else:
        path = (directory / source_file).resolve()
        if not path.exists():
            report.skip(4, "Citations of overturned sources",
                        f"{source_file} not found")
        else:
            retracted = load_retracted(
                path,
                config.get("source_id_pattern", DEFAULTS["source_id_pattern"]),
                [t.lower() for t in config.get(
                    "retracted_sources_ignore_headings",
                    DEFAULTS["retracted_sources_ignore_headings"])])
            ack_field = config.get("acknowledged_field", "checked")
            items, notes = [], []
            for slug, node in sorted(nodes.items()):
                acknowledged = set(node.sources(ack_field))
                cited = set(node.sources(field))
                # An acknowledgement may name the superseding entry as well
                # as the overturned one, so only an id the node does not cite
                # at all is reported - either a citation removed with its
                # acknowledgement left behind, or an entry the body discusses
                # without citing. A note, because the second is legitimate.
                uncited = sorted(acknowledged - cited)
                if uncited:
                    notes.append(Item(
                        slug,
                        f"`{ack_field}` lists {', '.join(uncited)}, not cited "
                        f"in `{field}` - stale if the citation was removed, "
                        f"fine if the body only discusses it"))
                hits = sorted(s for s in cited
                              if s in retracted and s not in acknowledged)
                if not hits:
                    continue
                if node.status in SUPPORTED:
                    items.append(Item(
                        slug,
                        f"cites {', '.join(hits)} - overturned, "
                        f"unacknowledged, and this node is marked supported",
                        f"read the superseding entry for each, make sure the "
                        f"body reflects it, then add the id to `{ack_field}`; "
                        f"leave it unacknowledged only when the node's point "
                        f"is that its own support is retracted"))
                else:
                    notes.append(Item(slug, f"cites {', '.join(hits)}"))
            report.section(
                4,
                f"Supported nodes citing overturned sources without "
                f"acknowledgement ({len(retracted)} sources listed in "
                f"{source_file})",
                items, notes,
                note_header="unsupported nodes citing an overturned source, "
                            "and acknowledgements of uncited entries; not "
                            "counted")

    # --- 5 -----------------------------------------------------------------
    # The missing end of a broken edge is reported as the missing end: under a
    # mirror heading the asserting note is the *target* of the wikilink, so the
    # edge's direction is not the one the wikilink is written in.
    rejected_types = {n.slug: n.type for n in rejected}
    items = []
    for holder, src, rel, dst, missing in sorted(graph.broken):
        why = (f"rejected at check 0 for type {rejected_types[missing]!r}, so "
               f"it is not in the graph" if missing in rejected_types
               else "no such note")
        items.append(Item(
            None,
            f"{src} -{rel}-> {dst}: [[{missing}]] {why} (link written in "
            f"{holder}.md)",
            f"create {missing}.md, correct the spelling of the wikilink in "
            f"{holder}.md, or remove the link"))
    report.section("5a", "Broken wikilinks", items)

    reachable = graph.connected_to(theses)
    items = [Item(
        slug, f"{nodes[slug].type}, no edge path to the thesis",
        "link it into the argument with `supports`, `refutes` or `requires`, "
        "or remove it - a node with no path to the thesis bears on nothing")
        for slug in sorted(set(nodes) - reachable)]
    report.section("5b", "Nodes with no path to the thesis", items)

    print()
    if report.findings:
        print(f"{report.findings} finding(s).", end=" ")
    else:
        print("No findings.", end=" ")
    if report.note_count and not report.strict:
        print(f"{report.note_count} note(s), not counted.", end=" ")
    if report.skipped:
        print(f"Skipped: {'; '.join(report.skipped)}.", end=" ")
    print()

    if as_json:
        json.dump({
            "directory": str(directory),
            "config": {k: config.get(k) for k in CONFIG_KEYS},
            "counts": {
                "type": counts, "status": statuses, "evidence": evidence,
                "edges": {r: sum(1 for e in graph.edges if e[1] == r)
                          for r in RELATIONS},
            },
            "strict": strict,
            "findings": report.findings,
            "notes": report.note_count,
            "skipped": report.skipped,
            "checks": report.sections,
            "exit": 1 if report.findings else 0,
        }, real_stdout, indent=2)
        real_stdout.write("\n")

    return 1 if report.findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
