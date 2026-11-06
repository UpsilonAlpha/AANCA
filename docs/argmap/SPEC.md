# argmap - an argument map as a layer over a project journal

A format for recording **what a project believes and why**, as one markdown
note per claim, with typed links between them and a validator that can check
the result. It is designed to sit beside an append-only journal (a decision
log, a lab notebook, a changelog), not to replace one.

Read this file to understand the format. `GUIDE.md` is the self-contained
description of the whole system - the problem, every field, every check, the
configuration and how to adopt it elsewhere - and `AGENT.md` is how to work in
a map. The worked example is `../csr/claims/`.

---

## The one rule

**The graph holds what we believe; the journal holds what happened.**

A journal's chronology is itself evidence - it records a prediction's
timestamp, how long a contradiction went unnoticed, which decision was made
before which measurement. None of that can be migrated into a graph without
being destroyed, so it is not migrated. Instead every graph node **cites the
journal entries it came from**, and the journal is never edited to match the
graph.

The failure this fixes is a **linkage** failure, not a search failure. In an
append-only journal a correction lives inside a later entry, so a grep lands on
the retracted half, and a conclusion can keep being quoted after the premise it
rested on has been measured away. An index of retractions patches that for a
reader who checks the index. Typed inbound edges fix it structurally: the
refutation is attached to the thing it refutes, and a command can find every
place a standing refutation is being ignored.

---

## A node

One file per node, kebab-case filename, `.md`. The filename stem is the node's
id and the target of every wikilink to it. Minimal YAML frontmatter, then plain
markdown with `[[wikilinks]]` so a graph view works without plugins.

```markdown
---
type: claim
status: supported
entries: [D108, D208]
checked: [D145]
---

# A one-line statement of the claim, as the heading

Why it is held, with the numbers and the entry each number came from. A claim
that cannot be traced to a measurement, a source, or a named calculation is an
assertion: say so, and give it `status: unsupported`.

## Supports
- [[some-other-node]]

## Refuted by
- [[a-measurement-that-contradicts-it]]
```

### Frontmatter

| key | required | meaning |
| --- | --- | --- |
| `type` | yes | `thesis`, `claim`, `observation` or `question` |
| `status` | yes | see below |
| `evidence` | observations only | `experimental`, `literature` or `ours` |
| `entries` | see `require_sources` | the journal entries this node came from. Required on `supported` nodes by default when a `journal` is configured; see the configuration table |
| `checked` | optional | journal entries listed as overturned whose correction has been read (see check 4) |

Nothing else, and **an unrecognised key is reported**, because a misspelled key
reads as a missing field. The frontmatter exists so a validator can read the
graph; the argument lives in the body. The parser accepts `key: scalar`,
`key: [a, b]`, a bare `key:` followed by `- item` lines, and `#` comments;
anything else - nesting, a stray line, a duplicate key, unclosed frontmatter -
is reported rather than skipped.

### Node types

- **`thesis`** - the root. Exactly one per map. Everything else exists to
  support it, refute it, or be required by it. A thesis must be an **assertion**:
  "X is different from Y", not "what is X?". A question cannot be mapped -
  nothing can support or refute it - and a map rooted in one has no shape.
- **`claim`** - an assertion the project makes. It is held on the strength of
  its inbound `supports` edges, not on its own authority.
- **`observation`** - a datum: a measurement, a published result, a number from
  a named calculation. Carries an evidence class. An observation is where the
  argument touches the world, so it is the only node type that may be cited as
  evidence.
- **`question`** - something open that bears on the argument. It asserts
  nothing, so it never passes support on to a claim above it. Link it with
  `requires` from the claim whose settling it blocks, so it is visible as a gap
  rather than invisible as an absence.

A note whose `type` is not one of the four is **excluded from the graph
entirely** - it contributes no edges and no reachability - and reported at
check 0.

### Status

| value | meaning |
| --- | --- |
| `supported` | the project holds this, and the support is recorded here |
| `open` | stated, not settled; evidence is partial or contested |
| `unsupported` | asserted with no traceable support - an assertion, marked as one |
| `withheld` | evidence exists and is deliberately unread (a blind prediction, a held-back dataset) |
| `void` | held once, now retracted; kept for the record, with a `superseded-by` edge |

Every one of the five is validated: a missing or misspelled `status` is a
check-0 finding, because an unrecognised status is not load-bearing and would
otherwise remove the node from scrutiny silently.

`supported` is the only status the validator treats as load-bearing. Marking
something `supported` is therefore a commitment, and checks 1, 2, 2b and 4 are
the cost of making it.

`withheld` is not a weaker `unsupported`: the evidence exists and is unread *by
design*, so as a premise of the thesis it is reported as a note rather than a
finding (check 3) - but it cannot carry a `supported` claim either (check 1).

`void` is enforced at both ends by check 2b: the void node must name its
replacement with a `superseded-by` edge, and nothing `supported` may still rest
on it.

### Evidence classes, in order

**`experimental` > `literature` > `ours`.**

- **`experimental`** - a deposited structure, a dataset, a measurement taken
  from one. Ground truth.
- **`literature`** - a published study, read rather than re-derived.
- **`ours`** - our own calculation, model or simulation.

This encodes an evidence hierarchy as *data* rather than as a rule someone has
to remember. The consequence is mechanical: **a claim whose only support is
`ours` is weakly held by construction**, and check 1 says so without anyone
needing to notice.

The class describes the *provenance of the number*, not who typed it. A
measurement your own script takes from a deposited structure is
`experimental`; an energy your own code computes on a geometry your own code
produced is `ours`. The distinction that matters: a calculation performed on a
geometry it was handed is evidence about the energy function, not about the
geometry.

---

## Edges - exactly four

| edge | meaning |
| --- | --- |
| `supports` | A is evidence or argument for B |
| `refutes` | A is evidence or argument against B |
| `requires` | A cannot stand unless B does - a premise, or an open question that gates it |
| `superseded-by` | A has been replaced by B; A is kept for the record |

**Four, and resisting a fifth is part of the design.** Link discipline is the
thing a long journal fails at; a vocabulary with a shade of meaning for every
occasion is a vocabulary nobody applies consistently, and an edge nobody
applies consistently cannot be checked.

Write an edge **on the node that makes the assertion**, in a section named
after the relation. The validator builds the reverse index, so inbound edges
are discoverable from either end, and mirror headings are parsed too so a
refutation can be visible on the node being refuted:

| outbound heading | mirror heading |
| --- | --- |
| `## Supports` | `## Supported by` |
| `## Refutes` | `## Refuted by` |
| `## Requires` | `## Required by` |
| `## Superseded by` | `## Supersedes` |

An outbound heading on note *N* produces `(N, rel, target)`; a mirror heading
produces `(target, rel, N)` - the same edge seen from the other end. Both forms
produce the same directed edge, and writing both ends is harmless - duplicates
collapse. A short level-2 heading that looks like a misspelled edge heading
(`## Supporting`) is reported at check 0, because an unrecognised heading
silently discards every link beneath it. Prefer the outbound form, and add a mirror where a reader
of *that* node needs to see the edge: a refuted claim should carry its own
`## Refuted by`.

---

## The validator

`argmap.py`, standard library only, no dependencies:

```
python argmap.py [directory] [--json] [--strict]    # default: .
```

It discovers the thesis from frontmatter rather than from a filename, so it
works on any map. `--json` writes the whole report to stdout as JSON and
nothing else, every finding as `{node, problem, fix}` under its check, for a
tool or an agent that acts on findings rather than reading them. `--strict`
counts the advisory notes as findings so they affect the exit status.

Exit status is 0 when nothing is flagged, 1 when something is, 2 on a usage or
structural error (not a directory, no notes, no thesis, an unknown option) - so
it can gate a commit.

### What it checks

0. **Note format** - every note parses; `type`, `status` and `evidence` are
   valid; no unknown frontmatter keys; no duplicate node ids; `entries` present
   where `require_sources` demands it; no misspelled edge headings.
0b. **More than one thesis** - a map has one root.
1. **Support beneath supported claims** - for each `supported` `claim` or
   `thesis`, inbound `supports` is followed **transitively**, and the check
   reports an empty support set, no observation anywhere beneath it, support
   that reaches only a `withheld` node, or every observation beneath it classed
   `ours`. The walk descends through an intermediate only when that
   intermediate is itself `supported`: `open`, `unsupported`, `void`,
   `withheld` and `question` nodes terminate the chain, because a chain through
   a node nobody holds carries nothing. A claim resting only on `literature` is
   printed as a note, not a finding - the middle rank is legitimate but weaker
   than a measurement.
2. **Supported claims with a standing refutation** - a node still marked
   `supported` that has an inbound `refutes` edge from a node that is itself
   `supported`. This is the check that catches a conclusion outliving its
   premise.
2b. **Retracted nodes, and live nodes resting on them** - a `void` node with no
   outbound `superseded-by` edge, and any `supported` node whose transitive
   support runs through a `void` node or which `requires` one.
3. **Unsupported premises of the thesis** - everything the thesis `requires`
   **transitively** (a premise of a premise is a premise) with an empty support
   set, no observation beneath it, or a `void` status. Premises that are
   `question` nodes, or `withheld`, are printed as notes.
4. **Citations of overturned sources** - nodes citing a journal entry that the
   project's retractions index lists as overturned, excluding entries listed in
   that node's `checked` field. Only `supported` nodes count as findings; the
   rest are printed as notes.
5. **Broken wikilinks** (5a) and **nodes with no path to the thesis** (5b).
   A broken edge is reported in the edge's own direction, so a mirror heading
   is not reported backwards, and it names both the missing id and the file the
   link was written in. Path (5b) is undirected connectivity over all four edge
   types, so a node is "connected" if it participates in the argument at all.

### Per-project configuration

Optional `argmap.json` beside the notes. Every key is optional, and every
check that depends on one **degrades to `skipped`** rather than failing, so the
validator runs on a project with no journal at all.

```json
{
  "journal": "../../csr-decision-log.md",
  "retracted_sources": "../retractions.md",
  "retracted_sources_ignore_headings": ["already marked"],
  "source_id_pattern": "D\\d+",
  "source_field": "entries",
  "acknowledged_field": "checked",
  "require_sources": "auto"
}
```

| key | default | meaning |
| --- | --- | --- |
| `journal` | none | printed in the header, so a reader knows where the history is |
| `retracted_sources` | none | file listing overturned sources; without it, check 4 is skipped |
| `retracted_sources_ignore_headings` | `["already marked"]` | sections of that file to skip, matched as lowercased substrings |
| `source_id_pattern` | `[A-Za-z]+\d+` | regex for a source id, e.g. `D\d+`, or a date pattern for a dated notebook |
| `source_field` | `entries` | frontmatter key holding citations |
| `acknowledged_field` | `checked` | frontmatter key holding read-the-correction acknowledgements |
| `require_sources` | `auto` | which nodes check 0 requires a non-empty `source_field` on: `all`, `supported`, `none`, or `auto` (= `supported` when `journal` is set, else `none`) |

The retractions reader is format-generic: it takes ids from the first cell of
any table row and from any `- **ID**` bullet, which is how such an index is
normally written.

---

## Scope

**Build the premise tree, not the archive.** A map is for the claims that bear
on whether the thesis is true. Operational history - job ids, config bugs,
queue rates, resume wiring - belongs in the journal; it has no argumentative
role, and converting it adds noise in proportion to effort.

Thirty nodes that are all load-bearing is a tool. Two hundred nodes of mixed
provenance is a second journal with worse search.
