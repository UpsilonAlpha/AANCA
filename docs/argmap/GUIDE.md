# argmap - the whole system in one file

A reusable tool for recording **what a project believes and why**, as a
directory of markdown notes with typed links and a validator that checks them.
This file is the self-contained description: the problem, the format, every
field, every check, the configuration, and how to adopt it elsewhere. `SPEC.md`
is the terse format reference, `AGENT.md` is the operating procedure for
whoever edits a map, and `../csr/claims/` is a worked example.

## The problem, and the design principle

A project that keeps an append-only journal - a decision log, a lab notebook, a
changelog - can find anything in it and still get the wrong answer. The failure
is **linkage**, not search: a correction lives inside a *later* entry, so a grep
lands on the retracted half, and a conclusion keeps being quoted after the
premise it rested on has been measured away.

argmap fixes that structurally by **attaching the refutation to the thing it
refutes**, as an edge, so one command lists every place a standing refutation is
being ignored. The design principle is one line:

> **The graph holds what we believe; the journal holds what happened.**

The journal is never migrated and never edited to match the graph - its
chronology is itself evidence (when a prediction was timestamped, how long a
contradiction went unnoticed). Instead every graph node *cites* the journal
entries it came from.

## A node file

One file per node, kebab-case name, `.md`. **The filename stem is the node's
id** and the target of every wikilink to it. Files whose stem is `readme`,
`spec`, `agent`, `guide` or `index` (any case) are skipped, so documentation can
live beside the notes.

```markdown
---
type: claim
status: supported
entries: [D108, D208]
checked: [D145]
---

# A one-line statement of the claim, as the heading

Why it is held, with the numbers and the entry each number came from.

## Supports
- [[some-other-node]]

## Refuted by
- [[a-measurement-that-contradicts-it]]
```

### Frontmatter, field by field

Five keys, and **no others** - an unrecognised key is reported, because a
misspelled key reads as a missing field.

| key | required | meaning |
| --- | --- | --- |
| `type` | yes | `thesis`, `claim`, `observation` or `question`. An invalid value **excludes the note from the graph entirely** - it contributes no edges and no reachability - and is reported. |
| `status` | yes | one of the five below. Missing or misspelled is reported, because an unrecognised status is not load-bearing and would quietly remove the node from checks 1, 2, 2b, 3b and 4. |
| `evidence` | observations only | `experimental`, `literature` or `ours`. Required on an `observation`; reported if present on anything else. |
| `entries` | see `require_sources` | the journal entries this node came from. Key name is configurable. |
| `checked` | optional | journal entries the retractions index lists as overturned whose correction has been read. Key name is configurable. See check 4. |

The parser accepts `key: scalar`, `key: [a, b]`, a bare `key:` followed by
`- item` lines, and `#` comments. Anything else - nesting, a stray line, a
duplicate key, unclosed frontmatter - is **reported**, not skipped.

### The four node types

- **`thesis`** - the root. Exactly one per map; found by frontmatter, not by
  filename. It must be an assertion, not a question: nothing can support or
  refute a question, and a map rooted in one has no shape.
- **`claim`** - an assertion the project makes, held on the strength of its
  inbound `supports` edges rather than its own authority.
- **`observation`** - a datum: a measurement, a published result, a number from
  a named calculation. It carries an evidence class and is **the only type that
  may be cited as evidence**; the checks look for observations beneath a claim.
- **`question`** - something open that bears on the argument. It asserts
  nothing, so it never passes support on. Link it with `requires` from the
  claim it gates, so a gap is visible rather than absent.

### The five statuses

| value | meaning to the validator |
| --- | --- |
| `supported` | the project holds this. **The only load-bearing status**: checks 1, 2, 2b, 3b and 4 examine these nodes, which is the cost of the commitment. |
| `open` | stated, not settled. Not load-bearing, and does not pass support through to a claim above it. |
| `unsupported` | an assertion with no traceable support, marked as one. |
| `withheld` | evidence exists and is **deliberately unread** (a blind prediction, a sealed dataset). Treated apart from missing evidence: as a thesis premise it is a note, not a finding, but it cannot carry a `supported` claim. |
| `void` | held once, now retracted, kept for the record. Must carry a `superseded-by` edge, and nothing live may rest on it (check 2b). |

### The three evidence classes

**`experimental` > `literature` > `ours`.** `experimental` is a deposited
structure or a measurement from one; `literature` is a published study, read
rather than re-derived; `ours` is our own calculation, model or simulation.

The class describes the **provenance of the number, not who typed it**. A
measurement your own script takes from a deposited structure is `experimental`;
an energy your own code computes on a geometry your own code produced is
`ours`. The hierarchy is therefore data rather than a rule to remember: a claim
whose only support is `ours` is weakly held by construction, and check 1 says so
without anyone needing to notice.

## Edges - exactly four

`supports` (A is evidence or argument for B), `refutes` (against B), `requires`
(A cannot stand unless B does - a premise, or an open question gating it), and
`superseded-by` (A has been replaced by B and is kept for the record). Resisting
a fifth is part of the design: an edge nobody applies consistently cannot be
checked.

**Mechanism.** An edge is written as a body section on the node that *makes* the
assertion. The parser walks the body line by line, skipping fenced code
blocks; a line beginning with `#` sets the current section from its lowercased
text with any trailing colon stripped; any following list item - a line
beginning `- `, `* `, `+ `, or a numbered `1.` / `1)` - has its `[[wikilinks]]`
read as edge targets. Wikilinks in prose under an edge heading are prose, not
edges; an edge section with wikilinks but no list item at all is reported. `[[target|alias]]` and `[[target#heading]]` both resolve
to `target`. Nine headings are recognised, giving one directed edge set:

| outbound heading | mirror heading | edge produced |
| --- | --- | --- |
| `## Supports` | `## Supported by` | `supports` |
| `## Refutes` | `## Refuted by` | `refutes` |
| `## Requires` | `## Required by` | `requires` |
| `## Superseded by` (or `## Superseded-by`) | `## Supersedes` | `superseded-by` |

An outbound heading on note *N* produces `(N, rel, target)`; a **mirror**
heading produces `(target, rel, N)` - the same edge seen from the other end.
Writing both ends is harmless, since duplicates collapse. Prefer the outbound
form and add a mirror where a reader of *that* node needs to see the edge; a
refuted claim should carry its own `## Refuted by`. A short level-2 heading that
looks like a misspelled edge heading (`## Supporting`) is reported, because an
unrecognised heading silently discards every link beneath it.

## The checks

Numbered as the output numbers them.

- **0 - Note format.** Frontmatter parses (a UTF-8 byte-order mark is
  tolerated); `type`, `status` and `evidence` are valid; no unknown frontmatter
  keys; no two node ids differing only in case (they are one note on Windows,
  macOS and most wikilink editors); `entries` present where `require_sources`
  demands it; no misspelled edge headings, no edge section whose links are all
  off list items, no unclosed code fence.
- **0b - More than one thesis.** A map has one root.
- **1 - Support beneath supported claims.** For each `supported` `claim` or
  `thesis`, follow inbound `supports` **transitively** and report: an empty
  support set; no observation anywhere beneath it; support that reaches only a
  `withheld` node; or every observation beneath it classed `ours`. The walk
  descends through an intermediate only when that intermediate is itself
  `supported` - `open`, `unsupported`, `void`, `withheld` and `question` nodes
  terminate the chain, because a chain through a node nobody holds carries
  nothing. A claim resting only on `literature` is printed as a note, not a
  finding: the middle rank is legitimate but weaker than a measurement.
- **2 - Supported claims with a standing refutation.** A `supported` node with
  an inbound `refutes` edge from a node that is itself `supported`. This is the
  check that catches a conclusion outliving its premise.
- **2b - Retracted nodes, and live nodes resting on them.** A `void` node with
  no outbound `superseded-by` edge, and any `supported` node whose transitive
  support runs through a `void` node or which `requires` one.
- **3 - Unsupported premises of the thesis.** Everything the thesis `requires`
  **transitively** (a premise of a premise is a premise) with an empty support
  set, no observation beneath it, or a `void` status. Premises that are
  `question` nodes, or `withheld`, are printed as notes: a question is not
  expected to carry support, and withheld evidence is unread by design.
- **3b - Supported nodes resting on premises that are not held.** `requires`
  propagates from every `supported` node, not only the thesis. Anything a
  supported claim or observation requires, transitively, that is not itself
  `supported` is a finding (a direct `requires` of a `void` node is check
  2b's); `question` and `withheld` premises are notes. This is what lets a
  defect in an input propagate: an observation computed on a structure the
  project built `requires` that structure's node, so demoting the structure
  flags every result computed on it.
- **4 - Citations of overturned sources.** Nodes citing a source the project's
  retractions index lists as overturned, minus the ids in that node's
  acknowledged field. `supported` nodes are findings; the rest are notes. An
  acknowledged id the node does not cite at all is also a note - stale if the
  citation was removed, fine if the body only discusses the entry.
- **5a - Broken wikilinks.** An edge whose missing end does not exist, reported
  as the edge's own direction (so a mirror heading is not reported backwards),
  naming both the missing id and the file the link was written in. A target
  rejected at check 0 for an invalid `type` says so instead of "no such note".
- **5b - Nodes with no path to the thesis.** Undirected reachability over all
  four edge types, so a node counts as connected if it participates in the
  argument at all.

## Configuration

Optional `argmap.json` beside the notes. Every key is optional; an unknown key
is warned about, and **every check that depends on one degrades to `skipped`**
rather than failing, so the validator runs on a project with no journal at all.

| key | default | meaning |
| --- | --- | --- |
| `journal` | none | free-text pointer printed in the header, so a reader knows where the history is. Also switches `require_sources: "auto"` on. |
| `retracted_sources` | none | path (relative to the notes) of a file listing overturned sources. Without it, or if it is missing, check 4 is skipped. |
| `retracted_sources_ignore_headings` | `["already marked"]` | sections of that file to skip, matched as lowercased substrings of the heading. |
| `source_id_pattern` | `[A-Za-z]+\d+` | regex for a source id - e.g. `D\\d+` for a decision log, or a date pattern for a dated notebook. |
| `source_field` | `entries` | frontmatter key holding the citations. |
| `acknowledged_field` | `checked` | frontmatter key holding read-the-correction acknowledgements. Listing an id there is an assertion: "I read the correction." |
| `require_sources` | `auto` | which nodes check 0 requires a non-empty `source_field` on: `all`, `supported`, `none`, or `auto` (= `supported` when `journal` is set, else `none`). |

The retractions reader is format-generic: it takes ids matching
`source_id_pattern` from the first cell of any table row and from any
`- **ID**` bullet, which is how such an index is normally written.

## Running it

```
python argmap.py [directory] [--json] [--strict]     # default: .
```

Standard library only, no dependencies, no network. `--json` writes the whole
report to stdout as JSON and nothing else - every finding as
`{node, problem, fix}` under its check - for a tool or an agent that wants to
act on findings rather than read them. `--strict` counts the advisory notes as
findings so they affect the exit status. `--help` prints the summary.

**Exit status: 0** when nothing is flagged, **1** when something is, **2** on a
usage or structural error (not a directory, no notes found, no `type: thesis`,
an unknown option, more than one directory argument). So it can gate a commit.
In `--json` mode a structural error still writes `{"error": ..., "exit": 2}` to
stdout, so a consumer never parses an empty stream.

`python test_argmap.py` runs the validator's fixture suite (stdlib
`unittest`, one tiny map per test in a temporary directory).

A project can ship a two-line shim (`check.py`) that points the validator at its
notes directory and passes options through, so `python check.py` works from
inside the map.

## Adopting it for another project

**Generic - copy unchanged:** `argmap.py`, `test_argmap.py`, `SPEC.md`,
`AGENT.md` and this file.
The vocabulary is generic too: the four node types, five statuses, three
evidence classes, four edge types and nine headings are hard-coded on purpose,
because a per-project vocabulary is a vocabulary nobody applies consistently.

**Project-specific - write these:** the notes themselves; `argmap.json` naming
your journal, your retractions index, your source-id pattern and your two
frontmatter field names; optionally a `check.py` shim and a `README.md`
describing what *this* map argues.

To start: get the thesis as an **assertion** (if the project's framing is a
question, it cannot be mapped), write the thesis node first with its `requires`
edges, then work downward from "what would have to be true" rather than forward
through the journal, which only produces a chronology with node headings. Target
tens of nodes, not hundreds - thirty load-bearing nodes is a tool; two hundred of
mixed provenance is a second journal with worse search.
