"""Run configs: a named section per arm, so the settings are a file and not a shell line.

Modelled on ``mcube``'s layout (``config_examples/*.yaml`` and the comment above
``build_parser`` in ``src/mcube/cli.py``), and the rule is that comment's rule:

    **Nothing that changes a result is a command-line option.** Every setting
    that moves a number lives in a named config section, so the config alone
    says what was computed and the same command reproduces it. Options may only
    *address* the config (``--section``) or *name a resource* (``--device``,
    ``--max-seconds``).

The problem it fixes here is specific. Fifteen growth arms were launched as
twenty-token shell lines inside ``scripts/bunya_submit.sh`` invocations, and the
only record of what any of them was is the ``args :`` line in a Slurm log --
which is not diffable, not reviewable before submission, and not attached to the
checkpoint. Two arms that should differ in one setting could differ in two and
nothing would say so. g13, g14 and g15 are the comparison the project now rests
on, and the evidence that they were single-variable arms is three log headers.

Three deliberate choices, each with its reason:

**TOML by way of stdlib ``tomllib``, not YAML.** mcube uses YAML, and this is
the one place the layout is not copied. The package declares two dependencies
(``numpy``, ``biotite``) and adding ``pyyaml`` would have to hold on the laptop
and in ``.venv_bunya`` mid-experiment. ``tomllib`` is in the standard library
from 3.11, is read-only -- which is the right shape, since configs are written
by hand and read by the trainer -- and keeps comments, which is where mcube's
configs carry their reasoning.

**The base section states every key explicitly.** A config that leans on the
parser's defaults is a config whose meaning changes when a default changes, and
a recorded run would then be silently reinterpreted. ``check_base_is_complete``
makes that an error rather than a convention.

**A section states only its difference from the base.** That is what makes an
arm's claim to be single-variable readable: ``[g13_natedge.inputs]`` is one line
long, and no comparison rests on diffing two log headers by eye.
"""

from __future__ import annotations

import argparse
import difflib
import tomllib
from pathlib import Path
from typing import Any

#: Options that may stay on the command line, by the rule above. Everything
#: else must come from the config.
#:
#: ``resume`` and ``init_from`` are the awkward pair and they split. ``resume``
#: continues *this* run after a walltime, so it changes nothing about what the
#: run is -- a resource matter, and the restart chain in
#: ``scripts/bunya_submit.sh`` sets it without knowing anything about the
#: experiment. ``init_from`` starts a NEW run from another run's weights, which
#: is part of what the new run *is*, so it belongs in the section.
RUNTIME_ONLY = ("device", "max_seconds", "stop_file", "resume")

#: Keys a section may not set, because they are supplied some other way.
#: ``tag`` is the section name: a run directory that could disagree with the
#: settings that produced it is the thing this module exists to prevent.
DERIVED = ("tag",)


class ConfigError(Exception):
    """A config that cannot be used, reported with what to do about it."""


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """``dest -> action`` for every real option, the parser being the authority.

    The parser stays the single declaration of names, types and defaults, so a
    new flag is usable from a config the moment it is added and the two cannot
    drift. This is why the loader takes a parser rather than a schema of its
    own.
    """
    out = {}
    for action in parser._actions:  # noqa: SLF001 -- argparse exposes no public view
        if action.dest in ("help", argparse.SUPPRESS):
            continue
        out[action.dest] = action
    return out


def _expected_type(action: argparse.Action) -> type:
    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        return bool
    if action.type is not None:
        return action.type  # type: ignore[return-value]
    return str


def load(path: str | Path) -> dict[str, Any]:
    """The whole config file as a mapping of section name to its body."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"no config at {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from None
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"{path} holds no sections")
    return data


def sections(data: dict[str, Any]) -> list[str]:
    """Section names a run may be launched from, base excluded."""
    return sorted(name for name in data if name != "base")


def resolve(data: dict[str, Any], section: str) -> dict[str, Any]:
    """``base.inputs`` overlaid with the section's own, following ``inherits``.

    An ``inherits`` chain lets a variant say only what it changes about another
    arm rather than about the base -- mcube's ``metadynamics_lu``, which names
    an ion and a parameter set and nothing else. A cycle is an error rather
        than a silent stop, because the resolved value would otherwise depend
    on which link the walk happened to start from.
    """
    if section == "base":
        raise ConfigError(
            "base is the shared ground, not an arm: name a section to run. "
            f"Available: {', '.join(sections(data)) or '(none)'}"
        )
    if section not in data:
        known = sections(data)
        near = difflib.get_close_matches(section, known, n=1)
        hint = f" Did you mean {near[0]}?" if near else ""
        raise ConfigError(
            f"no section {section!r} in this config. "
            f"Available: {', '.join(known) or '(none)'}.{hint}"
        )

    chain: list[str] = []
    name: str | None = section
    while name is not None and name != "base":
        if name in chain:
            raise ConfigError(
                "inherits forms a cycle: " + " -> ".join([*chain, name])
            )
        if name not in data:
            raise ConfigError(
                f"section {chain[-1]!r} inherits {name!r}, which this config "
                "does not define"
            )
        chain.append(name)
        nxt = data[name].get("inherits")
        if nxt is not None and not isinstance(nxt, str):
            raise ConfigError(f"{name}.inherits must be a section name")
        name = nxt

    merged: dict[str, Any] = dict((data.get("base") or {}).get("inputs") or {})
    for link in reversed(chain):
        body = data[link]
        inputs = body.get("inputs")
        if inputs is None:
            raise ConfigError(f"section {link!r} has no [{link}.inputs] table")
        if not isinstance(inputs, dict):
            raise ConfigError(f"{link}.inputs must be a table")
        merged.update(inputs)
    return merged


def check_base_is_complete(
    data: dict[str, Any], parser: argparse.ArgumentParser
) -> None:
    """Raise unless ``base.inputs`` names every setting the trainer reads.

    So a recorded run cannot be reinterpreted by a later change to a parser
    default: whatever the defaults become, the config still says what the run
    was. The cost is one long base section per config file, which is the thing
    being bought rather than an overhead.
    """
    base = dict((data.get("base") or {}).get("inputs") or {})
    wanted = {
        dest
        for dest in _actions(parser)
        if dest not in RUNTIME_ONLY and dest not in DERIVED
    }
    missing = sorted(wanted - set(base))
    if missing:
        raise ConfigError(
            "[base.inputs] must state every setting explicitly so a change to "
            "a parser default cannot reinterpret a recorded run. Missing: "
            + ", ".join(missing)
        )


def to_namespace(
    parser: argparse.ArgumentParser,
    inputs: dict[str, Any],
    section: str,
    runtime: argparse.Namespace,
) -> argparse.Namespace:
    """Validate ``inputs`` against the parser and build the trainer's namespace.

    Unknown and mistyped keys are errors, not omissions. A key that silently
    does nothing is the same fault as a command-line option that is silently
    ignored -- which is the one mcube records as having bitten it, where ``md``
    read its structure from the config and ignored ``--pdb``, so a run
    equilibrated the wrong file and looked entirely normal.
    """
    actions = _actions(parser)
    args = argparse.Namespace()

    unknown = sorted(set(inputs) - set(actions))
    if unknown:
        lines = []
        for key in unknown:
            near = difflib.get_close_matches(key, list(actions), n=1)
            lines.append(key + (f" (did you mean {near[0]}?)" if near else ""))
        raise ConfigError(
            f"section {section!r} sets keys the trainer does not read: "
            + "; ".join(lines)
        )

    for key in sorted(set(inputs) & set(DERIVED)):
        raise ConfigError(
            f"section {section!r} sets {key!r}, which is derived: the tag is "
            "the section name, so a run directory cannot disagree with the "
            "settings that produced it"
        )

    for key in sorted(set(inputs) & set(RUNTIME_ONLY)):
        raise ConfigError(
            f"section {section!r} sets {key!r}, which names a resource rather "
            "than part of the experiment, so it stays on the command line"
        )

    for dest, action in actions.items():
        if dest in RUNTIME_ONLY:
            setattr(args, dest, getattr(runtime, dest))
            continue
        if dest in DERIVED:
            continue
        if dest not in inputs:
            setattr(args, dest, action.default)
            continue
        value = inputs[dest]
        want = _expected_type(action)
        if want is bool:
            if not isinstance(value, bool):
                raise ConfigError(
                    f"{section}.inputs.{dest} must be true or false, got "
                    f"{value!r}"
                )
        elif want is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(
                    f"{section}.inputs.{dest} must be a number, got {value!r}"
                )
            value = float(value)
        elif want is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(
                    f"{section}.inputs.{dest} must be an integer, got {value!r}"
                )
        elif want is str:
            if value is not None and not isinstance(value, str):
                raise ConfigError(
                    f"{section}.inputs.{dest} must be a string, got {value!r}"
                )
        if action.choices is not None and value not in action.choices:
            raise ConfigError(
                f"{section}.inputs.{dest} must be one of "
                f"{', '.join(map(str, action.choices))}, got {value!r}"
            )
        setattr(args, dest, value)

    args.tag = section
    return args


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if value is None:
        # TOML has no null. An unset path is the empty string on the way back
        # in, which is what the parser's own default already is for the one
        # key this reaches (`stop_file`); recorded as "" so the snapshot
        # round-trips rather than dropping the key.
        return '""'
    return repr(value)


def resume_drift(stored: dict, args, allow=("updates",)) -> dict:
    """Which settings the checkpoint disagrees with the config about.

    ``{key: (what the checkpoint says, what the config says)}``, empty when a
    resume is safe. Pure, so it is testable without a checkpoint and without a
    GPU -- the growth trainer went fifteen arms with no resume check at all,
    and a check that can only be exercised by a real two-segment run on a
    cluster is a check nobody exercises.

    ``updates`` is allowed to differ by default because extending a run is the
    legitimate reason to touch a section mid-chain. Runtime keys are never
    compared: ``max_seconds`` is derived from the allocation and differs on
    every segment by design.
    """
    return {
        key: (was, getattr(args, key))
        for key, was in sorted(stored.items())
        if key not in RUNTIME_ONLY
        and key not in allow
        and hasattr(args, key)
        and was != getattr(args, key)
    }


def snapshot(args: argparse.Namespace, section: str, source: str | Path) -> str:
    """The run's fully resolved settings as a TOML section, for the run directory.

    Written beside the checkpoint so a finished run carries its own settings
    rather than pointing at a config file that may have moved on. Fully
    resolved and explicit for the same reason ``check_base_is_complete``
    exists: the snapshot has to mean the same thing when read next year.
    """
    lines = [
        "# Resolved settings for one run. Written by the trainer; not an input.",
        f"# section {section} of {Path(source).name}",
        "#",
        "# **Readable back by the trainer that wrote it:**",
        "#",
        f"#     python scripts/train_growth.py <this file> --section {section}",
        "#",
        "# which is why the settings sit under [base.inputs] and the run's own",
        "# section is empty. The first version of this function emitted them",
        "# under [<section>.inputs] with no base: complete as a record and",
        "# unloadable as a config, because check_base_is_complete refuses it",
        "# before resolve ever runs. A snapshot you cannot feed back is a",
        "# transcript, not a configuration.",
        "",
        "[base.inputs]",
    ]
    for dest in sorted(vars(args)):
        if dest in RUNTIME_ONLY or dest in DERIVED:
            continue
        lines.append(f"{dest} = {_toml_value(getattr(args, dest))}")
    lines += [
        "",
        "# The run's own section. Empty because the base above is already fully",
        "# resolved; the tag is the section name, so this header is what makes",
        f"# --section {section} name this run and not another.",
        f"[{section}.inputs]",
    ]
    return "\n".join(lines) + "\n"
