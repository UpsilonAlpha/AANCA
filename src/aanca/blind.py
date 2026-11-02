"""Backbone-and-CB perception, which is what makes inverse folding a real task.

Predicting a residue's identity from a structure that still contains its side
chain is not a task: the atom set alone gives the answer. ProteinMPNN's input
is therefore backbone plus a virtual CB, and this module is that input.

**Hiding it at the feature level is not enough, and that is the trap.** Zeroing
the chi channels and the side-chain edge block leaves two channels open that no
feature mentions:

- **Neighbour selection.** ``torch_graph`` picks the k nearest by *minimum
  heavy-atom* distance, so which residues are even in the graph depends on how
  far everyone's side chain reaches.
- **The envelope.** Also a minimum heavy-atom distance, and it multiplies every
  message.

A rule could read "my side chain is long" off either and narrow its own
identity to TRP or ARG without a single side-chain feature being present. So
the truncation happens to the *coordinates*: every atom beyond CB is moved onto
its residue's CB. Distances involving a side-chain atom then equal the CB
distance, minima over them are minima over backbone-and-CB, and the side-chain
summary collapses to CB for every residue of every type. Duplicate points do
not change a minimum, so the atom count leaks nothing either.

What remains is exactly one identity channel, and it is the one we hand over
deliberately: the type one-hot in ``node_features``, which carries the
*currently committed* label rather than the native one. That is the ripple's
pathway and nothing else is.

``tests/test_blind.py`` asserts the whole claim in one line -- randomise every
chi and the perception must not change by one bit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .torch_graph import FeatureTopology, frame_atom_coords

#: Everything a blind rule may see of a residue. OXT is the C-terminal oxygen,
#: which is backbone; CB is the first side-chain atom and the one ProteinMPNN
#: keeps, real or constructed.
VISIBLE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "OXT", "CB"})


def beyond_cb_mask(topo: FeatureTopology) -> Tensor:
    """``(n_res, n_atoms)`` bool: atoms to be collapsed onto CB.

    Depends on the sequence and not on the coordinates, so compute it once per
    structure and pass it down a rollout rather than rebuilding it per step.
    """
    names = topo.torsion.atom_names
    width = topo.atom_mask.shape[1]
    mask = torch.zeros(len(names), width, dtype=torch.bool)
    for r, residue in enumerate(names):
        for a, name in enumerate(residue):
            if a < width and name not in VISIBLE_ATOMS:
                mask[r, a] = True
    return mask.to(topo.atom_mask.device)


def blind_coords(
    coords: Tensor, topo: FeatureTopology, beyond: Tensor | None = None
) -> Tensor:
    """``coords`` with every atom beyond CB moved onto its residue's CB.

    CB comes from :func:`frame_atom_coords`, so a glycine gets the constructed
    virtual CB and the result is defined for every residue. The returned tensor
    is differentiable with respect to the backbone and carries no gradient to
    the side-chain positions it overwrote -- correct here, because a blind rule
    is not predicting them.
    """
    if beyond is None:
        beyond = beyond_cb_mask(topo)
    cb = frame_atom_coords(coords, topo)[:, 4]
    return torch.where(beyond.unsqueeze(-1), cb.unsqueeze(1), coords)
