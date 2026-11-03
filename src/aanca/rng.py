"""Random draws that survive being on a GPU.

**This module exists because of job 31516313**, the first inverse-folding run,
which took a GPU allocation and died at update 0 with::

    RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

in ``torch.multinomial(flat, 1, generator=generator)``. It is a device-only
failure and **the local test suite structurally cannot see it**: every test
runs on CPU, where the generator and the tensor are both CPU and the
constraint is satisfied for free. 806 passing tests said nothing about it.

The constraint is narrow and easy to miss. A ``torch.Generator`` carries a
device, and any draw that *consumes* a generator requires the generator's
device to match the tensor it is drawing from or into. That interacts badly
with this project's reproducibility convention, stated at
:func:`rollout.update_mask`: every random choice is drawn on the **CPU** and
moved, so that a run's random stream is a property of its seed and not of the
hardware it lands on. The two rules are in direct tension, and the resolution
has to be explicit rather than incidental -- draw on the generator's device,
then move the *result*. The draw stays reproducible and the caller gets its
answer where it wants it.

Three call sites needed it: :func:`identity.straight_through`'s multinomial,
:func:`identity._gumbel_noise`'s uniform (which passed ``device=logits.device``
and so would have failed identically), and :func:`potts._sample`. The first is
the one that crashed; the other two had never been reached on a GPU.

``scripts/device_check.py`` is the systemic half of the fix: it exercises every
one of these paths on whatever device it is given, so the GPU check job answers
"does a training step run on this card" rather than only "does torch see it".
"""

from __future__ import annotations

import torch
from torch import Tensor


def generator_device(generator: torch.Generator | None) -> torch.device | None:
    """Where ``generator`` draws, or ``None`` when there is no generator.

    ``None`` means "use the ambient default generator", which torch resolves
    per device, so no move is needed in that case.
    """
    return None if generator is None else generator.device


def draw_multinomial(
    probabilities: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """One index per row of ``probabilities``, on ``probabilities``' own device.

    Shapes: ``(..., C)`` in, ``(...)`` out. The draw happens wherever the
    generator lives and the index is moved back, so a CPU generator keeps a
    GPU run's stream reproducible instead of raising.

    ``detach`` before the move because the choice is discrete and carries no
    gradient; the gradient path in
    :func:`identity.straight_through` goes through the softmax, not through
    the index.
    """
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    home = flat.device
    where = generator_device(generator)
    if where is not None and where != home:
        flat = flat.detach().to(where)
    drawn = torch.multinomial(flat, 1, generator=generator)
    return drawn.reshape(*probabilities.shape[:-1]).to(home)


def draw_uniform(
    shape: tuple[int, ...] | torch.Size,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Uniform ``[0, 1)`` of ``shape``, drawn on the generator and returned on ``device``."""
    where = generator_device(generator)
    target = torch.device(device)
    origin = target if where is None else where
    values = torch.rand(shape, generator=generator, dtype=dtype, device=origin)
    return values if origin == target else values.to(target)
