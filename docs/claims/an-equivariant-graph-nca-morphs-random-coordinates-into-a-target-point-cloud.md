---
type: observation
status: supported
evidence: literature
sources: [Gala 2024]
---

# A one-hop equivariant graph NCA, 5K parameters, converges randomly initialised coordinates onto a target point cloud and holds it

Gala, Grattarola & Quaeghebeur, *E(n)-equivariant Graph Neural Cellular
Automata*, TMLR 2024 (arXiv 2301.10497), section 4.1, "pattern formation".

Published result, not our calculation. **This is the first `literature`
observation in this map**, which until now held only `experimental`
(depositions) and `ours`.

## What was done

Transition rule: a single E(n)-equivariant graph convolution per CA step.

    m_ij = phi_m(||x_i - x_j||^2, h_i, h_j, e_ij)
    m_i  = sum_{j in N(i)} m_ij
    x_i' = x_i + (1/|N(i)|) sum_{j in N(i)} (x_i - x_j) phi_x(m_ij)
    h_i' = phi_h(h_i, m_i)

phi_x maps to a scalar, so the coordinate shift is a mean of neighbour
difference vectors with learned scalar weights. Hidden 16, message 32, **5K
parameters total**. One hop per step; the authors attribute global reach to
iteration rather than depth -- *"the effective receptive field of E(n)-GNCA
localized message-passing grows larger with each state update until eventually
encompassing the whole graph"*. The state is one-step Markovian, with no step
index.

## The numbers

  * Initial coordinates drawn from an isotropic Gaussian; node features all 1.
  * **8 target geometric graphs, "we succeed with all of them."** The three
    reported: a 2D grid (256 nodes), a 3D torus (256 nodes), the Stanford bunny
    (2503 nodes).
  * Trained by BPTT over a rollout of **15 to 25 steps**, sampled uniformly,
    with a replay pool -- the Growing-NCA persistence trick.
  * *"Remarkably, our model learns to converge to a stable attractor of the
    given geometric graph, i.e. the model can maintain the target state after
    any number of time steps t > 15."* Damage applied at t=24; the rule is run
    1000 further steps after perturbation.
  * Regeneration is trained for: half of each batch is perturbed with Gaussian
    noise, part globally and part locally.
  * No anchors. Every node starts random.

The paper reports no scalar error values in text -- `L_INV` appears only inside
figure panels -- so **the result is qualitative convergence on 8 of 8 graphs,
not a quoted RMSD.** Recorded that way deliberately.

## What they also say about the conditions

  * The loss is all-pairs: `L_INV = mean over all (i,j) of
    (||x_i - x_j|| - ||xhat_i - xhat_j||)^2`, and this is *"necessary to
    uniquely identify the target state"*. Nodes "are not aware of their global
    location, and have |V|-1 constraints to satisfy".
  * Message passing for this task runs over the **fixed target graph**, not a
    graph recomputed from current positions. Their one dynamic-graph task
    (Boids) has no target shape.
  * No step-size cap; phi_x is unbounded. Training was stabilised with weight
    decay and gradient clipping, and they report exploding gradients.
  * Stated limitation: the 1-to-1 node correspondence makes the loss
    permutation-dependent and "leads to abrupt dynamics especially in the first
    time steps".

## Supports
- [[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
- [[question-can-the-architecture-place-residues-under-any-supervision]]
