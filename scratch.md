Anchor case used throughout: **Qwen3-MoE / EP32** → L=94, E=128, D=32, R=32, P=160, S=5, top-k=8. Other competition cases follow the same structure with different (L, E, D, R) — see "OTHER CASES" at the bottom.

=========================== THE MODEL (whole picture) ===========================

  input tokens
       |
       v
   [embedding]
       |
       v
  +---------+    +---------+         +---------+
  | block 1 | -> | block 2 | -> ... -| block N |    N = total decoder blocks
  +---------+    +---------+         +---------+
       |                                  |
       v                                  v
                                       logits

  Each block = attention + FFN. Some FFNs are dense, some are MoE; only the
  MoE FFNs are what we balance. The simulator exposes only the MoE layers,
  so the L in `hotness.shape = (T, L, E)` is the count of MoE layers:
      L = 58  for DS-R1   (58 of 61 blocks are MoE; the first 3 are dense)
      L = 94  for Qwen3   (all 94 blocks are MoE)


==================== INSIDE ONE MoE LAYER ====================

  For each token x in the batch:

          x
          |
          v
      +--------+
      | router |   linear -> softmax -> top-k = 8
      +--------+
          |
          v
       picks 8 of {e0, e1, ..., e127}             (E = 128 routed experts in Qwen3)
          |
      +---+---+---+---+---+---+---+---+           <-- 8 parallel expert paths
      v       v       v     ...        v
    [e_i1]  [e_i2]  [e_i3]  ...     [e_i8]        each expert = SwiGLU MLP,
      |       |       |     ...        |          ~88 MB of BF16 weights
      +---+---+---+---+---+---+---+---+
          |
          v
    weighted sum (using the router's top-k softmax weights) -> layer output

  Routed vs shared: the router only selects from E "routed" experts; some
  models additionally have a "shared" expert per MoE layer that every token
  uses unconditionally, outside the router. Shared experts do NOT appear in
  hotness and are not load-balanced (they're replicated on every device).
  Per-model attribution is in PER-MODEL SPEC at the bottom.


===================== PHYSICAL PLACEMENT (EP across D=32 GPUs) =====================

  D is GLOBAL: the same D=32 physical GPUs host every MoE layer. Each layer
  chooses its own (D, S=5) expert-to-slot assignment, sharing the same devices.
  Per-GPU expert storage = L * S * 88 MB  ≈ 94 * 5 * 88 MB ≈ 41 GB just for
  experts. That is exactly why `deployment_table` has shape (L, D, S) — one
  independent placement per layer, on a fixed device set.

  ONE layer of the deployment table (shape D=32 × S=5; only a few devices shown):

              Device 0         Device 1    ...    Device 31
            +-----------+    +-----------+      +-----------+
   slot 0: |    e0     |    |    e5     |  ... |   e125    |
   slot 1: |    e1     |    |    e6     |  ... |   e126    |
   slot 2: |    e2     |    |    e7     |  ... |   e127    |
   slot 3: |    e3     |    |    e8     |  ... |   e64 *   |    * = redundant slot
   slot 4: |   e17 *   |    |    e9     |  ... |   e17 *   |      placed by the algo
            +-----------+    +-----------+      +-----------+      on a hot expert.

              total D * S = P = 160 physical slots; each holds one expert copy.

  logcnt[e] = number of physical copies of logical expert e in this layer.
  sum_e logcnt[e] = P = 160; logcnt[e] ≥ 1 (required: every expert appears).
  With R=32 extras spread across 128 experts the algorithm gives roughly 1/4
  of the experts a second copy — but greedy allocation concentrates the
  extras on the hottest experts, so a few may end up with 3-4 copies and
  most stay at 1.


====================== ONE ITERATION OF TOKEN FLOW (per MoE layer) ======================

  T0     The batch is sharded across the D=32 devices (DP). Each device holds
         its own token slice and a copy of the router weights.

  T1     Router picks top-k = 8 expert ids for every token.

  T2     ALL-TO-ALL (dispatch): each (token, picked-expert) pair is sent to a
         device that hosts a copy of that expert. When logcnt[e] > 1 the
         framework load-balances across the copies.

  T3     Each device runs its 5 experts on whatever tokens landed there.
            Dev 0  load ##########         <-- bottleneck of this phase
            Dev 1  load ####
              ...
            Dev 31 load ######
                            ^ wall-clock of expert compute = max over devices

  T4     ALL-TO-ALL (combine): expert outputs return to each token's home device.
  T5     Weighted sum with the router's softmax weights -> layer output.

         Repeat T2-T5 for every MoE layer (94 of them in Qwen3), every iteration.


============================ PAR — WHAT WE'RE MINIMIZING ============================

  Per-layer, per-iter formula (from quickstart.py:43):

      cut[e]      = logcnt[e]                       # copies of expert e
      weights[e]  = hotness[e] / cut[e]             # amortized load per copy
      load[d]     = sum over slots on device d of weights[ slot's expert ]
      PAR         = max_d load[d] / mean_d load[d]

      PAR = 1.0   perfectly balanced; no GPU idles.
      PAR = 1.5   the slowest GPU runs 50% over the mean -> 50% wasted compute.

  Scoring uses mean_PAR = average of per-(iter, layer) PAR over the trace.
  Modeled compute time = 60.0 s * mean_PAR   (BALANCED_COMPUTE_SECONDS constant).


====================== THE COLLECTION / REBALANCE LOOP ======================

  iter ->  1   2   3 ...  T=1024              1025      1026 ...
           |   |   |       |                    |         |
           v   v   v       v                    v         v
        run inference normally                 call rebalance(hotness[1..1024], D, R)
                                                 |
        meanwhile the trace records              v
        hotness[t, l, e] = # tokens          new deployment_table proposed
        routed to expert e at layer l         |
        during iter t                         v
                                         migrate exactly ONE layer per subsequent
                                         iter, in the order given by layers_priority.
                                         transmit_amount = sum over migrated layers
                                         of |{(d, s) : new ≠ old}|.


====================== WHAT YOUR rebalance() RETURNS ======================

  change           : bool      -- True => use the new deployment_table.
  layers_priority  : (L,) int  -- layer migration order; one layer per iter.
  deployment_table : (L, D, S) -- the new logical-expert-to-slot map.
  aux              : None      -- ignored.

  Scoring formula (from quickstart.py):

      modeled_time     = compute_time + transfer_time
      compute_time     = 60.0 * mean_PAR                           (s)
      transfer_time    = transmit_amount * 88_080_384 / 9e11       (s)
                                           ^^^^^^^^^^   ^^^
                                           bytes/expert  bytes/sec

      score            = 100 * baseline_modeled_time / your_modeled_time

  Where the constants come from (fixed, not topology-aware):

      88_080_384 = 3 * 7168 * 2048 * 2 bytes
                   = SwiGLU MLP (up, gate, down) at (7168, 2048) in BF16
                   = one DeepSeek-style expert's weight tensor
      9e11       = 900 GB/s, H100-class single-pipe abstraction
      transmit_amount = number of (layer, device, slot) positions whose
                        stored expert id changes during redeployment,
                        summed over all migration events in the trace
                        (see compute_redeploy_cost in quickstart.py:62).

  Useful unit: one changed slot ≈ 88e6 / 9e11 ≈ 98 microseconds modeled.
  So ~1 million slot changes ≈ 98 s, comparable to the 60 s compute budget.

  Tradeoff:
      lower PAR  (replicate hot experts, spread them across devices)
  vs. lower transmit (don't move slots you don't have to)


============================ DS-EPLB (the baseline) ============================

  DS-EPLB is what your score is measured against (the "100" anchor). It is the
  deepseek-ai/EPLB algorithm in its non-hierarchical / "global" mode (used by
  the simulator with num_groups=1, num_nodes=1). The full code is mirrored in
  submissions/my_sub/submission.py. Mechanically, every rebalance call does:

  ── Step 0. Collapse the time window ───────────────────────────────────────
     w[l, e] = sum_{t=0..T-1} hotness[t, l, e]
     One scalar load per (layer, expert). No variance, no recency weighting.
     All L layers are then handled independently and identically.

  ── Step 1. Greedy REPLICATION  (_replicate_experts, per layer) ───────────
     Start: each logical expert has 1 copy.   logcnt[e] = 1 for all e.
     Repeat (R times, where R = n_red_expert):
        e*  = argmax_e  w[e] / logcnt[e]              ← amortized load
        logcnt[e*] += 1                                ← add one physical copy
     After R steps, sum_e logcnt[e] = E + R = P.
     This greedily minimizes the max amortized load — i.e., LPT (longest
     processing time first) applied to "shrink the bottleneck expert".

  ── Step 2. Balanced PACKING  (_balanced_packing, per layer) ──────────────
     Now you have P physical slots, each with amortized load w[e]/logcnt[e].
     Distribute them to D devices, exactly S = P/D slots per device.
     LPT-greedy:
        sort the P slots by amortized load, descending
        for each slot in order:
            assign to whichever device has the smallest current load total
            AND still has free capacity (< S slots assigned)
     This greedily minimizes max_d load[d] given the replication from Step 1.

  ── Step 3. Return ─────────────────────────────────────────────────────────
     deployment_table[l, d, s]  = logical-expert id at (device d, slot s)
     change            = True            (always; baseline ignores transit cost)
     layers_priority   = [0, 1, ..., L-1]  (all layers, in natural order)

  Properties / limitations of DS-EPLB:

    • Pure PAR optimizer: minimizes peak compute load only; pays no attention
      to transmit_amount. It always proposes a full replacement, then the
      simulator pays D * S * (# changed slots) to migrate. Beating this score
      mostly comes from cutting transit, since DS-EPLB's PAR is already low.

    • Memoryless: uses only the most recent T-window sum. If routing patterns
      drift slowly, recomputing from scratch each time discards useful prior
      structure (and so causes unnecessary migrations).

    • Per-layer independent: same algorithm, L times. There is no joint
      reasoning across layers (and there cannot be — the experts are
      distinct weight tensors).

    • Stateless w.r.t. the current placement: it never looks at where
      experts are now. Identical inputs → identical outputs regardless of
      what the prior table looked like. This is the principal source of
      avoidable transit cost.

  Where to improve (qualitative):
      a) Return change=False when expected PAR gain < transit cost.
      b) When you do redeploy, bias the new table to be close to the
         current one (re-use the existing device assignment of cold
         experts; only reshuffle around the hot tail).
      c) Trim layers_priority to the layers that actually benefit, instead
         of migrating all L of them.
      d) Use a load estimate richer than a plain sum (EMA, robust mean, …)
         to filter transient spikes that don't justify migration.


============================== PER-MODEL SPEC ==============================

   The only target models are DS-R1 and Qwen3 (see MODEL_SHAPES in
   quickstart.py). Per MoE LAYER:

   Model    L     routed E   shared exp   top-k routed   active/tok   EP sizes
   --------------------------------------------------------------------------
   DS-R1    58    256        1            8              9            {32,64,128,256}
   Qwen3    94    128        0            8              8            {32,64,128}

   "routed E" is exactly the E in hotness.shape = (T, L, E). The shared expert
   (DS-R1 only) is held on every device, always used, never load-balanced, and
   not counted in the score model.

   Simulator uses R = D for all cases. Worked (P, S) = (E+R, P/D):

       Qwen3   EP32  -> P = 160, S = 5     (anchor case above)
       Qwen3   EP64  -> P = 192, S = 3
       Qwen3   EP128 -> P = 256, S = 2
       DS-R1   EP32  -> P = 288, S = 9
       DS-R1   EP64  -> P = 320, S = 5
       DS-R1   EP128 -> P = 384, S = 3
       DS-R1   EP256 -> P = 512, S = 2

   The competition is essentially: at each rebalance opportunity, pick a
   per-layer (D, S) tiling of expert copies that (a) gives every logical
   expert ≥1 copy, (b) packs hot experts onto more slots, (c) spreads
   those slots evenly across devices, and (d) differs minimally from the
   current placement.
