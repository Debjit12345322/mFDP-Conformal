# ML Optimization Proxy for Economic Dispatch

Replacing a classical LP solver with a neural network that is **feasible by
construction** and **~1000x faster**, trained **without any labeled/solved
data** — plus a natural-language layer that lets you edit the problem itself
in plain English and re-optimize.

Based on the *optimization proxy* concept from Van Hentenryck (2025),
"Optimization Learning" (arXiv:2501.03443) — the same technique used at
industrial scale for real-time power grid risk assessment.

---

## The problem: Economic Dispatch

Given a set of generators with capacity limits and costs, and a target
electricity demand, decide how much each generator should produce so that:

- total generation = demand (exact),
- every generator stays within `[P_min, P_max]`,
- total cost is minimized.

This is solved thousands of times a day by grid operators (every 5 minutes,
for every scenario in a risk simulation), which makes solver speed a real
bottleneck at scale.

## The idea: learn the solver, but never let it be infeasible

A plain neural network trained to imitate the solver (regression) will
**violate constraints on unseen inputs** — it has no way to guarantee
`sum(dispatch) == demand`. This project instead uses a two-stage
architecture:

```
demand ──▶ [MLP] ──▶ raw dispatch (bounded per generator via sigmoid)
                          │
                          ▼
              [Repair Layer]  ← closed-form, differentiable rescaling
                          │
                          ▼
        feasible dispatch (sum EXACTLY equals demand, always)
```

The repair layer (`repair_power_balance` in `model.py`) is not a solver
call — it's a proportional-scaling formula, so gradients flow straight
through it. That means the whole pipeline can be trained **end-to-end**,
and the network only has to learn *how to be cheap*; feasibility is
guaranteed structurally, before training even starts.

### Self-supervised training

Because the objective function (`cost = Σ cᵢ·pᵢ`) is known in closed form,
no labeled data is needed. Training just minimizes expected cost directly
over randomly sampled demand values — the solver is *never called during
training*, only used afterward to check the results.

## Results (1000 held-out test instances)

| Metric | Value |
|---|---|
| Mean optimality gap vs. LP solver | **1.31%** |
| Median optimality gap | 0.28% |
| Max power-balance violation | 0.00006 MW (~0) |
| Max generator-bound violation | 0.00002 MW (~0) |
| Solver time (1000 instances) | 1519 ms |
| ML proxy time (1000 instances, one batch) | 1.5 ms |
| **Speedup** | **~1000x** |

![dispatch curve](plots/dispatch_curve.png)

The network learns merit-order dispatch (cheapest generators fill first)
purely from the cost signal — no one told it the generators' cost ranking.

![training curve](plots/training_curve.png)
![gap distribution](plots/gap_distribution.png)
![speed comparison](plots/speed_comparison.png)

### Why the repair layer matters (ablation)

A baseline MLP trained with a soft feasibility *penalty* (the standard
Lagrangian-style approach) still violates power balance by multiple MW on
many test instances — because penalties only discourage violations, they
don't prevent them.

![ablation](plots/ablation.png)

## Natural-language scenario editing

Grid operators don't reason in numpy arrays — they reason in scenarios: *what
if the peaker trips?*, *what if gas gets 30% more expensive?* The LLM layer
turns sentences like that into concrete edits of the optimization problem, and
re-solves.

```
"take the peaker offline and push demand to 500"
            │
            ▼
      [Gemini planner]  ← constrained to emit JSON patches only
            │
            ▼
  [{"op": "disable_generator", "generator": "G5_peaker", "value": null},
   {"op": "set_demand",        "generator": null,        "value": 500}]
            │
            ▼
      [Patch applier]  ← validates, clamps, checks feasibility
            │
            ▼
   patched system ──▶ LP solver ──▶ before/after dispatch + cost delta
```

**The LLM never computes anything.** It doesn't produce a dispatch, a cost, or
even a numpy array — it only maps English onto a fixed, closed vocabulary of
patch operations (`set_cost`, `scale_p_max`, `disable_generator`,
`set_demand`, …). Every patch is then validated in Python: unknown ops and
hallucinated generator names are dropped, values are type-checked, bounds are
clamped so `p_min` can never exceed `p_max`. The optimization itself stays
entirely classical and verifiable.

Example run:

```bash
$ python reopt.py "make coal 60% more expensive"

Patches from planner:
  scale_cost         G1_coal      1.6
  scale_cost         G2_coal      1.6

Resulting changes:
  G1_coal: cost 18.0 -> 28.8 $/MWh
  G2_coal: cost 22.0 -> 35.2 $/MWh

Feasible: demand 350.0 MW inside [120.0, 610.0] MW.

Generator        before      after      delta
----------------------------------------------
G1_coal          200.00     200.00      +0.00  .
G2_coal          120.00      40.00     -80.00
G3_gas            20.00     100.00     +80.00
G4_gas            10.00      10.00      +0.00  .
G5_peaker          0.00       0.00      +0.00  .
----------------------------------------------
TOTAL            350.00     350.00      +0.00

Cost before: $7,360.00
Cost after:  $11,088.00
Change:      $+3,728.00  (+50.65%)
```

That output shows a genuine **merit-order flip**, not a cosmetic rescale:
pushing G2_coal from 22 to 35.2 \$/MWh lifts it just past G3_gas at 35.0, so
the solver backs G2 down to its 40 MW floor and pulls G3 up in its place. The
edit really re-enters the optimization.

Infeasible requests are caught *before* the LP runs, so you get an explanation
instead of a solver traceback:

```bash
$ python reopt.py "shut down all the gas units and set demand to 500"
Demand 500.0 MW exceeds total capacity 350.0 MW — not enough generation available.
Stopping: no dispatch exists for this system.
```

### Current scope: solver only

The re-optimization path deliberately runs against the **LP solver, not the
proxy**. The proxy takes demand as its only input and was trained with the
nominal cost vector baked into its loss, so a cost or capacity patch never
reaches it — on the example above it would return the unchanged 200/120/20/10/0
dispatch, and the reported optimality gap would jump to ~12% with nothing
actually broken. Demand-only patches are in-distribution and fine; everything
else is not.

Making the proxy respond to edited systems means conditioning it on the system
itself — feeding `[demand, p_min…, p_max…, cost…]` and sampling perturbed
systems during training. The repair layer already handles arbitrary bounds, so
the feasibility guarantee survives unchanged; only the predictor needs to
grow. That's the natural next step for this repo.

## Project structure

```
problem.py     5-generator test system (costs, limits)
solver.py      Ground-truth LP solver (scipy.optimize.linprog)
data.py        Demand sampling for train/val/test (self-supervised: no labels)
model.py       MLP predictor + differentiable repair layer
train.py       Self-supervised training loop
evaluate.py    Optimality gap / feasibility / speed comparison vs. solver
ablation.py    Baseline without repair layer, to show why it's needed
plots.py       Generates all plots in plots/
app.py         Interactive Streamlit demo (slide demand, compare live)

llm_planner.py Natural language -> validated JSON patches (Gemini)
patcher.py     Applies patches to the system; bounds + feasibility checks
reopt.py       End-to-end: prompt -> patches -> patched system -> re-solve
```

`solver.solve_ed` takes an optional `system=` argument so it can be run
against a patched system; passing nothing keeps the original behaviour, so
`evaluate.py`, `ablation.py` and `plots.py` are unaffected.

## Running it

```bash
pip install -r requirements.txt

python problem.py     # inspect the test system
python solver.py      # sanity check the LP baseline
python train.py        # train the proxy (saves proxy_model.pt)
python evaluate.py     # full comparison report
python ablation.py     # feasibility ablation
python plots.py        # regenerate all plots

streamlit run app.py   # interactive demo
```

For the natural-language layer, set a Gemini API key first:

```bash
export GEMINI_API_KEY=...

python reopt.py "take the peaker offline and push demand to 500"
python reopt.py "double G3's capacity" --demand 420
```

## What this demonstrates

- Differentiable "repair layers" for hard-constraint satisfaction in
  neural networks (relevant beyond power systems — routing, scheduling,
  resource allocation, anywhere with linear/simple constraint structure).
- Self-supervised training directly against a known objective function,
  avoiding the cost of generating large labeled optimization datasets.
- Rigorous evaluation methodology for ML-as-solver-replacement: optimality
  gap, feasibility guarantees, *and* wall-clock speed — not just accuracy.
- LLMs as a *front-end to* optimization rather than a substitute for it:
  natural language constrained to a validated patch schema, with all
  arithmetic left to the solver.

## Reference

Van Hentenryck, P. (2025). *Optimization Learning*. arXiv:2501.03443.
