# Coordinate descent

`cd_core.py` implements the coordinate-descent hyperparameter search used for full
retuning (the per-batch-size retune of the language model at 256K tokens and of CIFAR-5M at
batch sizes 512 to 4096). It is pure Python (standard library only) with no scheduler: a `Tuner` holds one search,
`Tuner.step(lookup)` advances it as far as the finished runs allow and returns the runs it
still needs as `(values, seed, priority)` tuples. `lookup(values, seed)` returns a loss,
`FAILED` (treated as +inf) or `None` (not yet finished). Any job runner can drive it.

## Protocol

- **Coordinates.** Each hyperparameter is a coordinate of kind
  - `mult`: multiply the centre value by each factor in {1/2, 1/sqrt(2), sqrt(2), 2}
    (several keys can move together, e.g. tied auxiliary learning rates);
  - `refresh`: for a retention coefficient q (momentum, Adam betas), scale the refresh rate
    1 - q by the same factors, keeping 0 < q < 1;
  - `grid`: move to the two grid points below and the two above the centre (used for
    weight decay and the cooldown fraction in the language model).
- **Seeds.** A search has three seeds `seeds = (s0, s1, s2)`.
- **Centre.** Every centre is run with the first `center_seeds` seeds (default 2). The centre
  median is the median over all finished centre runs among the three seeds (after a move,
  this includes the confirmation runs of the new centre).
- **Sweep.** Coordinates are visited in turn. At the current centre, the current coordinate's
  candidates are run with `s0`.
- **Move rule** (`delta = 0.002`, `large = 0.006` in our searches; a failed run counts as
  +inf). Let the gain be the centre median minus the best candidate's `s0` loss.
  1. If the gain is at most `delta`, the centre is kept.
  2. Otherwise the best candidate is confirmed on `s0, s1` when the gain is at least `large`,
     and on `s0, s1, s2` otherwise.
  3. If a two-seed confirmation's median gain lies strictly between `delta` and `large`,
     `s2` is added and the median over all three seeds is used.
  4. The move is accepted when the confirmation median beats the centre median by more than
     `delta`; otherwise the candidate is rejected.
- **Convergence.** The search stops after as many consecutive coordinates as there are
  coordinates (7 in our searches) without a move.
- **Speculation.** With `speculate=True` (the default), the tuner also requests the next
  coordinate's candidates at the current centre so that parallel workers stay busy; it does
  not change any decision.

## Example

```python
from cd_core import Coord, Tuner, key

coords = [Coord("lr", "mult", ("lr",)), Coord("momentum", "refresh", ("momentum",))]
tuner = Tuner("demo", coords, center={"lr": 0.02, "momentum": 0.95},
              seeds=(42, 43, 44), delta=0.002, large=0.006, center_seeds=1)
finished = {}
while not tuner.done:
    for values, seed, _ in tuner.step(lambda v, s: finished.get((key(v), s))):
        finished[(key(values), seed)] = train_and_evaluate(values, seed)  # your runner
print(tuner.center, tuner.events)
```

## Running candidates

The tuner only proposes hyperparameter values; the training scripts evaluate them.
For the language model, each candidate becomes a one-case plan via
`llm/scripts/make_rule_sweep.py --rules FFFFFF --set KEY=VALUE ...` (see `llm/README.md`).
For CIFAR-5M, each candidate is one call of `cifar5m/train_groupnorm_resnet_cifar5m.py` with
explicit values and `fixed` rules (see `cifar5m/README.md`).
