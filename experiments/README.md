# Earlier experiments

The scripts behind the experiments that led to the product, kept as they ran. They are listed by name in [`baselines/results/experiments.tsv`](../baselines/results/experiments.tsv). Run them from the top of the repo. Some need data the generators in `testbed/` make first.

Their results shaped the method, and none of the public numbers come from them. Accuracy results from `v2-curve2`, `v2-seen` and `v2-combo` were thrown out when we found a bug in the training rows.

`v2_*` is the second version of the desk. In these scripts, `argon` or "the comparison model" is Gemini 4 Argon, which wrote reference procedures from the same records so we could compare. The product does not use it.

Module paths in these scripts were updated when `island/` became `testbed/`. Nothing else changed.
