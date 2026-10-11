# Training-only objective alignment audit

Rows 167; positive-label states 148; temperature 0.01.

Teacher is rho minimum: 148/148; target peaks are reward optimal: 148/148.
Prior128 teacher coverage: 87/148. Mean teacher mass 0.197259; optimal-set mass 0.244396; effective candidates 53.12.
Mean target expected rho gap 0.005959; mass beyond .001 gap 0.5591.

Valid peaks/reward alignment are verified, not a code defect. Distributed soft mass and bounded prior residual are approximations; neither proves downstream harm nor hard-label superiority. Independent residual promotion possibility is only a mathematical capacity bound, not realizability by the shared network.

Saved source checks are training-only. No fitting, rollout or heldout outcome access. A sharpening trial, if chosen, is a model configuration hypothesis and not new methodological contribution.
