# Information Span Economics

A standalone economics + machine-learning research laboratory for studying how the **breadth** and **precision** of borrower information change competition, relationship rents, and lending outcomes.

The project is inspired by the distinction between **information span** (how many economically relevant dimensions become hard/transferable) and **signal precision** (how accurately each covered dimension is measured). It is an independent methodological extension, not a replication and not affiliated with the original authors.

## Core economic hypothesis

Let `R_id` indicate a relationship/informed lender and let `eta_d` be information span. A simple reduced-form benchmark is

```text
Y_id = ... + beta_R R_id + beta_eta eta_d + beta_Reta (R_id * eta_d) + error_id.
```

The main hypothesis is `beta_Reta < 0`: broader transferable information erodes relationship-specific information rents.

The project deliberately keeps **span** and **precision** separate because they can have different competitive effects.

## Pipeline

```text
financial text
    -> interpretable economic dimensions
    -> information span + information precision
    -> synthetic credit-market equilibrium
    -> interaction benchmark / optional causal forest
    -> heterogeneous effects by opacity, competition, and relationship status
```

## Features

- deterministic offline text-to-information features;
- optional external Transformer scoring hook;
- synthetic lender/borrower market with endogenous loan pricing;
- interaction benchmark that works with NumPy/Pandas only;
- optional `econml.CausalForestDML` wrapper;
- no proprietary data and no dependency on Dealscan/WRDS.

## Quick start

```bash
pip install -e .
python examples/run_demo.py
pytest
```

Optional causal ML:

```bash
pip install -e '.[causal]'
python examples/run_demo.py --causal-forest
```

## Real-data research designs

Potential sources of quasi-experimental variation include credit-registry expansion, open-banking/data-portability reforms, lender software rollouts, digitization shocks, and other changes that plausibly harden previously soft information. ML should estimate nuisance functions or heterogeneity **after** the identification design is fixed; it does not create identification by itself.

See `docs/RESEARCH_DESIGN.md` for a fuller roadmap.
