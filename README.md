This repository contains the code for the paper: **"Reinforcement Learning for Microcanonical Graph Ensemble with Assortativity Constraints"**.

It implements two methods for generating graphs under constraints on assortativity:

1. **ERGM** — an Exponential Random Graph Model (canonical ensemble with soft constraints) using Metropolis–Hastings sampling
2. **DMGG** — *Deep Microcanonical Graph Generator* (microcanonical ensemble with hard constraints) using a learned rewiring policy trained via Proximal Policy Optimization (PPO)

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [ERGM](#ergm)
- [DMGG](#dmgg)

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (strongly recommended for DMGG training)

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. Follow the steps below to set up the environment from scratch.

#### 1. Check that `uv` is installed

```bash
uv --version
```

If `uv` is not installed, see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

#### 2. Create a virtual environment inside the project directory

```bash
# From the root of this repository (where pyproject.toml and uv.lock lives)
uv venv
```

This creates a `.venv/` directory in the project root.

#### 3. Activate the virtual environment

```bash
source .venv/bin/activate          # Linux / macOS
# or
.venv\Scripts\activate             # Windows
```
After activation your shell prompt will be prefixed with `((dmgg))`.

#### 4. Install the project dependencies

```bash
uv sync
```

This installs all the packages requied in this project.


## Project Structure

```
project_root/
├── path.py                      # Global paths configuration
├── pyproject.toml               # Project configuration and dependencies
├── graph/                       # Graph utilities
├── ERGM/
│       ├── main.py              # ERGM sampler
│       ├── joint_degree_flux.py # Joint-degree flux computation
│       ├── warmup_detector.py   # Warmup detection for MCMC chains
│       └── data.py              # Data save and load utilities
├── DMGG/
│   ├── train.py                 # DMGG PPO training
│   ├── evaluate.py              # DMGG graph generation evaluation
│   ├── experiment_setup.py      # Training/evaluation setup utilities
│   ├── hyperparameter.py        # Hyperparameter and CLI parsing
│   ├── env/                     # Rewiring environment
│   ├── net/                     # Neural network modules
│   ├── RL/                      # Custom Stable-baselines3 modules
│   └── scatter/                 # Custom scatter operations for graph data
└── README.md
```

## ERGM

ERGM provides a Markov chain Monte Carlo sampler for generating graphs constrained on assortativity. It uses a canonical ensemble approach, where constraints are enforced only in expectation, and individual realizations fluctuate around the target.

Run `main.py` with the arguments listed in the parser help.

**Example** (targeting N=1000, E=3000, ER-type initial graph):

```bash
python ERGM/main.py --num_nodes 1000 --num_edges 3000 --graph_type ER --graph_seed 0 --seeds 0 1000 --max_steps 100_000 --assortativity_weight 0.126 --store metadata edge_list history
```

## DMGG

DMGG (Deep Microcanonical Graph Generator) reformulates graph generation as a Markov decision process. A learned policy selects rewiring actions that navigate the graph toward a target assortativity $\rho^*$, satisfying the hard constraint $|\rho − \rho^*| < \varepsilon$ with given tolerance $\varepsilon$ in every realization.

#### Training

The following hyperparameter values are drawn from Table 1 of the paper. To reduce computational cost, training is restricted to small, sparse graphs from WS, ER, and BA topologies with a narrow target range and a loose tolerance.

```bash
python DMGG/train.py \
  --graph_types er ws ba \
  --num_nodes_range 100 1000 \
  --mean_degree_range 3.0 10.0 \
  --target_rho_range -0.5 0.5 \
  --tolerance 0.005 \
  --max_rewires 3000 \
  --n_steps 2048 \
  --num_envs 32 \
  --total_timesteps 50_000_000
```

#### Evaluation

After training, evaluate on larger graphs, a wider range of target assortativity, and unseen topologies (SBM, RGG, CL, HK, BA).

```bash
python DMGG/evaluate.py \
  --exp_id <experiment_id> \
  --graph_type er \
  --num_nodes 1000 \
  --mean_degree 6.0 \
  --graph_seed 0 \
  --target 0.8 \
  --num_graphs 1000 \
  --store metadata edge_list history
```

For full lists of arguments, see each module with `python <module>.py --help`.
