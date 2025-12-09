# SoftRMA: Compliant Control for Unitree H12

Model-free Rapid Motor Adaptation for learning compliant, human-interactive whole-body control on humanoid robots.

## Overview

SoftRMA enables the Unitree H12 robot to learn compliant behaviors that allow it to "yield" when pushed or pulled by a human. Instead of resisting external forces like traditional rigid controllers, the robot learns to minimize energy expenditure by intelligently yielding while maintaining stability.

The system is trained in two phases:
1. **Base Policy**: Learns compliant control with access to ground-truth force information
2. **Adaptation Module**: Learns to estimate forces from the robot's joint history, enabling deployment without external sensors

## Key Features

- **Model-Free Learning**: No inverse kinematics or hand-crafted admittance controllers needed
- **Online Adaptation**: Estimates external forces in real-time from joint state history

### Prerequisites

- **Isaac Lab**: Follow the [official installation guide](https://docs.isaacsim.nvidia.com/)
- **Python 3.8+**: Recommended to use conda or uv

### Setup

1. Clone this repository outside your core Isaac Lab directory:
   ```bash
   git clone <repo-url>
   cd h12_rma_compliance
   ```

2. Install the extension in editable mode:
   ```bash
   python -m pip install -e source/h12_rma_compliance
   ```

3. Verify installation:
   ```bash
   python scripts/list_envs.py
   ```
   Look for `H12-SoftRMA-Base-v0`, `H12-SoftRMA-Adapt-v0`, and similar tasks in the output.
## Usage
<!-- 
### Phase 1: Train Base Policy

Train the oracle policy with ground-truth force information:

```bash
python scripts/rsl_rl/train.py --task=H12-SoftRMA-Base-v0 --headless
```

### Phase 2: Train Adaptation Module

Train the force estimation network from the base policy checkpoint:

```bash
python scripts/rsl_rl/train.py --task=H12-SoftRMA-Adapt-v0 --checkpoint=<path/to/base_policy> --headless
``` -->

<!-- ### Phase 3: Deploy and Visualize

Run the trained policy with the adaptation module enabled:

```bash
python scripts/rsl_rl/play.py --task=H12-SoftRMA-Deploy-v0 --checkpoint=<path/to/adapted_policy> --num_envs=1
``` -->
<!-- ## Project Structure

```
h12_rma_compliance/
├── scripts/                 # Training and inference scripts
│   └── rsl_rl/             # RL training pipeline (train.py, play.py)
├── source/
│   └── h12_rma_compliance/ # Main package
│       ├── tasks/          # Isaac Lab environment configurations
│       └── h12_rma_compliance/
│           ├── tasks/      # Task definitions
│           │   └── manager_based/h12_rma_compliance/
│           │       ├── h12_rma_compliance_env_cfg.py
│           │       ├── agents/      # Policy configurations
│           │       └── mdp/         # Reward functions
│           └── ui_extension_example.py
└── README.md
``` -->

## License

[Add license information here]

