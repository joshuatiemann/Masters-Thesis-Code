# Graph Neural Network Policies for Robotic Locomotion

Supplementary code and experimental data for the Master's thesis by Joshua Tiemann.

This repository compares GNN-based actor policies (JOINT-Net, LEG-Net) against an MLP baseline for locomotion tasks (Ant, Centipede) trained with PPO in MuJoCo.

## Installation

```bash
cd Code/Baseline
conda env create -f environment.yml
conda activate GNN_Locomotion
```


## Running Training

Each environment variant has its own directory under `Code/`:

| Directory | Agent | Terrain |
|---|---|---|
| `Code/Baseline/` | Ant (4 legs) | Flat |
| `Code/Centipede/` | Centipede (3 segments, 6 legs) | Flat |
| `Code/Hills/` | Ant (4 legs) | Bumpy (curriculum) |

To train, navigate to the desired directory and run:

```bash
cd Code/Baseline   # or Centipede, Hills
python train.py
```

By default, `train.py` runs all three actor architectures sequentially (`leg_config`, `joint_config`, `mlp_config`). To run only a specific architecture, edit the `experiment_list` at the bottom of `train.py`:

```python
# train.py — bottom of file
experiment_list = [leg_config]          # LEG-Net only
# experiment_list = [joint_config]      # JOINT-Net only
# experiment_list = [mlp_config]        # MLP only
# experiment_list = [leg_config, joint_config, mlp_config]  # all three
```

## Configuring the GNN Architecture

### Changing the Number of GraphConv Layers (T)

The number of message passing iterations is controlled directly in `actors.py` by adding or removing `GraphConv` layers inside the `propagation_model` of `joint_actor` or `leg_actor`.

**T=2 (default for Ant):**
```python
self.propagation_model = Sequential("x, edge_index", [
    (GraphConv(64, 64), "x, edge_index -> x"),
    nn.Tanh(),
    (GraphConv(64, 64), "x, edge_index -> x"),
    nn.Tanh(),
])
```

**T=3 (default for Centipede — uncomment the third layer):**
```python
self.propagation_model = Sequential("x, edge_index", [
    (GraphConv(64, 64), "x, edge_index -> x"),
    nn.Tanh(),
    (GraphConv(64, 64), "x, edge_index -> x"),
    nn.Tanh(),
    (GraphConv(64, 64), "x, edge_index -> x"),
    nn.Tanh(),
])
```

In `Code/Baseline/actors.py`, the third layer is present but commented out — simply uncomment it to switch from T=2 to T=3.

### Changing the Node Feature Dimension (d)

The node feature dimension is the hidden size used throughout the GNN. To change it, update **all** matching dimension arguments in the actor class. For example, to change from `d=64` to `d=32` in `leg_actor`:

```python
# Input projection layers
self.joint_lin = nn.Sequential(Linear(4, 32))   # was 64
self.torso_lin = nn.Sequential(Linear(11, 32))  # was 64

# GraphConv layers
self.propagation_model = Sequential("x, edge_index", [
    (GraphConv(32, 32), "x, edge_index -> x"),   # was (64, 64)
    nn.Tanh(),
    (GraphConv(32, 32), "x, edge_index -> x"),   # was (64, 64)
    nn.Tanh(),
])

# Output heads
self.output_layer = nn.ModuleList([
    Linear(32, 4) for _ in range(4)              # was 64
])

# Also update the empty tensor allocation in forward():
h = torch.empty((num_nodes, 32), ...)            # was 64
```

The same pattern applies to `joint_actor` (change the `Linear` and `GraphConv` sizes identically).

### Changing the MLP Architecture

The MLP actor is configured via function arguments:

```python
def mlp_actor(num_cells=64, action_dim=8, hidden_layer=2, device="cpu"):
```

- `num_cells` — hidden layer width (e.g., 32, 64, 128)
- `action_dim` — number of actions (8 for Ant, 16 for Centipede)

The number of hidden layers is hardcoded in the function body (3 `LazyLinear` layers by default). Add or remove `LazyLinear` + `Tanh` pairs to change depth.

## Configuring PPO Hyperparameters

Edit the config objects in `config.py`:

```python
leg_config = ExperimentConfig(
    experiment='leg',
    actor='leg_actor',
    transform='leg_graph',
    terrain='flat',
    total_frames=1_000_000,     # total env steps
    frames_per_batch=2048,      # rollout buffer size
    sub_batch_size=64,          # minibatch size
    num_epochs=10,              # PPO epochs per batch
    lr=3e-4,                    # learning rate (cosine annealed)
    clip_epsilon=0.2,
    gamma=0.99,
    lmbda=0.95,
    entropy_eps=1e-4,           # entropy bonus coefficient
)
```

For the Centipede, the defaults differ — see `Code/Centipede/config.py` (notably `entropy_eps=0.02`, `sub_batch_size=256`, `num_epochs=4`).

For Hills training, set `terrain='hills'` (see `Code/Hills/config.py`).

## Code Files

| File | Purpose |
|---|---|
| `train.py` | PPO training loop (TorchRL `Trainer`). Runs configs from `experiment_list`. |
| `actors.py` | Actor networks: `mlp_actor`, `joint_actor` (JOINT-Net), `leg_actor` (LEG-Net) |
| `config.py` | `ExperimentConfig` with PPO hyperparameters and architecture selection |
| `transforms.py` | Converts flat observation vectors into PyTorch Geometric `Data` graphs |
| `hooks.py` | Custom hooks: cosine LR schedule, graph collation, evaluation logging |
| `sim_environment/` | Custom MuJoCo Gymnasium environments (Ant / Centipede) |
| `environment.yml` | Conda environment spec (in `Code/Baseline/`) |

## Experiment Data

Pre-trained checkpoints and training logs are stored in `Experiment1/`, `Experiment2/`, and `Experiment3/`. Each run directory contains:
- `trainer.pt` — TorchRL trainer checkpoint
- `scalars/` — TensorBoard-compatible training logs
- `videos/` — evaluation rollout recordings (where available)

## Dependencies

- **PyTorch** 2.4.1 &ensp;|&ensp; **TorchRL** 0.6.0 &ensp;|&ensp; **PyTorch Geometric** 2.6.1 &ensp;|&ensp; **MuJoCo** 2.3.7 &ensp;|&ensp; **Gymnasium** 0.29.1

Full spec: `Code/Baseline/environment.yml`
