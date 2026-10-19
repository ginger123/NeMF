# Training NeMF — Tutorial

How to configure, launch, and monitor a NeMF training run on the server.

## 1. Configuration and paths

Training is configured with Hydra. The config file is `configs/microphysics_train.yaml`, loaded by `train_NeMF.py`.

### `hydra.run.dir`: the output folder

```yaml
hydra:
  run:
    dir: ./NN_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Every run creates a new folder named after the date and time, and Hydra changes the working directory into it. Everything the run produces goes there:

```
NN_outputs/<date>/<time>/
├── .hydra/
│   ├── config.yaml      # the full config this run used (incl. command-line overrides)
│   ├── hydra.yaml
│   └── overrides.yaml
├── train_NeMF.log       # Hydra log (does NOT include the per-iteration prints)
├── events.out.tfevents… # TensorBoard logs
└── checkpoints/
    └── cp_<iter>.pth    # saved every checkpoint_iteration_interval iterations
```

- A relative `dir` (`./NN_outputs/...`) is relative to the directory you **launch from**. Use an absolute path if you want outputs in a fixed place.
- Keep each run folder together. `test_NeMF.py` finds the run folder from the checkpoint path (by cutting at `/checkpoints`) and loads `.hydra/config.yaml` from it to rebuild the network. A `.pth` copied on its own won't work for testing.

### Other path settings

- **`checkpoint_resume_path`**: only used when `resume: True`. A relative path is resolved against the launch directory, not the run folder. For a fresh run, keep `resume: False` and `checkpoint_resume_path: ''`.
- **Dataset location**: not in the yaml. `data.dataset_name` selects an entry, and the actual folder is hardcoded per dataset in `get_cloud_microphysics_datasets()` in `NeMF/microphysics_dataset.py`. The folder must contain `train/`, `validation/` and `test/` subfolders with `cloud*.pkl` files.

### Overriding settings from the command line

```bash
python train_NeMF.py hydra.run.dir=/abs/out_dir gpu=0 optimizer.lr=1e-4
```

Overrides are recorded in `.hydra/overrides.yaml` and in the saved `config.yaml`.

## 2. Launching on the server

Run inside `tmux` (or `screen`) so training survives an SSH disconnect, and `tee` the output. The per-iteration stats are written with `print`, which Hydra's log file doesn't capture.

```bash
tmux new -s nemf
cd ~/bproject/
source .venv/bin/activate
cd NeMF
python NeMF/train_NeMF.py 2>&1 | tee train_stdout.log
# detach: Ctrl-b d      reattach: tmux attach -t nemf
```

## 3. What gets logged

| What | When | Where |
|---|---|---|
| Console stats (loss, relative errors, lr, GPU memory) | every `stats_print_interval` (25) iterations | stdout |
| TensorBoard train scalars and scatter plots | every 25 iterations | `events.out.tfevents.*` |
| TensorBoard validation scalars and scatter plots | every `validation_iter_interval` (2500) iterations | `events.out.tfevents.*` |
| Checkpoints | every `checkpoint_iteration_interval` (10000) iterations | `checkpoints/cp_*.pth` |

The logger is `NeMF/visualization.py`, a wrapper around `tensorboardX`.

### In TensorBoard

**Scalars** (train and val overlaid on the same chart):
- `loss`
- `lwc_delta`, `lwc_epsilon`, and the same for `reff` and `veff`. Delta is the relative mass error and epsilon is the relative error.

**Images**: scatter plots of estimated vs. true values, with the correlation ρ in the title:
- `lwc_scatter_plot/…`
- `reff_scatter_plot/…`
- `veff_reff_scatter_plot/…` (color-coded by true reff)
- `veff_height_scatter_plot/…` (color-coded by height)

Each is shown per sample, as `train-0` and `val-<i>`. Use the step slider on an image to watch the points move toward the red y=x line as training goes on.

Notes:
- With the mask loss (`L2_relative_error_with_mask`), the main loss and the mask BCE loss are both logged under the `loss` tag, so the curve has two values per step and looks jagged. This comes from the original code.
- Validation plots first appear at iteration 2500. Before that there are only train scalars.

## 4. Viewing TensorBoard from your laptop

`tensorboardX` only writes the logs. To view them, install `tensorboard` on the server:

```bash
pip install tensorboard
```

Start it on the server (in another tmux window). Point it at `NN_outputs/` to see all runs side by side:

```bash
# NOTE: you have to use the .venv as well!
 python -m tensorboard.main --logdir NN_outputs --port 6006
```

Tunnel the port from your laptop, then open http://localhost:6006:

 Do This in Windows
```bash
ssh.exe -L 6006:127.0.0.1:6006 user@SERVER
```
