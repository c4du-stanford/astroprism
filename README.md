# astroprism
A probabilistic forward modeling framework for multi-wavelength astronomical imaging

## Installation

Homebrew Python is "externally managed" (PEP 668), so install into a virtual environment rather
than system-wide:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `astroprism` CLI entry point. Activate the same environment
(`source .venv/bin/activate`) in any new shell before running the CLI.

> Prefer conda? `conda create -n astroprism python=3.11 && conda activate astroprism && pip install -e .`
> works too — conda environments are not externally managed.

## Running inference

Inference is driven by a YAML config. The canonical reference is the packaged
`src/astroprism/configs/default.yaml` (also mirrored at `configs/default.yaml`) —
copy it, edit `data.path`, `data.instrument`, and `inference.output_directory`, then run:

```bash
astroprism run --config configs/my_run.yaml
```

Results (checkpoints, `config.yaml`, `files_used.yaml`, and any masks) are written to the
`inference.output_directory` set in the config. Runs are resumable: set `inference.resume: true`
and re-run the same command to continue from the last checkpoint toward `n_iterations`.

Once a run is complete, generate predictions from it:

```bash
astroprism predict --run-dir output/my_run --quantities signal response noise_std
```

## Running in a tmux session

Inference runs are long, so launch them inside [`tmux`](https://github.com/tmux/tmux) to keep
them alive after you disconnect (e.g. when running on a remote server over SSH).

```bash
# 1. Start a named session
tmux new -s astroprism

# 2. Inside the session, activate your environment and start the run
conda activate astroprism          # or: source .venv/bin/activate
astroprism run --config configs/my_run.yaml 2>&1 | tee output/my_run/run.log

# 3. Detach without stopping the run: press Ctrl-b then d
```

Reattach later to check progress:

```bash
tmux attach -t astroprism          # reattach
tmux ls                            # list running sessions
```

To stop a run, reattach and press `Ctrl-c`, or kill the session entirely:

```bash
tmux kill-session -t astroprism
```

Piping through `tee` keeps a copy of the console output in `run.log` so you can review it after
detaching. To watch the log live from another shell without attaching to tmux:

```bash
tail -f output/my_run/run.log
```
