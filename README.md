# Gaussian Processes

Educational implementation of Gaussian process regression in Python.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Doxygen for building the documentation
- LaTeX distribution for generating the PDF documentation

## Setup

Clone the repository:

```bash
git clone https://github.com/SamBale1109/gaussian-processes.git
cd gaussian-processes
```

Install the Python dependencies:

```bash
uv sync --locked
```

`uv` creates a virtual environment in `.venv`.

Activate the virtual environment on Linux, macOS, or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activation is optional when using `uv run`. For example:

```bash
uv run python
```

runs Python directly inside the project's environment.

To add a new dependency to project run:

```bash
uv add <DEPENDENCY_NAME>
```

## Tests

Run the test suite with:

```bash
uv run pytest
```

Run the linter with:

```bash
uv run ruff check .
```

Check formatting with:

```bash
uv run ruff format --check .
```

To automatically format the source code:

```bash
uv run ruff format .
```

## Obstacle avoidance testbed

`examples/reactive_avoidance_testbed.py` is a 2D testbed for reactive
obstacle avoidance on distance / gradient fields.  It ships a
built-in demo world (three obstacles, moving start / goal).

```bash
uv run python examples/reactive_avoidance_testbed.py              # live animation
uv run python examples/reactive_avoidance_testbed.py --no-anim    # static plot
uv run python examples/reactive_avoidance_testbed.py --save-plot out.png
```

The controller can be selected with `--controller {blended,
circulation, circulation-hysteresis, cbf, cbf-circ, mpc, mpc-circ,
rmp, rmp-circ, residual_rl}` (default: `circulation-hysteresis`).

### Custom environments

The world is customizable.  `examples/create_environment.py` is an
interactive editor: place circle and box obstacles (and the start /
goal markers) with the mouse, then save the result as a JSON file.

```bash
uv run python examples/create_environment.py                  # blank 10x10 world
uv run python examples/create_environment.py my_scenario.json # edit an existing file
```

Editor keys: `c` circle, `b` box, `d` delete, `m` move obstacle,
`s` move start, `g` move goal, `F2` save, `q` / `Esc` quit.
Obstacles are created static; to make one move, add a `"velocity"`
field to its entry in the saved JSON file.

Load the saved scenario in the main script with `--scenario`:

```bash
uv run python examples/reactive_avoidance_testbed.py --scenario my_scenario.json
```

### Riemannian Motion Policy (RMP) controller

`RMPController` (in `src/gaussian_processes/controllers.py`)
implements the RMP framework for obstacle avoidance: each task is a
generalized force `F` paired with a positive-definite metric `M`, and
tasks compose by summing forces and metrics,

```text
v = (sum_i M_i) ** -1 (sum_i F_i)
```

Two (optionally three) isotropic leaves are composed: goal attraction
(`F = m_g * v_nom`, `M = m_g * I`) and obstacle avoidance
(`F = m_a * w(d) * v_max * grad_d / ||grad_d||`, `M = m_a * w(d) * I`),
where `w(d)` is a smooth proximity weight.  Gating the avoidance
*metric* as well as its force means an inactive leaf contributes
neither force nor inertia, so the command reduces exactly to the
nominal policy in free space; the metric ratio `m_a / m_g` sets how
strongly avoidance competes with goal-seeking.  `rmp-circ` adds a
third tangential swirl leaf (latched direction) that breaks symmetric
stalls when the goal lies behind an obstacle, analogous to `cbf-circ`.

```bash
uv run python examples/reactive_avoidance_testbed.py --controller rmp
uv run python examples/reactive_avoidance_testbed.py --controller rmp-circ
```

### Residual RL policy

`examples/residual_rl_env.py` wraps the testbed as a Gymnasium
environment in which an agent learns only a *residual* velocity
`d_v` on top of a base policy (currently the goal-attraction
`NominalPolicy`; the same observation works for an LfD base policy
later).  The executed velocity is `v = v_nom + d_v`, saturated to
`v_max`.  `examples/train_residual_rl.py` trains a PPO policy and
compares it against the base policy on fresh random layouts (and
optionally a custom scenario).

The RL dependencies (gymnasium, stable-baselines3, torch) live in the
opt-in `rl` dependency group and are *not* installed by a plain
`uv sync`:

```bash
uv sync --group rl                       # one-time install
uv run --group rl python examples/train_residual_rl.py
```

Useful options:

```bash
# larger run
uv run --group rl python examples/train_residual_rl.py \
    --total-steps 1000000 --n-envs 32 --out runs/ppo_big

# evaluate an already-trained model (optionally on a custom scenario)
uv run --group rl python examples/train_residual_rl.py \
    --eval-only runs/ppo_big/model.zip --scenario my_scenario.json
```

World-sampling ranges (e.g. `--max-radius 1.6 --max-obstacles 3`) and
reward weights (e.g. `--w-time 0.5`) can be overridden on the command
line; see `--help` for all options.

You can also run a saved model in the interactive testbed (demo world
or a custom scenario):

```bash
uv run --group rl python examples/reactive_avoidance_testbed.py \
    --controller residual_rl --rl-model runs/residual_rl_main/model.zip
uv run --group rl python examples/reactive_avoidance_testbed.py \
    --controller residual_rl --rl-model runs/residual_rl_main/model.zip \
    --scenario my_scenario.json
```

After training, the model and an evaluation summary are written to
`--out` (default: `runs/residual_rl_<timestamp>/`).

## Documentation

The API documentation is generated with Doxygen.

### HTML documentation

Generate the documentation from the repository root:

```bash
mkdir -p docs/_build
doxygen Doxyfile
```

The generated HTML documentation is located at:

```text
docs/_build/html/index.html
```

On WSL, it can be opened in the Windows browser with:

```bash
explorer.exe docs/_build/html/index.html
```

Alternatively, open `docs/_build/html/index.html` directly in a browser.

### PDF documentation

Doxygen also generates LaTeX sources in:

```text
docs/_build/latex/
```

Build the PDF with:

```bash
make -C docs/_build/latex
```

The resulting PDF is:

```text
docs/_build/latex/refman.pdf
```

On WSL:

```bash
explorer.exe docs/_build/latex/refman.pdf
```

A working LaTeX installation providing `pdflatex` is required.

On Ubuntu/WSL, the required packages can be installed with:

```bash
sudo apt update
sudo apt install doxygen texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

## Project structure

```text
.
├── Doxyfile
├── README.md
├── pyproject.toml
├── examples/
│   ├── plot_gp_prior.py
│   ├── create_environment.py
│   ├── reactive_avoidance_testbed.py
│   ├── residual_rl_env.py
│   └── train_residual_rl.py
├── src/
│   └── gaussian_processes/
│       ├── controllers.py
│       ├── distance_gradient_fields.py
│       ├── kernels.py
│       ├── prior.py
│       └── qp.py
├── tests/
│   └── test_kernels.py
└── uv.lock
```