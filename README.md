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
├── src/
│   └── gaussian_processes/
│       └── kernels.py
├── tests/
│   └── test_kernels.py
└── uv.lock
```