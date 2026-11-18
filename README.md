# MAE 263F: Mechanics of Flexible Structures and Soft Robots

**Instructor:** Prof. M. Khalid Jawed

This repository includes the course materials and codes for MAE 263F: *Mechanics of Flexible Structures and Soft Robots*.

## Installation

Create and activate a virtual environment, then install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```bash
venv\Scripts\activate
```

### Neural ODE example (optional)

`requirements.txt` covers every script except the neural ODE example in `src/neuralODE/`, which also needs PyTorch. Install it only if you plan to run that example:

```bash
pip install -r requirements-neuralode.txt
```

On Linux, the default PyTorch package is the CUDA (GPU) build, which is several GB. If you don't need GPU support, install the CPU-only build first:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-neuralode.txt
```

## Running the Code

Run a Python file from the terminal using:

```bash
python newton1D.py
```

Replace `newton1D.py` with the name of the file you want to run.