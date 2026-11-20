# Deep Learning from Scratch

Part I starts with softmax regression and a reusable stochastic optimizer. The current stage sets up Python and inspects the supplied data. See [context.md](context.md) for decisions, findings, and next steps.

## Setup (Windows PowerShell)

From the project directory, with Python 3.13 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The three original `.mat` files are included in `data/`. They were extracted without modification from the supplied `HW1_Data.zip` archive.

Dependencies are pinned to the versions verified on Windows with Python 3.13. Other platforms or Python versions may need different compatible versions.

## Inspect the data

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py
```

The script reports matrix shapes and data types, checks the supplied train/validation splits and one-hot labels, and prints class counts. A warning or error makes the script exit with a nonzero status so unexpected formats do not pass unnoticed. It does not transform the data.

## Softmax regression

`softmax.py` implements the mean cross-entropy loss and its analytical gradient. Samples are columns. `add_bias` appends a row of ones to the features; the last row of the weight matrix then holds the class biases. Targets are one-hot columns (or probability columns that sum to one). The implementation subtracts the largest score for each sample before computing log probabilities, avoiding exponential overflow.

Run the checks and create the assignment's gradient-test plot from the project directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m experiments.gradient_check
```

The second command saves `results/softmax_gradient_check.png` and prints, for each k with ε = 0.1·0.5^k, the zero-order error |F(W + εd) − F(W)| and the first-order error |F(W + εd) − F(W) − ε⟨d, ∇F⟩|. As in the course notes (section 2.10.1), both are plotted on a semilogarithmic scale against k. The results directory is generated locally and is not tracked in Git.
