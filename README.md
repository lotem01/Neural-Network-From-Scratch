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
