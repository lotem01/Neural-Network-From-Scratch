# Project context

## Goal and assignment

Build a neural network classifier from scratch to learn the mathematics, optimization, and engineering behind deep learning frameworks. The supplied brief currently specifies Part I: implement stable softmax regression and its weight/bias gradients, demonstrate the gradient test, implement and verify SGD or a variant on synthetic least squares, and run softmax experiments on GMM plus either SwissRoll or Peaks. Report training and validation accuracy and loss by epoch, experiments with learning rates and batch sizes, and the requested plots and optimizer code. Part I is worth 30%; its stated deadline is December 10, without a year explicitly attached to that date.

## 2026-11-20: project setup and data inspection

### Decisions and reasons

- Use Python 3.13 in a local `.venv`, with NumPy, SciPy, and Matplotlib: these support vectorized numerical work, MATLAB data files, and the required plots. Isolation avoids altering system packages, at the cost of a separate installation.
- Keep the original archive in Downloads and extract its three named `.mat` files to `data/`. Compare extracted bytes with the archive to verify the inputs were preserved. At the user's request, include the three files in Git so collaborators can run the project immediately. This adds approximately 1 MB to the repository and shares the supplied course data with repository collaborators.
- Inspect all three datasets before choosing their internal representation or normalization. This avoids silently guessing their orientation, at the cost of deferring the data loader implementation.
- Use plain scripts and a small dependency list. This is easier to understand for a course project, though it provides less automation than a larger project framework.
- Pin the installed direct and transitive package versions in `requirements.txt` for repeatability on Python 3.13 and Windows. This may require updating versions for other platforms or future security fixes.
- The user requested connecting the project to their GitHub repository. Ignore the virtual environment and generated files, while tracking the course datasets at the user's request.
- Keep code precise, purposeful, and easy to explain in a frontal review. Prefer simple, readable implementations and document meaningful decisions and trade-offs as the project develops.

### Environment and obstacles

See `README.md` for setup and inspection commands. Creating the virtual environment initially failed because Python could not write to its temporary directory under the sandbox; rerunning with approved elevated access succeeded. Package installation also required approved network access.

### Verification

The three extracted files were compared byte-for-byte with their entries in `HW1_Data.zip`; all matched. The inspection script found all four expected arrays in each file. All arrays are `float32`, with samples in columns and features or classes in rows. All labels are one-hot, and training and validation feature and class dimensions match.

| Dataset | Features | Classes | Training samples | Validation samples |
| --- | ---: | ---: | ---: | ---: |
| SwissRoll | 2 | 2 | 20,000 | 5,000 |
| Peaks | 2 | 5 | 25,000 | 6,250 |
| GMM | 5 | 5 | 25,000 | 6,250 |

Class counts are approximately balanced; exact counts and every array shape are printed by `scripts/inspect_data.py`. Inspection only reads the files. The classifier's eventual internal orientation remains a separate decision.

Two fresh inspection runs both exited successfully and produced identical output. `pip check` reported no broken dependencies. NumPy 2.5.5, SciPy 1.18.1, and Matplotlib 3.11.2 imported successfully. Matplotlib reported a cache-directory permissions warning in the restricted sandbox; future plotting commands may need `MPLCONFIGDIR` set to a writable project directory. No plotting was required for this step.

## Next steps

Choose a documented internal sample orientation and decide whether to normalize using training-set statistics. Then implement a vectorized, numerically stable softmax loss and analytical gradient, and verify the gradient with the course's directional test before integrating SGD.
