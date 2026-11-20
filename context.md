# Project context

## Goal and assignment

Build a neural network classifier from scratch to learn the mathematics, optimization, and engineering behind deep learning frameworks. The supplied brief currently specifies Part I: implement stable softmax regression and its weight/bias gradients, demonstrate the gradient test, implement and verify SGD or a variant on synthetic least squares, and run softmax experiments on GMM plus either SwissRoll or Peaks. Report training and validation accuracy and loss by epoch, experiments with learning rates and batch sizes, and the requested plots and optimizer code. Part I is worth 30%; its stated deadline is December 10, without a year explicitly attached to that date.

## 2026-11-20: project setup and data inspection

### Decisions and reasons

- Use Python 3.13 in a local `.venv`, with NumPy, SciPy, and Matplotlib: these support vectorized numerical work, MATLAB data files, and the required plots. Isolation avoids altering system packages, at the cost of a separate installation.
- Keep the original archive in Downloads and extract its three named `.mat` files to `data/`. Compare extracted bytes with the archive to verify the inputs were preserved. At the user's request, include the three files in Git so collaborators can run the project immediately. This adds approximately 1 MB to the repository and shares the supplied course data with repository collaborators.
- Keep samples in columns: features have shape `features × samples` and targets have shape `classes × samples`. This matches the supplied data and course notes; Python examples that use samples in rows will need their formulas adapted.
- Simplified the inspection script to check the chosen column layout directly: feature and label column counts must match, and each label column must be one-hot. This is easier to explain than trying both orientations; the script does not automatically detect or convert transposed input.
- Start with the original feature values, without normalization. This keeps the baseline simple and close to the supplied data, although different feature scales may make SGD more sensitive to its learning rate. If needed later, evaluate normalization using training-set statistics only and apply the same transformation to validation data.
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

Class counts are approximately balanced; exact counts and every array shape are printed by `scripts/inspect_data.py`. Inspection only reads the files.

Two fresh inspection runs both exited successfully and produced identical output. `pip check` reported no broken dependencies. NumPy 2.5.5, SciPy 1.18.1, and Matplotlib 3.11.2 imported successfully. Matplotlib reported a cache-directory permissions warning in the restricted sandbox; future plotting commands may need `MPLCONFIGDIR` set to a writable project directory. No plotting was required for this step.

## 2026-11-20: softmax loss and gradient

- Added `softmax.py` with an `add_bias` helper and `loss_and_gradient`. Features (including the bias row) have shape `(features + 1, samples)`, weights have shape `(features + 1, classes)`, and targets have shape `(classes, samples)`. Biases are the final weight row. This follows the course notation and avoids separate bias derivatives; the trade-off is explicitly adding a row of ones.
- For scores `W.T @ X`, subtract the largest score per sample before computing log probabilities. Compute the loss directly from log probabilities, which avoids both exponential overflow and taking the logarithm of an underflowed probability. The gradient is `X @ (P - C).T / samples`; it applies to targets whose columns sum to one. The functions do not normalize inputs or loop over samples.
- Added a reproducible directional gradient test with seed 42 and a generated plot in `results/softmax_gradient_check.png`. The direction includes the bias weights. Halving the step approximately halves the loss change and quarters the first-order Taylor remainder, as expected. The plot is generated locally and ignored by Git; its script is tracked.
- Three unit checks passed: a hand-computed loss and gradient including bias, finite results with extreme logits, and central-difference checks of every weight. A smoke check on the first 100 training samples of each supplied dataset produced the expected uniform-class loss `log(number of classes)` at zero weights.

## Next steps

Implement a reusable mini-batch SGD optimizer and verify it on a synthetic least-squares problem before applying it to the softmax classifier.
