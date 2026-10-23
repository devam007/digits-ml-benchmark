# Model card

- **Purpose:** Learn reproducible classical ML model selection and error analysis.
- **Input:** 64 grayscale features representing an 8x8 digit; intensities 0-16.
- **Output:** One digit label from 0 to 9.
- **Selection:** Best validation macro F1 among the declared candidates.
- **Training data:** The training and validation partitions of sklearn load_digits.
- **Evaluation:** Fixed held-out partition; see the generated metrics and predictions.
- **Out of scope:** Arbitrary photographs, document OCR, writer identification,
  production deployment, and claims about accuracy on other populations.
- **Reproducibility:** Seed, split membership, dataset hash, and library versions
  are saved by the training command. Floating-point behavior may vary by platform.

## Questions to answer before an interview

1. Why use a baseline, and what does it predict here?
2. Why fit StandardScaler after splitting instead of before?
3. How does macro F1 differ from accuracy?
4. Which classes are confused most often in your report?
5. Why can the validation winner change with another split?
6. What would you change before testing real handwriting photographs?

## Personal experiment log

Add your own dated observations and modifications after reproducing the baseline.
