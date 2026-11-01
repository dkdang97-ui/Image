# Tom and Jerry Frame Classifier

A Jupyter notebook that trains a deep learning model to recognise which characters appear in a frame from the *Tom and Jerry* cartoon. Each frame is sorted into one of four classes:

| Class | Meaning |
|---|---|
| `tom` | only Tom is in the frame |
| `jerry` | only Jerry is in the frame |
| `both` | Tom and Jerry are both in the frame |
| `neither` | neither character is in the frame (empty scenes or other characters) |

The model uses **transfer learning**: a ResNet-34 network pretrained on ImageNet photographs is fine-tuned on cartoon frames with the [fastai](https://docs.fast.ai/) library.

## Results

Results from the most recent run, measured on 1,095 validation frames the model never trained on:

| Metric | Value |
|---|---|
| Validation error rate | 8.8% (96 of 1,095 frames) |
| Error rate before fine-tuning the whole network | 26.6% |

| Class | Correctly classified | Accuracy |
|---|---|---|
| `tom` | 365 of 387 | 94% |
| `neither` | 296 of 322 | 92% |
| `jerry` | 195 of 219 | 89% |
| `both` | 143 of 167 | 86% |

Key findings:

- The model never confuses Tom with Jerry. Almost all errors are about whether a character is present at all, most often missing a small or partly visible Jerry.
- Several of the frames with the highest loss appear to be **mislabelled** in the dataset, so part of the measured error is not the model's fault.
- The model does **not** generalise to images that look different from cartoon frames: it classified the official Wikipedia picture of Jerry as `neither`.
- Neighbouring video frames are near-copies, and the random train/validation split can put them on both sides, so the 8.8% error rate is probably optimistic for new episodes.

## Data

- **Tom and Jerry Image Classification** dataset on Kaggle: https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification
  - 5,478 labelled frames in four folders: `tom`, `jerry`, `tom_jerry_1` (both) and `tom_jerry_0` (neither).
  - Downloaded automatically with [`kagglehub`](https://github.com/Kaggle/kagglehub). No Kaggle account is needed for this public dataset.
  - The frames come from a copyrighted cartoon and are used for learning and research only. See the Kaggle page for the dataset's terms of use.
- **Labels** come from the folder each image is stored in, because the file names are just frame numbers (`frame1234.jpg`).
- **Extra test picture**: the image of Jerry from the [Jerry Mouse Wikipedia page](https://en.wikipedia.org/wiki/Jerry_Mouse).

## Notebook contents

`tom_and_jerry_image_classification.ipynb`

1. **Introduction and data source**: the problem and where the data come from.
2. **Set up**: imports and a fixed random seed so results can be reproduced.
3. **Data processing**: download the frames, create labels from folder names, resize to 224 × 224 pixels, and split 80/20 into training (4,383) and validation (1,095) sets.
4. **Exploratory data analysis**: look at example frames and count the images in each class.
5. **Modeling**: fine-tune ResNet-34 with `fine_tune(epochs=4)`. First only the new final layer is trained, then the whole network.
6. **Model assessment**: a confusion matrix, the most common mistakes, and the frames with the highest loss.
7. **Deployment**: predictions on a frame from the dataset and on a picture of Jerry from outside it.
8. **Conclusion**: results, limitations and possible next steps.

## How to run

### Google Colab (recommended)

1. Upload `tom_and_jerry_image_classification.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Click **Runtime → Change runtime type** and set **Hardware accelerator** to **T4 GPU**.
3. Click **Runtime → Run all**. Training takes about 5 minutes on a T4 GPU.

### Locally

```bash
pip install fastai kagglehub pandas seaborn requests jupyter
jupyter notebook tom_and_jerry_image_classification.ipynb
```

Then choose **Kernel → Restart & Run All**. A GPU is strongly recommended. The first run downloads the dataset and the pretrained ResNet-34 weights (about 83 MB). Both are saved locally, so later runs reuse them.

## Reproducibility

The notebook calls `set_seed(42, reproducible=True)`, and the train/validation split uses `RandomSplitter(seed=1)`, so repeated runs on the same hardware should give the same results. Numbers may still differ slightly on different GPUs because some GPU operations are not fully deterministic.

## Possible next steps

- Split the data by episode or by blocks of consecutive frames, so validation frames are not near-copies of training frames.
- Correct the mislabelled frames found among the highest-loss images.
- Use data augmentation (random crops, zooms and lighting changes) to help with small and partly visible characters.
- Add other kinds of Tom and Jerry images to the training data so the model works beyond cartoon frames.

## Requirements

- Python 3.9+
- fastai (with PyTorch)
- kagglehub
- pandas
- matplotlib
- seaborn
- requests
