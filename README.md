# 😤 Facial Emotion Recognition: Traditional CNN vs Dilated CNN

A deep learning project that compares two CNN architectures for facial emotion recognition, trained on **FER2013** and evaluated on **RAF-DB** for cross-dataset generalization.

---

## 👥 Team Members

| Name | Student ID |
|---|---|
| Omar Waleed | 192100103 |
| Youssef Gabr | 192100069 |
| Youssef Anis | 192100029 |
| Shehab El Gohary | 192100097 |

---

## 📌 Overview

| | Details |
|---|---|
| **Task** | 7-class Facial Emotion Classification |
| **Train Dataset** | FER2013 (28,709 images) |
| **Val/Test Dataset** | RAF-DB (12,271 val / 3,068 test) |
| **Framework** | PyTorch |
| **Runtime** | Google Colab (T4 GPU) |
| **Classes** | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |

---

## 🧠 Models

### TraditionalCNN
A standard CNN with 4 convolutional blocks using regular 3×3 convolutions, BatchNorm, ReLU, and MaxPooling — followed by a fully connected classifier.

### DilatedCNN
Same structure as TraditionalCNN but uses **dilated convolutions** (atrous convolutions) to capture larger receptive fields without losing spatial resolution.

---

## 📁 Project Structure

```
FER_Project/
├── DilatedCNN_best.h5          # Best DilatedCNN weights
├── TraditionalCNN_best.h5      # Best TraditionalCNN weights
├── comparison_results.png      # Final model comparison chart
├── learning_curves.png         # Training/validation curves
└── results.csv                 # Metrics comparison table
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install kaggle torchmetrics codecarbon thop opencv-python
```

### 2. Download datasets via Kaggle
```bash
# FER2013
kaggle datasets download -d msambare/fer2013 -p /content/fer2013 --unzip

# RAF-DB
kaggle datasets download -d shuvoalok/raf-db-dataset -p /content/rafdb --unzip
```

> Make sure your `kaggle.json` credentials file is in `/root/.kaggle/`

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Make sure you're using a **T4 GPU** runtime (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells from top to bottom
4. Models are automatically saved to `/content/FER_Project/`

---

## 🧪 How to Test the Model

Add this cell at the end of the notebook to test both models on a custom face image:

```python
import torch, cv2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

# Load models
trad_model = TraditionalCNN().to(DEVICE)
trad_model.load_state_dict(torch.load('/content/FER_Project/TraditionalCNN_best.h5'))
trad_model.eval()

dila_model = DilatedCNN().to(DEVICE)
dila_model.load_state_dict(torch.load('/content/FER_Project/DilatedCNN_best.h5'))
dila_model.eval()

# Predict function
def predict_emotion(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img).convert('L')
    tensor = eval_transforms(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        trad_pred = CLASSES[trad_model(tensor).argmax(1).item()]
        dila_pred = CLASSES[dila_model(tensor).argmax(1).item()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, pred in zip(axes, ['TraditionalCNN', 'DilatedCNN'], [trad_pred, dila_pred]):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{name}\nPredicted: {pred}', fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Upload image and predict
uploaded = files.upload()
fname = list(uploaded.keys())[0]
predict_emotion(fname)
```

---

## 📊 Key Hyperparameters

| Parameter | Value |
|---|---|
| Image Size | 48×48 |
| Batch Size | 128 |
| Epochs | 50 |
| Learning Rate | 1e-4 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |

---

## 📝 Notes

- **Cross-dataset evaluation**: Trained on FER2013, tested on RAF-DB — tests real-world generalization across different data distributions.
- **Class balancing**: `WeightedRandomSampler` handles FER2013's class imbalance.
- **Augmentation**: Random horizontal flip, rotation (±15°), and affine translation applied during training only.
- **Energy tracking**: `codecarbon` tracks energy consumption (kWh) and CO2 emissions for both models.
