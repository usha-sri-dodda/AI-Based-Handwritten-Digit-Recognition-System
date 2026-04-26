# 🔢 AI-Based Handwritten Digit Recognition System

CNN + MLP trained on MNIST with a Streamlit web interface.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model files + visualizations)
python train_model.py

# 3. Launch the Streamlit app
streamlit run app.py
```

## Project Structure

```
digit_recognition/
├── train_model.py          # Full training pipeline (CNN + MLP)
├── app.py                  # Streamlit web interface
├── digit_recognition.ipynb # Jupyter Notebook walkthrough
├── requirements.txt        # Python dependencies
├── models/                 # Saved Keras models (created after training)
│   ├── cnn_mnist.keras
│   └── mlp_mnist.keras
└── assets/                 # Generated plots (created after training)
    ├── sample_digits.png
    ├── training_curves.png
    ├── cm_cnn.png
    ├── cm_mlp.png
    ├── predictions.png
    ├── misclassified.png
    └── metrics.json
```

## Model Architecture (CNN)

| Layer | Config |
|---|---|
| Conv2D ×2 | 32 filters, 3×3, ReLU |
| MaxPool + Dropout | 2×2, 25% |
| Conv2D ×2 | 64 filters, 3×3, ReLU |
| MaxPool + Dropout | 2×2, 25% |
| Dense + Dropout | 256 units, 50% |
| Softmax Output | 10 classes |

**Test Accuracy: ~99.3%** on MNIST 10,000 test images.

## Features
- ✅ CNN + MLP training with EarlyStopping & ReduceLROnPlateau
- ✅ Confusion matrix, classification report, loss curves
- ✅ Prediction visualiser with confidence scores
- ✅ Misclassified examples explorer
- ✅ Streamlit app: upload your own digit image OR pick an MNIST sample
- ✅ Full Jupyter Notebook for step-by-step walkthrough
