"""
AI-Based Handwritten Digit Recognition System
CNN Model Training Script
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess():
    """Load MNIST and preprocess for CNN input."""
    print("📦 Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Test samples     : {X_test.shape[0]}")
    print(f"   Image shape      : {X_train.shape[1:]}")

    # Normalize pixel values: 0–255 → 0.0–1.0
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Reshape for CNN: (samples, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat  = to_categorical(y_test,  num_classes=10)

    print("✅ Preprocessing complete.\n")
    return (X_train, y_train, y_train_cat), (X_test, y_test, y_test_cat)


# ─────────────────────────────────────────────
# 2. MODEL ARCHITECTURES
# ─────────────────────────────────────────────

def build_cnn():
    """Build Convolutional Neural Network."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(28, 28, 1), name='conv1'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', name='output'),
    ], name='CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_mlp():
    """Build Multilayer Perceptron for comparison."""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ], name='MLP')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train_cat, X_test, y_test_cat,
                epochs=15, batch_size=128):
    """Train model with callbacks."""
    cb_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=2, verbose=1),
    ]

    history = model.fit(
        X_train, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=cb_list,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n🎯 Test Accuracy : {test_acc*100:.2f}%")
    print(f"   Test Loss     : {test_loss:.4f}")

    return history, test_loss, test_acc


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_sample_digits(X_train, y_train, save_path="assets/sample_digits.png"):
    """Visualize sample digits from dataset."""
    os.makedirs("assets", exist_ok=True)
    fig, axes = plt.subplots(4, 10, figsize=(16, 7))
    fig.patch.set_facecolor('#0f0f1a')

    for digit in range(10):
        idxs = np.where(y_train == digit)[0][:4]
        for row, idx in enumerate(idxs):
            ax = axes[row, digit]
            ax.imshow(X_train[idx].reshape(28, 28), cmap='plasma')
            ax.axis('off')
            if row == 0:
                ax.set_title(str(digit), color='white', fontsize=14, fontweight='bold')

    plt.suptitle('MNIST Dataset — Sample Digits (0–9)', color='white',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"   Saved → {save_path}")


def plot_training_history(histories, labels, save_path="assets/training_curves.png"):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    colors = ['#00e5ff', '#ff6b6b', '#69ff47', '#ffbe0b']

    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaaaaa')
        ax.spines[:].set_color('#333355')

    for i, (hist, label) in enumerate(zip(histories, labels)):
        c = colors[i]
        axes[0].plot(hist.history['accuracy'],     color=c, lw=2, label=f'{label} Train')
        axes[0].plot(hist.history['val_accuracy'], color=c, lw=2, ls='--', alpha=0.7,
                     label=f'{label} Val')
        axes[1].plot(hist.history['loss'],     color=c, lw=2, label=f'{label} Train')
        axes[1].plot(hist.history['val_loss'], color=c, lw=2, ls='--', alpha=0.7,
                     label=f'{label} Val')

    for ax, title in zip(axes, ['Model Accuracy', 'Model Loss']):
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', color='#aaaaaa')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        ax.grid(color='#333355', linestyle='--', alpha=0.5)

    plt.suptitle('Training Curves', color='white', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"   Saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name="CNN",
                          save_path="assets/confusion_matrix.png"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10),
                ax=ax, linewidths=0.5, linecolor='#0f0f1a',
                annot_kws={"size": 11, "color": "white"})

    ax.set_xlabel('Predicted Label', color='white', fontsize=13)
    ax.set_ylabel('True Label',      color='white', fontsize=13)
    ax.set_title(f'{model_name} — Confusion Matrix', color='white',
                 fontsize=15, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"   Saved → {save_path}")


def plot_predictions(model, X_test, y_test, n=20,
                     save_path="assets/predictions.png"):
    """Show model predictions with confidence."""
    idxs   = np.random.choice(len(X_test), n, replace=False)
    images = X_test[idxs]
    labels = y_test[idxs]
    preds  = model.predict(images, verbose=0)

    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    fig.patch.set_facecolor('#0f0f1a')

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='plasma')
        ax.axis('off')
        pred  = np.argmax(preds[i])
        conf  = preds[i][pred] * 100
        color = '#69ff47' if pred == labels[i] else '#ff6b6b'
        ax.set_title(f'True: {labels[i]}  Pred: {pred}\n{conf:.1f}%',
                     color=color, fontsize=10, fontweight='bold')

    plt.suptitle('Model Predictions (Green = Correct, Red = Wrong)',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"   Saved → {save_path}")


def plot_misclassified(model, X_test, y_test, n=12,
                       save_path="assets/misclassified.png"):
    """Show misclassified examples."""
    preds_prob = model.predict(X_test, verbose=0)
    preds      = np.argmax(preds_prob, axis=1)
    wrong_idxs = np.where(preds != y_test)[0][:n]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')

    for i, idx in enumerate(wrong_idxs):
        ax   = axes[i // 4, i % 4]
        conf = preds_prob[idx][preds[idx]] * 100
        ax.imshow(X_test[idx].reshape(28, 28), cmap='hot')
        ax.axis('off')
        ax.set_title(f'True: {y_test[idx]}  Pred: {preds[idx]}\n{conf:.1f}%',
                     color='#ff6b6b', fontsize=10, fontweight='bold')

    plt.suptitle('Misclassified Examples', color='white',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"   Saved → {save_path}")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs("assets",  exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # Load data
    (X_train, y_train, y_train_cat), (X_test, y_test, y_test_cat) = load_and_preprocess()

    # Visualize samples
    print("🖼️  Generating sample digit plot...")
    plot_sample_digits(X_train, y_train)

    # ── CNN ─────────────────────────────────
    print("\n🧠 Building CNN model...")
    cnn = build_cnn()
    cnn.summary()

    print("\n🚀 Training CNN...")
    hist_cnn, cnn_loss, cnn_acc = train_model(cnn, X_train, y_train_cat,
                                               X_test, y_test_cat, epochs=15)

    cnn.save("models/cnn_mnist.keras")
    print("💾 CNN model saved → models/cnn_mnist.keras")

    # ── MLP ─────────────────────────────────
    print("\n🧠 Building MLP model...")
    mlp = build_mlp()

    print("\n🚀 Training MLP...")
    hist_mlp, mlp_loss, mlp_acc = train_model(mlp, X_train, y_train_cat,
                                               X_test, y_test_cat, epochs=15)

    mlp.save("models/mlp_mnist.keras")
    print("💾 MLP model saved → models/mlp_mnist.keras")

    # ── Visualizations ──────────────────────
    print("\n📊 Generating visualizations...")
    plot_training_history([hist_cnn, hist_mlp], ['CNN', 'MLP'])

    y_pred_cnn = np.argmax(cnn.predict(X_test, verbose=0), axis=1)
    y_pred_mlp = np.argmax(mlp.predict(X_test, verbose=0), axis=1)

    plot_confusion_matrix(y_test, y_pred_cnn, 'CNN', 'assets/cm_cnn.png')
    plot_confusion_matrix(y_test, y_pred_mlp, 'MLP', 'assets/cm_mlp.png')
    plot_predictions(cnn, X_test, y_test)
    plot_misclassified(cnn, X_test, y_test)

    # ── Reports ─────────────────────────────
    print("\n📄 Classification Report — CNN:")
    print(classification_report(y_test, y_pred_cnn, digits=4))

    print("📄 Classification Report — MLP:")
    print(classification_report(y_test, y_pred_mlp, digits=4))

    # Save metrics summary
    metrics = {
        "CNN": {"test_accuracy": float(cnn_acc), "test_loss": float(cnn_loss)},
        "MLP": {"test_accuracy": float(mlp_acc), "test_loss": float(mlp_loss)},
    }
    with open("assets/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ All done!")
    print(f"   CNN Accuracy : {cnn_acc*100:.2f}%")
    print(f"   MLP Accuracy : {mlp_acc*100:.2f}%")


if __name__ == "__main__":
    main()
