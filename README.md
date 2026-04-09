# 🛡️ DeepShield — AI Deepfake Detector

> A production-ready deepfake image classifier built with TensorFlow + Streamlit.  
> Fine-tuned MobileNetV2 · Binary classification (Real vs. Fake) · Confidence scoring

---

## 📁 Project Structure

```
DeepShield/
├── app.py                  # Streamlit web application
├── train.py                # Model training script (Phase 1 + fine-tuning)
├── utils/
│   └── preprocess.py       # Image loading, normalisation, prediction decoding
├── model/
│   └── deepfake_model.h5   # Saved model (generated after training)
├── dataset/
│   ├── train/
│   │   ├── real/           # Real face images for training
│   │   └── fake/           # Deepfake images for training
│   └── val/
│       ├── real/           # Real face images for validation
│       └── fake/           # Deepfake images for validation
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Place images into the folder structure shown above.

Recommended public datasets:
- **FaceForensics++** — https://github.com/ondyari/FaceForensics
- **DFDC (Deepfake Detection Challenge)** — https://ai.facebook.com/datasets/dfdc/
- **140k Real and Fake Faces** — https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Aim for **at least 1,000 images per class** for meaningful accuracy.

### 3. Train the model
```bash
python train.py
```

Training runs in two phases:
- **Phase 1** — Trains the classification head (MobileNetV2 frozen)
- **Phase 2** — Fine-tunes the top 30 layers of MobileNetV2

The best checkpoint is saved to `model/deepfake_model.h5`.

### 4. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

```
MobileNetV2 (ImageNet pretrained, frozen)
    └── GlobalAveragePooling2D
        └── Dense(256, relu)
            └── Dropout(0.4)
                └── Dense(64, relu)
                    └── Dropout(0.2)
                        └── Dense(1, sigmoid)   ← output: P(fake)
```

- **Input size**: 224 × 224 × 3
- **Output**: Sigmoid score in [0, 1] — closer to 1.0 = likely FAKE
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam (lr=1e-4 → 1e-5 after fine-tuning)

---

## 🖥️ App Features

| Feature | Detail |
|---|---|
| Image upload | JPG, PNG, WEBP |
| Verdict | REAL ✅ or FAKE 🚨 with colour-coded card |
| Confidence bar | `st.progress()` showing detection certainty |
| Probability breakdown | Fake % and Real % side by side |
| Raw score | Model sigmoid output for transparency |

---

## 📊 Tips for Better Accuracy

1. **Balance your dataset** — equal real/fake samples per split.
2. **Use face-cropped images** — crop to face bounding box before training.
3. **Augment carefully** — avoid flips/rotations that destroy deepfake artefacts.
4. **Increase epochs** — use early stopping to find the sweet spot.
5. **Experiment with thresholds** — adjust `threshold` in `decode_prediction()` to tune precision/recall.

---

## ⚠️ Disclaimer

DeepShield is an **educational project**. No deepfake detector achieves 100% accuracy. Do not use this as the sole tool for legal or journalistic verification of media authenticity.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
