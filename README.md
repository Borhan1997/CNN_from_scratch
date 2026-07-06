# CNN from scratch : Neural Ear — Audio Classification CNN Built From Scratch

A custom ResNet-style convolutional neural network, implemented from scratch in PyTorch (no pretrained backbone), trained to classify 50 categories of environmental sound — from animal calls to household and urban noises. The model is trained and served on [Modal](https://modal.com)'s serverless GPU infrastructure, and paired with an interactive Next.js frontend that visualizes the network's predictions, input spectrogram, waveform, and internal feature maps layer by layer.

## Features

- **Custom CNN architecture** — a ResNet-18-style network (residual blocks, 4 stages of `[3, 4, 6, 3]` blocks) written from first principles in PyTorch, not loaded from `torchvision.models`.
- **Audio-to-image pipeline** — raw waveforms are converted to log-mel spectrograms (128 mel bands) and fed to the CNN as single-channel images.
- **Trained on ESC-50** — the [ESC-50 environmental sound dataset](https://github.com/karolpiczak/ESC-50), 50 classes spanning animal, natural, human, domestic, and urban sounds.
- **Serious training recipe** — mixup augmentation, frequency/time masking, label smoothing, AdamW with a OneCycleLR schedule.
- **Serverless GPU training & inference** — both `train.py` and `main.py` run as [Modal](https://modal.com) apps, so training runs on an on-demand A10 GPU with no local GPU required, and inference is served as a FastAPI endpoint behind Modal's autoscaling.
- **Interactive visualizer (`neural-ear/`)** — a Next.js/React app where you upload a `.wav` file and see the model's top-3 predictions, the input spectrogram, the raw waveform, and the feature maps produced by every convolutional layer of the network.

## How it works

```
.wav file
   │
   ▼
Log-Mel Spectrogram  (128 mel bands, 44.1kHz)
   │
   ▼
AudioCNN  (custom ResNet-style CNN)
   ├─ conv1: 7x7 conv + BN + ReLU + maxpool
   ├─ layer1: 3 residual blocks (64 ch)
   ├─ layer2: 4 residual blocks (128 ch)
   ├─ layer3: 6 residual blocks (256 ch)
   ├─ layer4: 3 residual blocks (512 ch)
   ├─ global average pool
   └─ fully connected → 50-class softmax
   │
   ▼
Top-3 predictions + per-layer feature maps
   │
   ▼
neural-ear frontend (spectrogram / waveform / feature map visualization)
```

1. **`train.py`** downloads ESC-50 onto a Modal volume, builds mel-spectrogram datasets with augmentation, and trains `AudioCNN` on a Modal A10 GPU, checkpointing the best model (by validation accuracy) to a persistent Modal volume, with metrics logged to TensorBoard.
2. **`main.py`** loads that checkpoint and exposes a FastAPI inference endpoint (via `@modal.fastapi_endpoint`) that accepts base64-encoded audio, runs it through the model, and returns the top-3 predicted classes with confidence scores, plus the intermediate feature maps, input spectrogram, and waveform data needed for visualization.
3. **`neural-ear/`** is the frontend: upload a `.wav`, it calls the deployed inference endpoint, and renders the predictions, spectrogram, waveform, and every convolutional layer's feature maps in the browser.

## Project structure

```
CNN_from_scratch/
├── model.py            # AudioCNN + ResidualBlock — the CNN architecture
├── train.py             # Modal app: downloads ESC-50, trains the model on GPU
├── main.py              # Modal app: FastAPI inference endpoint
├── requirements.txt     # Python dependencies
└── neural-ear/          # Next.js frontend — audio upload & visualization UI
    └── src/
        ├── app/page.tsx           # main UI: upload, predictions, visualizations
        └── components/           # FeatureMap, Waveform, ColorScale
```

## Tech stack

**Backend / ML**: Python, PyTorch, torchaudio, librosa, FastAPI, [Modal](https://modal.com) (serverless GPU compute + storage volumes), pandas, TensorBoard

**Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS, shadcn/ui

## Getting started

### Prerequisites

- Python 3.10+
- A free [Modal](https://modal.com) account (for GPU training and hosting inference)
- Node.js 18+ (for the frontend)

### Train the model

```bash
pip install -r requirements.txt
modal setup                 # authenticate with your Modal account
modal run train.py          # downloads ESC-50 and trains AudioCNN on an A10 GPU
```

The best checkpoint (by validation accuracy) is saved to a Modal volume as `best_model.pth`, and training curves are logged to TensorBoard.

### Deploy the inference API

```bash
modal deploy main.py
```

This publishes a persistent HTTPS endpoint that accepts a base64-encoded `.wav` file and returns predictions plus visualization data.

### Run the frontend locally

```bash
cd neural-ear
npm install
npm run dev
```

Update the endpoint URL in `src/app/page.tsx` to point at your own deployed Modal inference URL, then open `http://localhost:3000` and upload a `.wav` file.

## Dataset & acknowledgments

- [ESC-50](https://github.com/karolpiczak/ESC-50) by Karol J. Piczak — 2,000 labeled environmental audio recordings across 50 classes, used here under its original license.
- Model training and serving powered by [Modal](https://modal.com).

## Results

Best validation accuracy: **80.25%** across the 50 ESC-50 classes, reached at epoch 79 of training.

