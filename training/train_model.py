import librosa
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(path):
    audio, sr = librosa.load(path, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])
    return np.mean(features, axis=1)


# ----------------------------
# Dataset
# ----------------------------
class VoiceDataset(Dataset):
    def __init__(self, base_dir):
        self.samples = []

        for label, cls in enumerate(["human", "ai"]):
            cls_path = os.path.join(base_dir, cls)

            for lang in os.listdir(cls_path):
                lang_path = os.path.join(cls_path, lang)

                for file in os.listdir(lang_path):
                    if file.endswith((".wav", ".mp3")):
                        self.samples.append(
                            (os.path.join(lang_path, file), label)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        features = extract_features(path)
        return torch.tensor(features).float(), torch.tensor(label).float()


# ----------------------------
# Model
# ----------------------------
class VoiceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ----------------------------
# Training
# ----------------------------
def train():
    dataset_path = "dataset"   # <-- YOU MUST CREATE THIS
    dataset = VoiceDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VoiceDetector()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(loader):
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/voice_detector.pt")
    print("✅ Model saved to model/voice_detector.pt")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    train()
