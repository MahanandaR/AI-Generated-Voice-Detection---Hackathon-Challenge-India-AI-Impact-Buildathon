import torch
import torch.nn as nn

MODEL_PATH = "model/voice_detector.pt"

# SAME model architecture used during training
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


# Load model safely
model = VoiceDetector()
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

def predict(features):
    tensor = torch.tensor(features).float().unsqueeze(0)
    with torch.no_grad():
        return model(tensor).item()
