
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# CNN 구조 정의
class BaseCNN(nn.Module):
    def __init__(self, channels, dropout, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(channels[2], channels[3], 3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[3], 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 클래스 이름 로드
CLASS_NAMES = sorted(os.listdir('./food_dataset/train'))

# 모델 로딩 함수
@st.cache_resource
def load_ensemble_models():
    configs = {
        'A': ([64, 128, 256, 512], 0.4),
        'B': ([96, 192, 384, 768], 0.5),
        'C': ([48, 96, 192, 384], 0.3)
    }
    models = []
    for name, (channels, dropout) in configs.items():
        model = BaseCNN(channels, dropout, len(CLASS_NAMES))
        path = f"saved_models/model_{name}/best_model.pth"
        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
    return models

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit 앱
st.title("🍱 음식 이미지 앙상블 분류기")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write("예측 중...")

    input_tensor = transform(image).unsqueeze(0)
    models = load_ensemble_models()

    with torch.no_grad():
        probs = []
        for model in models:
            out = F.softmax(model(input_tensor), dim=1)
            probs.append(out)
        avg_probs = torch.stack(probs).mean(dim=0)
        top_prob, top_idx = avg_probs.topk(1)

    pred_label = CLASS_NAMES[top_idx.item()]
    confidence = top_prob.item() * 100
    st.success(f"📌 앙상블 예측 결과: **{pred_label}** ({confidence:.2f}%)")
