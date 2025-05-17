import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from typing import Optional, Tuple

MODEL_PATH = 'face_verifier_model.pth'
IMG_SIZE = (160, 160)
THRESHOLD = 0.7

class FaceNet(nn.Module):
    """Сверточная сеть для получения эмбеддингов лица."""
    def __init__(self, embedding_size: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

class FaceVerifier:
    """Класс для верификации лиц по эмбеддингам."""
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FaceNet().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, img: Image.Image) -> Optional[torch.Tensor]:
        try:
            face = self.transform(img).unsqueeze(0).to(self.device)
            return face
        except Exception as e:
            print(f"Ошибка обработки изображения: {str(e)}")
            return None

    def get_embedding(self, img: Image.Image) -> Optional[np.ndarray]:
        face = self.preprocess_image(img)
        if face is None:
            return None
        with torch.no_grad():
            embedding = self.model(face)
        return embedding.cpu().numpy()

    def verify(self, img1: Image.Image, img2: Image.Image, threshold: float = THRESHOLD) -> Tuple[bool, float]:
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        if emb1 is None or emb2 is None:
            return False, 0.0
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity > threshold, similarity

@st.cache_resource
def load_verifier() -> FaceVerifier:
    return FaceVerifier(model_path=MODEL_PATH)

def load_image(label: str) -> Optional[Image.Image]:
    uploaded_file = st.file_uploader(label=label, type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=label, use_column_width=True)
        return image
    return None

st.title('Верификация лиц (Face Verification)')
st.write('Загрузите две фотографии для сравнения, чтобы определить, один ли это человек.')

img1 = load_image('Фото 1')
img2 = load_image('Фото 2')

if img1 and img2:
    if st.button('Сравнить лица'):
        verifier = load_verifier()
        is_same, similarity = verifier.verify(img1, img2)
        st.write(f'**Схожесть:** {similarity:.4f}')
        st.write(f'**Вердикт:** {"Один человек" if is_same else "Разные люди"}')
