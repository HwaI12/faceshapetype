import streamlit as st
import pickle
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# 顔タイプとおすすめのメガネ
face_type_glasses = {
    "Round": {
        "特徴": "頬から顎にかけて丸みを帯び、優しく若々しい印象。",
        "似合うメガネ": ["スクエア型", "ウェリントン型", "フォックス型"],
        "ポイント": "横幅が狭いフレームを選ぶと、顔の丸みが強調されるため、顔幅と同じくらいの横幅のものを選ぶと良い。"
    },
    "Oblong": {
        "特徴": "顔の縦の長さが目立ち、大人っぽく落ち着いた印象。",
        "似合うメガネ": ["ボストン型", "オーバル型", "ウェリントン型"],
        "ポイント": "縦幅の狭いフレームを選ぶと、顔の長さが強調されるため、縦幅のあるものを選ぶと良い。"
    },
    "Oval": {
        "特徴": "バランスの取れた理想的な輪郭で、どんなメガネでも似合う。",
        "似合うメガネ": ["オーバル型", "スクエア型", "ボストン型"],
        "ポイント": "フレームの大きさや色、素材などで個性を出すと良い。"
    },
    "Square": {
        "特徴": "顎のラインが角ばっており、知的で力強い印象。",
        "似合うメガネ": ["ラウンド型", "オーバル型", "ボストン型"],
        "ポイント": "直線的なフレームは避け、曲線的なフレームを選ぶと良い。"
    },
    "Heart": {
        "特徴": "顎が細く、逆三角形に近い輪郭で可愛らしい印象。",
        "似合うメガネ": ["ボストン型", "ウェリントン型", "オーバル型"],
        "ポイント": "上部にボリュームのあるフレームは避け、下部にボリュームがあるものを選ぶと良い。"
    }
}

# ResNet model for feature extraction (same as training script)
def get_resnet_model():
    model = models.resnet152(weights='DEFAULT') # Use string for weights
    model.fc = nn.Identity()
    model.eval()
    return model

# Preprocessing transformations for ResNet (same as training script)
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features using ResNet (same as training script)
model_resnet = get_resnet_model() # Instantiate the model here
def extract_features(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = transforms.ToPILImage()(image_rgb)
        input_tensor = resnet_transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            features = model_resnet(input_tensor).numpy().flatten()
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# モデルとPCAのロード
try:
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
except FileNotFoundError:
    st.error("必要なモデルファイルが見つかりませんでした。")
    st.stop()  # Stop execution if files are missing

st.title("顔タイプ診断アプリ")
uploaded_image = st.file_uploader("顔画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="アップロードした画像", use_container_width=True)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            st.error("顔が検出できませんでした。別の画像をお試しください。")
        else:
            x, y, w, h = faces[0]
            face_image = opencv_image[y:y+h, x:x+w]

            # Resize to 224x224 for ResNet input
            face_image_resized = cv2.resize(face_image, (224, 224)) #Correct resize size

            extracted_feature = extract_features(face_image_resized)

            if extracted_feature is None:
                st.error("顔の特徴量を抽出できませんでした。")
            else:
                extracted_feature_pca = pca.transform(extracted_feature.reshape(1, -1))
                face_type = model.predict(extracted_feature_pca)[0]

                st.subheader(f"あなたの顔タイプ: {face_type}")
                st.write(f"おすすめのメガネデザイン: {face_type_glasses[face_type]['似合うメガネ']}")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")