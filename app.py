import streamlit as st
import pickle
from PIL import Image
import cv2
import numpy as np
from svm_model import detector, predictor

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

# 保存したSVMモデルを読み込む
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("顔タイプ診断アプリ")
st.write("顔画像をアップロードしてください")

uploaded_image = st.file_uploader("顔画像をアップロード", type=["jpg", "jpeg", "png"])

# 特徴量抽出の関数
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    features = np.array([[p.x, p.y] for p in landmarks.parts()]).flatten()
    return features

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="アップロードした画像", use_container_width=True)

    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    extracted_feature = extract_features(opencv_image)

    if extracted_feature is not None:
        face_type = model.predict(extracted_feature.reshape(1, -1))[0]  # 予測
        st.subheader(f"あなたの顔タイプ: {face_type}")
        st.write(f"おすすめのメガネデザイン: {face_type_glasses[face_type]['似合うメガネ']}")
    else:
        st.write("顔が検出できませんでした。別の画像を試してください。")
