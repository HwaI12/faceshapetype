import os
import dlib
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm
import random

# データセットのパス
dataset_path = "FaceShape Dataset"

# 顔タイプごとの画像
face_type_images = {
    "Round": os.path.join(dataset_path, "training_set", "Round"),
    "Oblong": os.path.join(dataset_path, "training_set", "Oblong"),
    "Oval": os.path.join(dataset_path, "training_set", "Oval"),
    "Square": os.path.join(dataset_path, "training_set", "Square"),
    "Heart": os.path.join(dataset_path, "training_set", "Heart")
}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 特徴量抽出の関数
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    features = np.array([[p.x, p.y] for p in landmarks.parts()]).flatten()
    return features

# データのロード
def load_data(dataset_path, num_samples=None): # num_samples引数を追加
    features = []
    labels = []
    for face_type in os.listdir(dataset_path):
        face_type_path = os.path.join(dataset_path, face_type)
        if os.path.isdir(face_type_path):
            image_files = [f for f in os.listdir(face_type_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # サンプリング処理を追加
            if num_samples is not None and len(image_files) > num_samples:
                image_files = random.sample(image_files, num_samples)

            for filename in tqdm(image_files, desc=f"Loading {face_type} images"):
                image_path = os.path.join(face_type_path, filename)
                image = cv2.imread(image_path)

                # 特徴量を抽出
                feature = extract_features(image)

                # Noneをスキップ
                if feature is not None:
                    features.append(feature)
                    labels.append(face_type)
                else:
                    print(f"Skipped: {image_path} (No face detected)")

    # 最終的なデータの整形
    return np.array(features), np.array(labels)

# サンプル数を指定してデータロード (例: 各タイプ100枚に縮小)
num_samples_per_type = 100 # 各タイプ100枚に制限
training_features, training_labels = load_data(os.path.join(dataset_path, "training_set"), num_samples_per_type)
testing_features, testing_labels = load_data(os.path.join(dataset_path, "testing_set"), num_samples_per_type)

print(f"Number of training samples: {len(training_features)}")
print(f"Number of testing samples: {len(testing_features)}")

# SVMモデルを学習
model = SVC(kernel='linear')

# tqdmを使って進捗表示
print("Training the SVM model...")
model.fit(training_features, training_labels)

# モデルの評価
y_pred = model.predict(testing_features)
accuracy = accuracy_score(testing_labels, y_pred)
print(f"Accuracy: {accuracy}")

# 学習したモデルを保存
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Model saved as svm_model.pkl")
