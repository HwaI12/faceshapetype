import os
import dlib
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm

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
def load_data(dataset_path):
    features = []
    labels = []
    for face_type in os.listdir(dataset_path):
        face_type_path = os.path.join(dataset_path, face_type)
        if os.path.isdir(face_type_path):  # ディレクトリのみを処理
            # tqdmを使用して進捗表示
            for filename in tqdm(os.listdir(face_type_path), desc=f"Loading {face_type} images"):
                image_path = os.path.join(face_type_path, filename)
                # 画像ファイルのみを処理する（非画像ファイルをスキップ）
                if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue  # 画像が読み込めなければスキップ
                    extracted_feature = extract_features(img)
                    if extracted_feature is not None:
                        features.append(extracted_feature)
                        labels.append(face_type)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return np.array(features), np.array(labels)

# 学習データをロード
training_features, training_labels = load_data(os.path.join(dataset_path, "training_set"))
testing_features, testing_labels = load_data(os.path.join(dataset_path, "testing_set"))

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
