import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
from tqdm import tqdm
from collections import Counter
import random
import torchvision.models as models
from torchvision.models import ResNet152_Weights

# ResNet model for feature extraction
def get_resnet_model():
    # model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
    # model = models.resnet152(weights=None)
    model.fc = nn.Identity()
    model.eval()
    return model

# Preprocessing transformations for ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features using ResNet
model = get_resnet_model()
def extract_resnet_features(image):
    if image is None or image.size == 0:
        raise ValueError("Empty image provided")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = transforms.ToPILImage()(image_rgb)
    input_tensor = resnet_transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor).numpy().flatten()
    return features

# Load dataset and extract features
def load_data(dataset_path, num_samples=None):
    features, labels, skipped = [], [], []

    for face_type in os.listdir(dataset_path):
        face_type_path = os.path.join(dataset_path, face_type)
        if os.path.isdir(face_type_path):
            image_files = [f for f in os.listdir(face_type_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if num_samples:  # Optional subsampling
                image_files = random.sample(image_files, num_samples)

            print(f"Processing images in: {face_type_path}")

            for filename in tqdm(image_files):
                image_path = os.path.join(face_type_path, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: {image_path} could not be read.")
                    skipped.append(image_path)
                    continue

                try:
                    feature = extract_resnet_features(image)
                    features.append(feature)
                    labels.append(face_type)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    skipped.append(image_path)

    return np.array(features), np.array(labels), skipped

# Dataset paths
dataset_path = "FaceShape-Dataset"
train_path = os.path.join(dataset_path, "training_set")
test_path = os.path.join(dataset_path, "testing_set")

# Load and process data
train_features, train_labels, train_skipped = load_data(train_path)
test_features, test_labels, test_skipped = load_data(test_path)

# Feature scaling
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# PCA
pca = PCA()
pca.fit(train_features_scaled)

n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.99) + 1
pca = PCA(n_components=n_components)
train_features_pca = pca.fit_transform(train_features_scaled)
test_features_pca = pca.transform(test_features_scaled)

# SVM training
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf'],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(SVC(), param_grid, cv=cv, verbose=3, n_jobs=-1)
model.fit(train_features_pca, train_labels)

# Save the model and PCA
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model.best_estimator_, f)

with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

# Evaluation
y_pred = model.predict(test_features_pca)
print(f"Test Accuracy: {accuracy_score(test_labels, y_pred)}")
print(classification_report(test_labels, y_pred))
