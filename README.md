# Face Shape Type

## 概要
顔の形状を分類するアプリケーション。

## 必要条件
- データセット: [Face Shape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset)
<!-- - 顔ランドマークファイル: `shape_predictor_68_face_landmarks.dat` -->

## インストール
```bash
pip install -r requirements.txt
```

## アプリケーションの実行
```bash
streamlit run app.py
```

## 機能
顔ランドマーク検出を利用した顔の形状分類

## 正答率
Test Accuracy: 0.448
```
              precision    recall  f1-score   support

       Heart       0.45      0.43      0.44       200
      Oblong       0.45      0.52      0.48       200
        Oval       0.44      0.35      0.39       200
       Round       0.48      0.56      0.52       200
      Square       0.41      0.38      0.39       200

    accuracy                           0.45      1000
   macro avg       0.45      0.45      0.44      1000
weighted avg       0.45      0.45      0.44      1000
```