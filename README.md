# Face Shape Type

## 概要
顔の形状を分類するアプリケーション。

## 実行手順
- データセット[Face Shape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset)ディレクトリに配置。
- 環境のインストール`pip install -r requirements.txt`
- `python script.py`を実行。
- 学習済みモデルが`svm_model.pkl`と`pca_model.pkl`に保存されます。

## 機能
- 特徴抽出: ResNet-152を用いて顔画像の高次特徴を抽出
- 分類: PCAで次元削減し、SVMで顔タイプを分類
- モデル保存: トレーニング済みSVMモデルとPCAを保存

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

## streamlitにデプロイ
https://hwai12-faceshapetype-app-yaeveg.streamlit.app/