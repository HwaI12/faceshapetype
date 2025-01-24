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

## レポート
[モデルトレーニングレポート](./reports/model_training_report.md)