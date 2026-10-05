# Smart Annotator

GitHub Pages上で動かすことを想定した、ブラウザ完結型の軽量アノテーションツールです。

## 現在の機能

- 複数画像のドラッグ&ドロップ / ファイル選択
- クラス追加・切り替え
- Bounding Box
- Polygon
- 選択・削除・Undo
- Project JSON出力
- COCO JSON出力
- YOLO形式ZIP出力
- 画像データはサーバーへ送信せず、ブラウザ内で処理

## 次に追加する予定

- SAM系モデルによるクリック指定→自動セグメンテーション
- WebGPU / ONNX Runtime Web
- 自動候補生成
- 信頼度の低い画像だけを優先確認するActive Learning的ワークフロー
- Project JSON再読込

## GitHub Pages

Repository Settings → Pages で `main` / `root` を公開元にすると利用できます。
