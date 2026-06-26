# Go2 ガイドロボット研究

Unitree Go2 を使って、四脚ロボットによる移動支援を研究するためのワークスペースです。

このリポジトリでは、いきなり実機や人を巻き込むのではなく、まず MuJoCo 上で歩行制御、誘導、視覚ナビゲーション、安全評価を小さく検証します。現段階の成果は「盲導犬の代替」ではなく、低速・屋内・限定環境でのガイドロボット研究基盤として扱います。

## まず読む場所

- 研究全体の方針: `ROADMAP.md`
- MuJoCo 上の Go2 歩行・遠隔操作: `projects/go2-mujoco/README.md`
- GuideNav と MuJoCo の接続実験: `projects/GuideNav/README.md`
- 論文メモの書き方: `papers/notes/README.md`
- AI エージェント向け運用ルール: `AGENTS.md`

## リポジトリ構成

全体像を追うための主要な階層だけを示します。実行ログ、PDF、モデル重み、仮想環境などのローカル資産は Git 管理の対象外です。

```text
.
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── papers/
│   ├── references.bib
│   ├── notes/
│   └── pdfs/                 # ローカル PDF 置き場
├── projects/
│   ├── go2-mujoco/
│   │   ├── README.md
│   │   ├── scripts/          # MuJoCo 実験スクリプト
│   │   ├── external/         # 外部モデル・ポリシーなど
│   │   └── logs/             # 実行ログ
│   └── GuideNav/
│       ├── README.md
│       ├── guidenav/         # 公式 GuideNav 本体
│       ├── mujoco_sim/       # MuJoCo 接続アダプタ
│       ├── sensor/
│       └── topogen/
└── local/                    # 一時メモ・生成物・確認用出力
```

## 主要領域

### `projects/go2-mujoco/`

Go2 を MuJoCo 上で動かすための中心プロジェクトです。学習済み ONNX ポリシーによる歩行、キーボード遠隔操作、障害物回避、粗い路面での評価、仮想カメラを使った簡単な追従実験を扱います。

最小の動作確認:

```bash
cd /home/daisuke/research/projects/go2-mujoco
.venv/bin/python scripts/go2_teleop.py --headless --duration 0.05
```

### `projects/GuideNav/`

公式 GuideNav を Go2 の MuJoCo 環境とつなぐための実験プロジェクトです。教示走行の録画、topomap 作成、CosPlace による場所認識、MuJoCo GUI 上での repeat 実験をこの領域で扱います。

基本の入口:

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate
python3 mujoco_sim/run.py check
```

### `papers/`

ガイドロボット、ナビゲーション、HRI、安全評価に関する論文を整理する領域です。

- `papers/references.bib`: BibTeX メタデータ
- `papers/notes/`: 論文ごとの要約と研究への使い道
- `papers/pdfs/`: ローカル PDF 置き場。明示的に必要な場合を除き Git には入れません

### `local/`

一時的なメモ、議論整理、確認用 HTML、画像、実験途中の生成物を置くローカル領域です。研究コードや再現に必要な設定はここに置かず、対応する `projects/` 配下か `papers/` 配下へ移します。

## 運用メモ

- 外部由来のモデル、ポリシー、重いデータは `external/` や `model_weights/` に置き、取得方法だけを README に残します。
- 実行ログやデバッグ出力は `logs/` または `local/` に置きます。
- 実機、人、視覚障害者協力者を含む評価は、シミュレーション上で危険行動を十分に減らし、安全手順を整えてから扱います。
- 研究コードを変更した場合は、対象プロジェクトの README にあるスモークテストで確認します。
