# AGENTS.md

このファイルは、このリポジトリで作業する AI エージェントと開発者向けの運用指針である。リポジトリ全体に適用する。

## 目的

このリポジトリは、Unitree Go2 を中心とした四脚ガイドロボット研究を管理するための研究開発ワークスペースである。主な対象は MuJoCo シミュレーション、歩行制御、視覚・言語モデルを含むナビゲーション、HRI、安全評価、実機移行の準備である。

現段階では、Go2 を「盲導犬の代替」として扱うのではなく、視覚障害者の移動支援を研究するための四脚ガイドロボット基盤として扱う。

## ディレクトリ構成

```text
.
├── AGENTS.md
├── README.md
├── ROADMAP.md
├── papers/
│   ├── references.bib
│   ├── notes/
│   └── pdfs/
├── projects/
│   └── go2-mujoco/
│       ├── README.md
│       ├── requirements.txt
│       ├── scripts/
│       ├── external/
│       ├── logs/
│       └── .venv/
└── local/
```

各ディレクトリとファイルの役割:

- `README.md`: リポジトリ全体の概要、構成、主要プロジェクトへの入口。
- `ROADMAP.md`: 研究方針、マイルストーン、安全方針、評価範囲、直近の実装計画。
- `AGENTS.md`: このリポジトリでの作業ルール。むやみに更新しない。
- `papers/`: 共有論文ライブラリ。
- `papers/references.bib`: 論文の BibTeX メタデータ。
- `papers/notes/`: 論文ごとの Markdown メモ、要約、研究との関係。
- `papers/pdfs/`: ローカル PDF 置き場。原則 Git に入れない。
- `projects/`: 個別の実装プロジェクトを置く場所。
- `projects/go2-mujoco/`: Go2 の MuJoCo シミュレーション、歩行制御、視覚追従などの実装。
- `projects/go2-mujoco/scripts/`: 研究コードを置く場所。
- `projects/go2-mujoco/external/`: 外部由来のモデル、MuJoCo 資産、学習済みポリシーなどのローカル実行資産。原則読み取り専用。
- `projects/go2-mujoco/logs/`: 実験ログ、デバッグ画像、実行結果。Git に入れない。
- `projects/go2-mujoco/.venv/`: ローカル Python 環境。Git に入れない。
- `local/`: 雑多なメモ、試作資料、一時的な Markdown 出力、個人用作業ログを置く場所。Git に入れない。

## 作業範囲

- MuJoCo 固有のコード、設定、説明は `projects/go2-mujoco/` に置く。
- 論文メモや引用情報は `papers/` に置く。
- リポジトリ全体の方針や研究計画は `ROADMAP.md` に集約する。
- 一時的なメモ、議論整理、スライド下書き、雑多な Markdown 出力は `local/` に置く。ただし、ユーザーが明示的に共有資料として Git に入れるよう依頼した場合はこの限りではない。

## Git に入れるもの・入れないもの

Git に入れるもの:

- 研究コード。
- 再現に必要な小さな設定ファイル。
- `README.md`、`ROADMAP.md`、`AGENTS.md` などの運用・説明文書。
- 論文の BibTeX と Markdown メモ。

Git に入れないもの:

- `projects/go2-mujoco/.venv/`
- `projects/go2-mujoco/external/`
- `projects/go2-mujoco/logs/`
- `papers/pdfs/`
- `local/`
- `__pycache__/`
- `.DS_Store`
- `._*`
- `.env`
- 実行ログ、デバッグ画像、ローカルキャッシュ

外部資産やローカル環境は、必要に応じて README に取得方法や配置場所だけを書く。

## README.md の更新ルール

`README.md` はリポジトリ全体の入口として扱う。

更新するべき場合:

- 新しい主要プロジェクトを追加した。
- ディレクトリ構成が変わった。
- 初回セットアップや代表的な実行コマンドが変わった。
- 研究全体の説明として重要な入口が増えた。

更新しないでよい場合:

- 個別スクリプト内部の小さな実装変更。
- 一時的な実験結果。
- `local/` に置くべき個人メモ。

## ROADMAP.md の更新ルール

`ROADMAP.md` は研究計画と判断の記録として扱う。

更新するべき場合:

- 研究方針、マイルストーン、安全方針、評価範囲が変わった。
- ガイドロボットとしての前提や対象タスクが変わった。
- 新しい評価基盤、主要プロトタイプ、実装フェーズが追加された。
- 実機移行、安全評価、ユーザー評価に関する方針が変わった。

更新しないでよい場合:

- 単なるコード整理。
- 一時的なデバッグ。
- README に書けば十分な実行方法の微修正。

## papers/ の更新ルール

論文を追加・議論する場合は、原則として次の 2 点を更新する。

1. `papers/references.bib`
   - BibTeX エントリを追加または修正する。
   - 論文タイトル、著者、年、URL、arXiv ID、DOI などを可能な限り正確に書く。

2. `papers/notes/`
   - 論文ごとの Markdown メモを追加または更新する。
   - 研究との関係、使えるアイデア、限界、再現に必要な情報を書く。
   - 論文の原題や手法名など、識別子として意味がある英語はそのまま書く。

PDF の扱い:

- ローカル PDF は `papers/pdfs/` に置く。
- 明示的に依頼されない限り PDF は Git に入れない。
- PDF を入れたこと自体よりも、`references.bib` と `notes/` に知識を残すことを優先する。

## projects/go2-mujoco/ の更新ルール

Go2 MuJoCo プロジェクトの研究コードは `projects/go2-mujoco/scripts/` に置く。

現在の代表的なスクリプト:

- `scripts/go2_teleop.py`: 学習済み ONNX 歩行ポリシーを使い、WASD/QE で Go2 を遠隔操作する。
- `scripts/go2_vision_target_follow.py`: MuJoCo の仮想前方カメラ画像から単純な色付き対象物を検出し、低速追従する。

`projects/go2-mujoco/README.md` を更新するべき場合:

- 新しい実行スクリプトを追加した。
- 必要な依存関係やセットアップ手順が変わった。
- 代表的な実行コマンドが変わった。
- 外部資産の配置方法や前提が変わった。

`projects/go2-mujoco/external/` は、明示的に依頼されない限り読み取り専用として扱う。外部由来の MuJoCo モデル、学習済みポリシー、配布元資産を研究コードとして直接編集しない。

## ローカルメモと一時出力

雑多なメモ、調査ログ、スライド下書き、一時的な Markdown、試行錯誤の出力は `local/` に置く。

例:

```text
local/weekly_notes.md
local/slide_draft.md
local/policy_inspection.md
local/tmp_experiment_notes.md
```

ただし、ユーザーが「共有する成果物として Git に入れたい」と明示した場合は、適切な場所に移す。例として、研究計画なら `ROADMAP.md`、論文メモなら `papers/notes/`、プロジェクト説明なら各プロジェクトの `README.md` に統合する。

## 言語方針

- このリポジトリでの回答には日本語を使う。
- Markdown ファイルを書く・編集するときは日本語を使う。
- コマンド、コード、ファイル名、パッケージ名、論文の原題、モデル名、API 名など、識別子として意味がある英語は変更しない。

## 安全・研究倫理

- 実機、人間、視覚障害者ユーザーを巻き込む前に、シミュレーションで危険行動を確認・低減する。
- 現段階の成果を「盲導犬の代替」として表現しない。
- 初期対象は屋内、低速、限定環境に絞る。
- 公道、横断歩道、駅、階段、混雑環境、視覚障害者協力者を含む評価は、倫理・安全レビュー、同意、安全担当者、緊急停止手順を整えてから扱う。
- ユーザーの停止、拒否、緊急停止を常に優先する設計を前提にする。

## 検証

MuJoCo の最小スモークテスト:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_teleop.py --headless --duration 0.05
```

視覚追従スクリプトのヘッドレス確認:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_vision_target_follow.py --headless --duration 0.4 --vision-fps 5
```

コード変更後は、影響範囲に応じて README に記載されたコマンドや関連スモークテストを実行する。実行できなかった場合は、理由を明記する。

## AGENTS.md の更新方針

`AGENTS.md` はむやみに更新しない。次のように、運用ルールや構造が実際に変わった場合に必要な部分だけ修正する。

- リポジトリのディレクトリ構成が変わった。
- 新しい主要プロジェクトが追加された。
- README、ROADMAP、papers、local、外部資産の運用ルールが変わった。
- 検証コマンドや安全方針が変わった。
- AI エージェントに守らせるべき作業ルールが増えた。

単なる文章の好み、軽微な言い換え、個別タスクの一時的な事情では更新しない。
