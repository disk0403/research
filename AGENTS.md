# AGENTS.md

## 範囲

このリポジトリは、複数プロジェクトにまたがる Go2 ガイドロボット研究を管理する。

## 構成

- `ROADMAP.md`: 研究方針、マイルストーン、安全方針、評価範囲。
- `papers/`: 共有論文ライブラリ。
- `projects/`: MuJoCo、ROS2、実機 Go2 実験、VLM ナビゲーション、HRI、ハードウェアテストなどの実装プロジェクト。

## プロジェクト方針

- MuJoCo 固有の作業は `projects/go2-mujoco/` に置く。
- 明示的に依頼されない限り、`projects/go2-mujoco/external/` は読み取り専用の実行資産として扱う。
- `projects/go2-mujoco/.venv/` はローカル環境として扱い、Git に入れない。

## 言語方針

- このリポジトリでの回答には日本語を使う。
- Markdown ファイルを書く・編集するときは日本語を使う。
- コマンド、コード、ファイル名、パッケージ名、論文の原題など、識別子として意味がある英語は編集しない。

## 更新方針

- 議論を通じて研究方針、マイルストーン、安全方針、評価範囲、ガイドロボットに関する前提が変わった場合は、`ROADMAP.md` を更新する。
- 論文を追加・議論する場合は、`papers/references.bib` を更新し、`papers/notes/` の関連メモを追加または修正する。
- ローカル PDF は `papers/pdfs/` に置く。ただし、明示的に依頼されない限り PDF は Git に入れない。
- MuJoCo 固有のコードやセットアップを変更した場合は、必要に応じて `projects/go2-mujoco/README.md` を更新する。
- ローカル環境と外部実行資産は Git に入れない。対象は `projects/go2-mujoco/.venv/`、`projects/go2-mujoco/external/`、`papers/pdfs/`、`__pycache__/`、`.DS_Store`、`._*`。

## 検証

現在の MuJoCo スモークテスト:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_teleop.py --headless --duration 0.05
```
