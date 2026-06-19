# Go2 ガイドロボット研究

Unitree Go2 をガイドロボット研究プラットフォームとして発展させるための研究ワークスペース。

## 現在の構成

```text
AGENTS.md                         Codex エージェント向け指示
.gitignore                        Git に入れないローカル資産の設定
ROADMAP.md                        研究ロードマップ
papers/references.bib             共有 BibTeX データベース
papers/notes/                     論文ごとの Markdown メモ
papers/pdfs/                      ローカル PDF 置き場。現在は .gitkeep のみ
projects/go2-mujoco/README.md     MuJoCo プロジェクトの説明
projects/go2-mujoco/requirements.txt
projects/go2-mujoco/scripts/go2_teleop.py
projects/go2-mujoco/scripts/go2_obstacle_avoidance_teleop.py
projects/go2-mujoco/scripts/go2_vision_target_follow.py
projects/go2-mujoco/scripts/go2_continuous_rough_terrain_teleop.py
projects/go2-mujoco/scripts/evaluate_locomotion_viewer.py
projects/go2-mujoco/external/     ローカル実行資産。読み取り専用として扱う
projects/go2-mujoco/.venv/        ローカル Python 環境
```

## 論文管理

- `papers/references.bib`: BibTeX データベース
- `papers/notes/`: 論文ごとの Markdown メモ
- `papers/pdfs/`: ローカル PDF 置き場。明示的に依頼されない限り PDF は Git に入れない

## プロジェクト

### `projects/go2-mujoco/`

Go2 の遠隔操作と、将来のガイドロボットシミュレーション実験に向けた MuJoCo プロジェクト。

現在のスモークテスト:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_teleop.py --headless --duration 0.05
```
