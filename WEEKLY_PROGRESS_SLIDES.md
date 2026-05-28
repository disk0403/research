# 週次進捗報告スライド構成案

想定: 4 枚。主題は `go2_teleop.py` と `go2_vision_target_follow.py` の 2 つに絞る。  
各スライドは「何を作ったか」よりも「どう動くか」が伝わる構成にする。

## スライド 1: 今週の到達点

### 載せる主メッセージ

今週は、Go2 を MuJoCo 上で動かす基本制御ループを作り、その上に簡単な視覚追従プロトタイプを追加した。

### 載せる内容

- `go2_teleop.py`: キーボード入力から速度指令を作り、学習済み歩行ポリシーで Go2 を歩かせる。
- `go2_vision_target_follow.py`: 仮想前方カメラ画像からマゼンタ色の球を検出し、対象物へ低速追従する。
- 今週の位置づけ: 「ガイドロボット用の高度な認識」ではなく、「歩行制御と視覚入力をつなぐ最小ループ」の確認。

### 載せる画像

左右 2 分割のスクリーンショットにする。

- 左: `go2_teleop.py` の MuJoCo 画面。Go2 が平面上に立っている状態。
- 右: `go2_vision_target_follow.py` の MuJoCo 画面。Go2 とマゼンタ色の球ターゲットが同時に見える状態。

撮影コマンド候補:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_teleop.py
.venv/bin/python scripts/go2_vision_target_follow.py --duration 0
```

画像に入れる注釈:

- `teleop: human command -> velocity command -> locomotion policy`
- `vision follow: camera image -> color detection -> velocity command`

## スライド 2: `go2_teleop.py` の制御構造

### 載せる主メッセージ

Go2 は直接「足先を動かす」のではなく、`vx, vy, yaw_rate` の速度指令を歩行ポリシーへ入れ、ポリシーが 12 関節の目標角を出す構造で制御している。

### 載せる内容

- Go2 の関節構成: 4 脚 x 3 関節 = 12 関節。
  - 各脚: hip abduction/adduction, thigh pitch, calf/knee pitch。
- 入力:
  - WASD: 平面速度 `vx, vy`
  - Q/E: 旋回速度 `yaw_rate`
  - Shift: 速度上限を上げる
  - Space: ジャンプ補助
- 制御周期:
  - MuJoCo は物理ステップごとに進む。
  - 歩行ポリシーは `step_dt = 0.02 s` で更新。
- 出力:
  - ONNX ポリシーが 12 関節の目標角を出す。
  - PD 制御で MuJoCo のアクチュエータへトルクを入れる。

### 載せる画像

制御フロー図を 1 枚作る。箱 5 個で十分。

```text
Keyboard
  -> velocity command [vx, vy, yaw]
  -> ONNX locomotion policy
  -> 12 target joint positions
  -> PD torque control
  -> MuJoCo Go2
```

右下に小さくコード抜粋を載せる場合:

- `projects/go2-mujoco/scripts/go2_teleop.py:159-182`
  - `update_policy()` で ONNX 推論
  - `apply_pd()` で目標関節角からトルク計算
- `projects/go2-mujoco/scripts/go2_teleop.py:872-888`
  - 速度指令の平滑化、ポリシー更新、PD 適用、`mujoco.mj_step()`

コード画像として載せるなら、特に本質的なのは `159-182`。

### 発表メモ

ここでは「Go2 の歩行を自分で一から設計した」のではなく、「学習済み歩行ポリシーを速度指令インターフェースとして使えるようにした」と説明する。

## スライド 3: 歩行ポリシーと関節制御の中身

### 載せる主メッセージ

使用している歩行ポリシーは、平地速度追従用に学習された ONNX ポリシーで、観測から 12 関節の目標角を出す。

### 載せる内容

- 使用ポリシー:
  - Hugging Face: `diasAiMaster/unitree-go2-velocity-flat`
  - URL: https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat
  - PPO / RSL-RL で平地速度追従を学習した Go2 用 locomotion policy。
- ローカル配置:
  - `projects/go2-mujoco/external/policies/unitree-go2-velocity-flat/policy.onnx`
  - `projects/go2-mujoco/external/policies/unitree-go2-velocity-flat/policy.onnx.data`
  - `projects/go2-mujoco/external/policies/unitree-go2-velocity-flat/params/deploy.yaml`
- 観測:
  - base angular velocity
  - projected gravity
  - velocity command
  - joint position / velocity
  - last action
- 行動:
  - 12 次元 action
  - `target_joint_pos = offset + scale * action`
  - 現在の設定では action scale は `0.5`

### 載せる画像

`deploy.yaml` の一部をスクリーンショットで載せる。コード全体ではなく、次の 3 箇所だけ見せる。

- `joint_ids_map`
- `step_dt: 0.02`
- `actions.JointPositionAction.scale: 0.5`
- `observations` の一覧

対象ファイル:

- `projects/go2-mujoco/external/policies/unitree-go2-velocity-flat/params/deploy.yaml`

載せる範囲の目安:

- `joint_ids_map` から `default_joint_pos` まで
- `actions.JointPositionAction` の `scale` と `offset`
- `observations` の項目名だけ

### 発表メモ

このスライドでは細かい学習手法に深入りしない。重要なのは、「上位層は速度指令だけを考えればよく、安定した足運びは policy + PD 制御に任せている」という点。

## スライド 4: 視覚追従プロトタイプ

### 載せる主メッセージ

仮想前方カメラ画像から対象物を色で検出し、画像中心からのずれを旋回指令、見かけ面積を前進速度に変換して追従した。

### 載せる内容

- 実行時に一時 XML を作成し、Go2 の `base_link` に前方固定カメラを追加。
- 平面上にマゼンタ色の球をターゲットとして配置。
- カメラ画像から RGB しきい値でマゼンタ領域を検出。
- 検出した重心位置:
  - 画像中心より左/右にあるほど旋回。
- 検出面積:
  - 小さいほど遠いので前進。
  - 大きいほど近いので減速。
- 確認結果:
  - デフォルト条件で対象物検出が継続。
  - 距離が約 `2.03 m` から `0.75 m` まで縮み、`target_reached` で停止。

### 載せる画像

1 枚に 2 要素を入れるのがわかりやすい。

- 左: GUI 画面。Go2 とマゼンタ球、上部 overlay の `detected`, `err_x`, `cmd`, `distance` が見える状態。
- 右: 仮想前方カメラの debug frame。マゼンタ球に緑の十字が重なっている画像。

debug frame の作成コマンド:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_vision_target_follow.py \
  --headless \
  --duration 2 \
  --debug-frames-dir logs/vision_frames
```

使用画像候補:

- `projects/go2-mujoco/logs/vision_frames/frame_0000.ppm`

コード抜粋を載せる場合:

- 対象物検出:
  - `projects/go2-mujoco/scripts/go2_vision_target_follow.py:280-307`
- 検出結果から速度指令を作る部分:
  - `projects/go2-mujoco/scripts/go2_vision_target_follow.py:310-335`
- メインループ:
  - `projects/go2-mujoco/scripts/go2_vision_target_follow.py:617-651`

特に本質的なのは `280-335`。色検出から速度指令までが 1 画面に収まる。

### 発表メモ

これは本格的な object detection ではない。今週の目的は、認識結果を速度指令に変換し、既存の歩行ポリシーの上で閉ループ追従できることを確認すること。

## 最後に 1 行だけ入れるまとめ

今週は、Go2 の歩行制御を「速度指令で扱える形」にし、その上に「カメラ画像から対象物へ追従する最小閉ループ」を接続した。

## 作成時の注意

- スライド本文は 1 枚あたり 3-5 bullet に抑える。
- コードを貼る場合は 10-20 行程度に切る。
- `external/` の巨大なファイル一覧は見せない。
- `go2_vision_target_follow.py` は実機カメラではなく MuJoCo 仮想カメラであることを明記する。
- 「盲導犬の代替」ではなく「四脚ガイドロボット研究のための基礎制御」と表現する。
