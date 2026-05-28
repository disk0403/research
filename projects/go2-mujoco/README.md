# Go2 MuJoCo キーボード遠隔操作

Unitree Go2 を MuJoCo 上で動かすための実験用プロジェクトです。

現在の実行スクリプトは `scripts/go2_teleop.py` の 1 本です。学習済み ONNX ポリシーを使い、平面シーン上の Go2 を WASD/QE キーで操作できます。Shift ダッシュと Space ジャンプ補助も入っています。

## クイックスタート

依存パッケージを入れます。

```bash
python3 -m pip install -r requirements.txt
```

このディレクトリにある既存のローカル仮想環境を使う場合:

```bash
source .venv/bin/activate
```

GUI で起動します。

```bash
python3 scripts/go2_teleop.py
```

表示先が `:1` ではない環境では `--display` を指定します。

```bash
python3 scripts/go2_teleop.py --display "$DISPLAY"
```

GUI なしで短時間テストする場合:

```bash
python3 scripts/go2_teleop.py --headless --duration 2 --test-command-vx 0.4
```

## 操作

```text
W            前進
S            後退
A            左方向へ横移動
D            右方向へ横移動
Q            左旋回
E            右旋回
Shift+WASD   ダッシュ
Space        押している間ジャンプ補助
Esc          終了

左ドラッグ       カメラ回転
右ドラッグ       カメラ水平移動
中央ドラッグ     カメラ垂直移動
マウスホイール   ズーム
```

キーを離すと速度指令は即座にゼロになります。キーを押している間の速度変化だけ `--command-smoothing` で平滑化されます。

## 主なパス

```text
requirements.txt
scripts/go2_teleop.py
external/unitree_mujoco/unitree_robots/go2/
external/policies/unitree-go2-velocity-flat/
.venv/
```

### `scripts/go2_teleop.py`

現在のメインスクリプトです。

- MuJoCo シーンを読み込む
- ONNX ポリシーで 12 関節の目標角を出す
- PD 制御で MuJoCo のアクチュエータへトルクを入れる
- WASD/QE 入力を速度指令へ変換する
- Shift ダッシュと Space ジャンプ補助を処理する
- GUI 表示とヘッドレス実行の両方に対応する

デフォルトでは次の平面シーンを使います。

```text
external/unitree_mujoco/unitree_robots/go2/scene_flat.xml
```

### `external/unitree_mujoco/unitree_robots/go2/`

Go2 の MuJoCo モデル資産です。現在は `scene_flat.xml`、`go2.xml`、OBJ メッシュだけを残しています。

このディレクトリは `external/unitree_mujoco` に含まれる外部リポジトリ由来の資産です。現在の `scene_flat.xml` はローカル追加ファイルです。

### `external/policies/unitree-go2-velocity-flat/`

歩行制御に使う学習済みポリシーです。

```text
policy.onnx
policy.onnx.data
params/deploy.yaml
```

`deploy.yaml` には関節 ID、PD ゲイン、デフォルト関節角、行動スケール、観測構成が入っています。

## コマンドラインオプション

よく使うオプション:

```bash
python3 scripts/go2_teleop.py --normal-speed 0.4
python3 scripts/go2_teleop.py --dash-forward-speed 1.5
python3 scripts/go2_teleop.py --yaw-speed 0.7
python3 scripts/go2_teleop.py --command-smoothing 0
python3 scripts/go2_teleop.py --render-fps 30
```

ヘッドレス確認:

```bash
python3 scripts/go2_teleop.py --headless --duration 3
python3 scripts/go2_teleop.py --headless --duration 3 --test-command-vx 0.5
python3 scripts/go2_teleop.py --headless --duration 3 --test-command-vy 0.3
python3 scripts/go2_teleop.py --headless --duration 3 --test-command-yaw 0.4
python3 scripts/go2_teleop.py --headless --duration 3 --test-jump-time 0.8 --test-jump-hold 0.35
```

シーンやポリシーの場所を変える場合:

```bash
python3 scripts/go2_teleop.py --scene path/to/scene.xml
python3 scripts/go2_teleop.py --policy-dir path/to/policy_dir
```

全オプションは次で確認できます。

```bash
python3 scripts/go2_teleop.py --help
```

## 現在のディレクトリ構成

`go2_teleop.py` の実行に必要なものだけを中心に整理しています。

```text
README.md
requirements.txt
scripts/go2_teleop.py
external/unitree_mujoco/LICENSE
external/unitree_mujoco/readme.md
external/unitree_mujoco/unitree_robots/go2/
external/policies/unitree-go2-velocity-flat/
.venv/
```

`.venv/` は GitHub に入れるものではありませんが、このローカル環境では依存パッケージ入りの実行環境として存在します。`external/` もローカル実行に必要な資産として存在しますが、研究コードとして編集する場所ではありません。

## 注意

- このスクリプトは MuJoCo 内の Go2 モデルを直接制御します。実機 Go2 へコマンドを送るものではありません。
- `external/unitree_sdk2_python/`、`external/cyclonedds/`、`mujoco_menagerie/`、wheel キャッシュ、OS メタデータ、実行ログは削除済みです。
- 学習済みポリシーを GitHub に含める場合は、配布元ライセンスを確認してください。
- `.venv/` と `external/` はローカル環境・外部実行資産として扱い、Git ではプロジェクト側のコードと文書を中心に管理します。
