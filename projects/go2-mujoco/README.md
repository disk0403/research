# Go2 MuJoCo キーボード遠隔操作

Unitree Go2 を MuJoCo 上で動かすための実験用プロジェクトです。

基本の実行スクリプトは `scripts/go2_teleop.py` です。学習済み ONNX ポリシーを使い、平面シーン上の Go2 を WASD/QE キーで操作できます。Shift ダッシュと Space ジャンプ補助も入っています。

追加の実験スクリプトとして `scripts/go2_vision_target_follow.py` があります。MuJoCo 内で Go2 の前方カメラ画像を描画し、マゼンタ色の球を簡単な色しきい値で検出して、対象物へ低速追従します。

`scripts/go2_continuous_rough_terrain_teleop.py` では、隙間や段差のない連続的な凹凸路面上で、`go2_teleop.py` と同じキー操作を試せます。

`scripts/evaluate_locomotion_viewer.py` では、連続的な凹凸路面上で既存ポリシーに一定の速度指令を自動入力し、複数エピソードを GUI で観察しながら JSONL ログへ結果を保存できます。

`scripts/go2_obstacle_avoidance_teleop.py` では、平面上に直方体障害物を配置し、`go2_teleop.py` と同じキー操作に raycast ベースの簡単な障害物回避を重ねて試せます。

## クイックスタート

SSH接続を行う。

```bash
ssh 100.85.29.86
```

Go2 MuJoCo プロジェクトへ移動し、仮想環境を有効化する。

```bash
cd /home/daisuke/research/projects/go2-mujoco
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

障害物回避付きで遠隔操作する場合:

```bash
python3 scripts/go2_obstacle_avoidance_teleop.py
```

GUI なしで短く確認する場合:

```bash
python3 scripts/go2_obstacle_avoidance_teleop.py \
  --headless \
  --duration 5 \
  --test-command-vx 0.35
```

障害物検知距離や停止距離を変える場合:

```bash
python3 scripts/go2_obstacle_avoidance_teleop.py \
  --sensor-range 1.8 \
  --avoid-distance 1.2 \
  --stop-distance 0.45
```

カメラ画像から対象物を検出して追従するプロトタイプ:

```bash
python3 scripts/go2_vision_target_follow.py --duration 8
```

このスクリプトは GUI ウィンドウを開き、MuJoCo 内の Go2 とターゲットを表示します。実機カメラではなく、MuJoCo の仮想前方カメラ画像を対象物検出に使います。外部モデル資産は編集せず、実行時に一時 XML を作ってカメラとマゼンタ色の球ターゲットを追加します。

GUI なしで実行する場合:

```bash
python3 scripts/go2_vision_target_follow.py --headless --duration 8
```

GUI を閉じるまで動かす場合:

```bash
python3 scripts/go2_vision_target_follow.py --duration 0
```

カメラ画像を確認したい場合は PPM 画像を書き出せます。

```bash
python3 scripts/go2_vision_target_follow.py --headless --duration 2 --debug-frames-dir logs/vision_frames
```

連続的な凹凸路面上で遠隔操作する場合:

```bash
python3 scripts/go2_continuous_rough_terrain_teleop.py
```

GUI なしで短く動作確認する場合:

```bash
python3 scripts/go2_continuous_rough_terrain_teleop.py \
  --headless \
  --duration 0.4 \
  --test-command-vx 0.3
```

凹凸路面上の自動評価を GUI で見る場合:

```bash
python3 scripts/evaluate_locomotion_viewer.py \
  --episodes 5 \
  --duration 5 \
  --terrain-amplitude 0.04 \
  --test-command-vx 0.3
```

GUI なしで短く動作確認する場合:

```bash
python3 scripts/evaluate_locomotion_viewer.py \
  --headless \
  --episodes 1 \
  --duration 0.2 \
  --goal-distance 0.01 \
  --test-command-vx 0.3
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
Space        単押しでジャンプ補助
R            転倒時・任意タイミングで初期姿勢へリセット
Z            stance crouch を増やして少し低い姿勢にする
X            stance crouch を減らして少し高い姿勢に戻す
Esc          終了

左ドラッグ       カメラ回転
右ドラッグ       カメラ水平移動
中央ドラッグ     カメラ垂直移動
マウスホイール   ズーム
```

キーを離すと速度指令は即座にゼロになります。キーを押している間の速度変化だけ `--command-smoothing` で平滑化されます。

旋回について:

- デフォルトの `--yaw-speed` は `0.5` rad/s。
- `--yaw-safety-limit 1.0` で、ポリシー設定上の yaw コマンド範囲に合わせて上限をかけている。
- 以前の実装では MuJoCo の free joint 角速度をさらに body frame へ回していたため、歩行中旋回で姿勢が崩れやすかった。現在はポリシー観測へ `qvel[3:6]` をそのまま渡す。

姿勢調整:

- デフォルトでは `--reset-base-height 0.25`、`--stance-crouch 0.08` を使い、以前より少し低い姿勢で開始します。
- `--stance-crouch` は腿関節を少し曲げ、膝を少し深くする関節目標バイアスです。値を大きくすると低姿勢になります。
- 実行中は `Z` で低く、`X` で高く調整できます。現在値は画面内オーバーレイに表示されます。
- `R` で任意タイミングに初期姿勢へ戻せます。転倒判定時はデフォルトで自動リセットします。
- 自動リセットを止める場合は `--no-auto-reset-on-fall` を使います。

ジャンプ補助:

- 静止ジャンプでは pitch 角と pitch 角速度を見て、前脚と後脚の伸展量を補正します。
- 移動中のジャンプでは進行方向と足先の接地状態を見て、接地している押し出し脚と着地準備脚を分けます。
- ジャンプ終了後は短いリカバリ区間を挟み、ポリシー目標へ急に戻らないようにしています。
- 入力ゼロで接地しているときだけ、弱い停止時ダンピングを加えて微振動が続きにくいようにしています。

## 主なパス

```text
requirements.txt
scripts/go2_teleop.py
scripts/go2_obstacle_avoidance_teleop.py
scripts/go2_vision_target_follow.py
scripts/go2_continuous_rough_terrain_teleop.py
scripts/evaluate_locomotion_viewer.py
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
- Shift ダッシュと Space 単押しジャンプ補助を処理する
- R で初期姿勢へリセットし、転倒時は自動リセットできる
- `--stance-crouch` や実行中の Z/X キーで低重心寄りの姿勢を調整できる
- GUI 表示とヘッドレス実行の両方に対応する

デフォルトでは次の平面シーンを使います。

```text
external/unitree_mujoco/unitree_robots/go2/scene_flat.xml
```

### `scripts/go2_obstacle_avoidance_teleop.py`

平面上に直方体障害物を置き、`go2_teleop.py` と同じ操作に簡単な障害物回避を追加する実験スクリプトです。

- 実行時に一時 XML を作り、Go2 と複数の直方体障害物を配置する
- Go2 の前方に複数の水平 raycast を飛ばし、近い障害物までの距離を見る
- 前進中に中央前方が塞がっている場合、前進速度を落とし、空いている側へ yaw 指令と少しの横移動指令を加える
- `--sensor-range`、`--avoid-distance`、`--stop-distance`、`--ray-count`、`--ray-angle-span-deg` で検知と回避の挙動を調整できる
- `--max-avoid-yaw-rate` と `--avoid-lateral-speed` で回避時の旋回量と横移動量を調整できる
- `--obstacle X,Y,Z,HALF_X,HALF_Y,HALF_Z` を繰り返して任意の直方体障害物を追加できる
- `--disable-avoidance` を使うと、障害物は残したまま通常 teleop と同じ入力で比較できる

実行例:

```bash
python3 scripts/go2_obstacle_avoidance_teleop.py
python3 scripts/go2_obstacle_avoidance_teleop.py --sensor-range 1.8 --avoid-distance 1.2 --stop-distance 0.45
python3 scripts/go2_obstacle_avoidance_teleop.py --obstacle 2.5,0.2,0.25,0.2,0.4,0.25
python3 scripts/go2_obstacle_avoidance_teleop.py --headless --duration 5 --test-command-vx 0.35
```

### `scripts/go2_vision_target_follow.py`

仮想カメラ画像から単純な対象物を検出して追従する実験スクリプトです。

- 実行時に一時ディレクトリへ Go2 XML を生成する
- `base_link` に前向き固定カメラを追加する
- 平面上にマゼンタ色の球ターゲットを置く
- MuJoCo のオフスクリーンレンダリング画像からターゲットを色しきい値で検出する
- GUI では三人称視点で Go2 とターゲットを表示する
- 画像上の左右ずれから旋回指令、見かけ面積から前進指令を出す
- 対象物へ近づいた、転倒姿勢になった、制限時間に達した、などの理由で停止する

実行例:

```bash
python3 scripts/go2_vision_target_follow.py --duration 8
python3 scripts/go2_vision_target_follow.py --target-x 2.5 --target-y -0.4 --duration 10
python3 scripts/go2_vision_target_follow.py --headless --duration 8
python3 scripts/go2_vision_target_follow.py --headless --duration 2 --debug-frames-dir logs/vision_frames
```

ヘッドレス実行では `MUJOCO_GL=egl` が必要になる場合があります。このスクリプトは `--headless` 指定時に、未設定なら `MUJOCO_GL=egl` を既定値として使います。GUI 実行時は `--display` で表示先を指定できます。

### `scripts/go2_continuous_rough_terrain_teleop.py`

連続的な凹凸路面上で、既存の ONNX 歩行ポリシーを手動確認する実験スクリプトです。

- 実行時に一時 XML を作り、MuJoCo の `hfield` 路面を追加する
- デフォルトでは毎回ランダムな `terrain_seed` を使う
- 路面の頂点を隣接パッチ間で共有し、隙間や独立した段差がない連続面にする
- 局所的な傾きがランダムに変わるよう、共有頂点の高さを平滑化しつつランダム生成する
- 開始地点の周囲だけ平らにし、その外側で徐々に凹凸を強くする
- `go2_teleop.py` と同じ WASD/QE、Shift ダッシュ、Space 単押しジャンプ補助を使う
- GUI 表示とヘッドレス確認の両方に対応する

GUI で遠隔操作:

```bash
python3 scripts/go2_continuous_rough_terrain_teleop.py
```

路面の形状を変える場合:

```bash
python3 scripts/go2_continuous_rough_terrain_teleop.py --terrain-seed random
python3 scripts/go2_continuous_rough_terrain_teleop.py --terrain-seed 12
python3 scripts/go2_continuous_rough_terrain_teleop.py --terrain-amplitude 0.08
python3 scripts/go2_continuous_rough_terrain_teleop.py --terrain-smoothing-passes 1
```

ヘッドレス確認:

```bash
python3 scripts/go2_continuous_rough_terrain_teleop.py \
  --headless \
  --duration 2 \
  --test-command-vx 0.3
```

### `scripts/evaluate_locomotion_viewer.py`

連続的な凹凸路面上で、既存の ONNX 歩行ポリシーを複数エピソード自動評価するスクリプトです。手動操作ではなく、一定の `vx, vy, yaw_rate` 指令を入れて、GUI 上で挙動と評価値を見ながら各エピソードの結果を JSONL に保存します。

- デフォルトではエピソードごとにランダムな `terrain_seed` を使う
- `--terrain-seed 20` のように数値指定した場合だけ、再現可能な seed 列にする
- GUI ではゴールラインをシアン色の線と両端の柱で表示する
- 速度指令、地形 seed、ゴールラインまでの進捗、経路長、最小相対ベース高さ、最小直立度を GUI に表示する
- デフォルトでは `--test-command-vx * --duration * --goal-progress-ratio` をゴール距離にする
- `--goal-progress-ratio` のデフォルトは `0.75` で、設定速度の 75% 程度で進めれば到達できるラインにする
- 評価用の路面はゴールラインのさらに 2m 先まで自動で確保し、横幅も `--test-command-vy * --duration` に応じて広げる
- 開始位置からゴール距離だけ x 方向に進んだら成功としてエピソードを終了する
- `--duration` まで転ばなくても、ゴールラインに届かなければ `timeout` とする
- 地形表面からの相対ベース高さ、または直立度がしきい値を下回ったら転倒としてエピソードを終了する
- 結果を `logs/locomotion_eval/locomotion_view_YYYYMMDD_HHMMSS.jsonl` に保存する
- 長期集計用に `logs/locomotion_eval/locomotion_results.csv` へ 1 エピソード 1 行で追記する
- `logs/locomotion_eval/locomotion_results.md` に日本語の成功率、平均値、条件別サマリ、最近の実行を再生成する
- `--headless` で GUI なしの短時間確認にも対応する

GUI で自動評価:

```bash
python3 scripts/evaluate_locomotion_viewer.py \
  --episodes 10 \
  --duration 5 \
  --terrain-amplitude 0.04 \
  --test-command-vx 0.3
```

速度や地形を変える場合:

```bash
python3 scripts/evaluate_locomotion_viewer.py --test-command-vx 0.4
python3 scripts/evaluate_locomotion_viewer.py --goal-progress-ratio 0.7
python3 scripts/evaluate_locomotion_viewer.py --goal-distance 1.5
python3 scripts/evaluate_locomotion_viewer.py --terrain-amplitude 0.08
python3 scripts/evaluate_locomotion_viewer.py --terrain-seed random --episodes 5
python3 scripts/evaluate_locomotion_viewer.py --terrain-seed 20 --episodes 5
```

#### 設定可能なパラメータ

基本設定:

| オプション | デフォルト | 値の例 | 説明 |
|---|---:|---|---|
| `--episodes` | `5` | `10` | 実行するエピソード数。各エピソードで地形を作り直す。 |
| `--duration` | `5.0` | `8` | 1 エピソードの最大秒数。ゴールに届かずこの時間を超えると `timeout`。 |
| `--headless` | なし | `--headless` | GUI を開かずに実行する。スモークテストや大量実行向け。 |
| `--render-fps` | `60.0` | `30` | GUI の描画 FPS。物理ステップや評価周期そのものではない。 |
| `--display` | `:1` | `"$DISPLAY"` | GUI 表示先。環境によって `--display "$DISPLAY"` が必要。 |
| `--policy-dir` | `external/policies/unitree-go2-velocity-flat` | `path/to/policy_dir` | `policy.onnx`、`policy.onnx.data`、`params/deploy.yaml` を含むディレクトリ。 |

速度指令:

| オプション | デフォルト | 値の例 | 説明 |
|---|---:|---|---|
| `--test-command-vx` | `0.3` | `0.4`, `-0.2` | 前後方向の速度指令。正で前進、負で後退。自動ゴール距離にも使う。 |
| `--test-command-vy` | `0.0` | `0.2`, `-0.2` | 横方向の速度指令。横移動を混ぜた評価に使う。地形の横幅もこの値に応じて広がる。 |
| `--test-command-yaw` | `0.0` | `0.3`, `-0.3` | yaw 角速度指令。旋回を混ぜた評価に使う。 |
| `--command-smoothing` | `12.0` | `0`, `8`, `20` | 速度指令を目標値へ近づける一次遅れの強さ。`0` は即時反映、大きいほど素早く反映。 |

ゴールラインと難易度:

| オプション | デフォルト | 値の例 | 説明 |
|---|---:|---|---|
| `--goal-distance` | `auto` | `1.5`, `-1.0`, `auto` | 開始位置から x 方向にどれだけ進めば成功か。数値指定なら固定距離、`auto` なら速度と時間から自動計算。 |
| `--goal-progress-ratio` | `0.75` | `0.6`, `0.9`, `1.0` | `--goal-distance auto` の難易度。`test-command-vx * duration * ratio` がゴール距離になる。小さいほど簡単、大きいほど難しい。 |

自動ゴール距離の例:

```text
--test-command-vx 0.3 --duration 5 --goal-progress-ratio 0.75
=> goal_distance = 0.3 * 5 * 0.75 = 1.125 m
```

地形:

| オプション | デフォルト | 値の例 | 説明 |
|---|---:|---|---|
| `--terrain-seed` | `random` | `random`, `20` | 地形乱数 seed。`random` ならエピソードごとに新しい seed。数値なら再現可能。 |
| `--terrain-seed-stride` | `1` | `10` | 数値 seed のときだけ使う。episode ごとに `seed + stride` ずつ増やす。 |
| `--terrain-amplitude` | `0.065` | `0.04`, `0.08` | 凹凸の最大高さ。単位 m。大きいほど荒い地形。 |
| `--terrain-smoothing-passes` | `2` | `0`, `1`, `4` | 地形高さの平滑化回数。小さいほど角度変化が急、大きいほどなめらか。 |
| `--start-flat-radius` | `0.65` | `0.3`, `1.0` | 開始位置周辺を平坦にする半径。単位 m。 |

地形サイズは実行時に自動調整される。デフォルトの最低サイズは `x=±18m`、`y=±5m` で、ゴールラインのさらに `2m` 先まで x 方向を確保し、横方向は `abs(test-command-vy) * duration + 2m` まで広げる。

転倒判定:

| オプション | デフォルト | 値の例 | 説明 |
|---|---:|---|---|
| `--fall-height` | `0.16` | `0.14`, `0.18` | ベース高さからその地点の地形高さを引いた相対高さが、この値を下回ると転倒。単位 m。 |
| `--fall-uprightness` | `0.55` | `0.5`, `0.7` | 胴体 z 軸の直立度がこの値を下回ると転倒。値域は `-1` から `1`。 |
| `--fall-warmup` | `0.5` | `0.2`, `1.0` | 開始直後の転倒判定を無効にする秒数。初期姿勢の立ち上がり猶予。 |

ログと結果蓄積:

| オプション | デフォルト | 値の例 | 説明 |
|---|---|---|---|
| `--log-dir` | `logs/locomotion_eval` | `logs/tmp_eval` | 実行ごとの JSONL ログ保存先。 |
| `--results-csv` | `logs/locomotion_eval/locomotion_results.csv` | `logs/my_results.csv` | 長期蓄積 CSV。1 エピソード 1 行で詳細値を追記する。 |
| `--results-md` | `logs/locomotion_eval/locomotion_results.md` | `logs/my_results.md` | CSV から再生成する日本語 Markdown サマリ。 |
| `--episode-pause` | `0.8` | `0`, `1.5` | GUI 実行時、エピソード終了後に結果表示を残す秒数。 |
| `--results-reset` | なし | `--results-reset` | CSV/Markdown をリセットしてから、そのまま評価を実行する。 |
| `--results-reset-only` | なし | `--results-reset-only` | CSV/Markdown をリセットして終了する。シミュレーションは実行しない。 |
| `--results-summary-only` | なし | `--results-summary-only` | CSV から Markdown サマリだけ再生成して終了する。 |
| `--results-delete-run` | なし | `--results-delete-run run_...` | 指定した `run_id` の行をまとめて削除する。複数回指定できる。 |
| `--results-delete-row` | なし | `--results-delete-row run_...-ep0001` | 指定した `row_id` の 1 エピソードだけ削除する。複数回指定できる。 |

成功・失敗の意味:

| `termination_reason` | 日本語表示 | 成功扱い | 説明 |
|---|---|---|---|
| `goal_reached` | 成功 | はい | ゴールラインに到達した。 |
| `timeout` | 時間切れ | いいえ | 転ばなかったが、`duration` 内にゴールへ届かなかった。 |
| `fall_height` | 転倒 | いいえ | 地形表面からの相対ベース高さが `--fall-height` 未満になった。 |
| `fall_uprightness` | 転倒 | いいえ | 直立度が `--fall-uprightness` 未満になった。 |
| `viewer_closed` | 中断 | いいえ | GUI ウィンドウを閉じた、または `Esc` で終了した。 |

ヘッドレス確認:

```bash
python3 scripts/evaluate_locomotion_viewer.py \
  --headless \
  --episodes 1 \
  --duration 0.2 \
  --goal-distance 0.01 \
  --test-command-vx 0.3
```

蓄積結果の Markdown サマリだけ再生成する場合:

```bash
python3 scripts/evaluate_locomotion_viewer.py --results-summary-only
```

蓄積結果をすべてリセットする場合:

```bash
python3 scripts/evaluate_locomotion_viewer.py --results-reset-only
```

一部だけ削除する場合:

```bash
python3 scripts/evaluate_locomotion_viewer.py --results-delete-run run_YYYYMMDD_HHMMSS_xxxxxx
python3 scripts/evaluate_locomotion_viewer.py --results-delete-row run_YYYYMMDD_HHMMSS_xxxxxx-ep0001
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
python3 scripts/go2_teleop.py --stance-crouch 0.10
python3 scripts/go2_teleop.py --reset-base-height 0.24
python3 scripts/go2_teleop.py --fall-height 0.14 --fall-uprightness 0.5
python3 scripts/go2_teleop.py --no-auto-reset-on-fall
python3 scripts/go2_teleop.py --idle-damping-scale 2.2
python3 scripts/go2_teleop.py --idle-base-damping 0
```

ヘッドレス確認:

```bash
python3 scripts/go2_teleop.py --headless --duration 3
python3 scripts/go2_teleop.py --headless --duration 3 --test-command-vx 0.5
python3 scripts/go2_teleop.py --headless --duration 3 --test-command-vy 0.3
python3 scripts/go2_teleop.py --headless --duration 3 --test-command-yaw 0.4
python3 scripts/go2_teleop.py --headless --duration 3 --test-jump-time 0.8
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

主な実行スクリプトと、ローカル実行に必要な資産を中心に整理しています。

```text
README.md
requirements.txt
scripts/go2_teleop.py
scripts/go2_obstacle_avoidance_teleop.py
scripts/go2_vision_target_follow.py
scripts/go2_continuous_rough_terrain_teleop.py
scripts/evaluate_locomotion_viewer.py
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
