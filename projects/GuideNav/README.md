# GuideNav + MuJoCo 実行環境

このディレクトリは、公式 `guidedogrobot-navigation/GuideNav` を clone したものです。公式の `guidenav/`, `sensor/`, `topogen/` はそのまま使い、MuJoCo との接続用に `mujoco_sim/` だけを追加しています。

目的は次の流れを MuJoCo で試すことです。

```text
1. MuJoCo 上の Go2 をテレオペで一度走らせる
   -> RGB画像、depth画像、odom.csv を保存

2. 公式 sensor/build_topomap.py で topomap を作る
   -> 公式 GuideNav が参照する keyframe 画像列を作る

3. 公式 GuideNav の place recognition 形式で CosPlace DB を作る
   -> topomap 画像を global-feats-cosplace_hub.h5 に変換

4. MuJoCo GUI で CosPlace 照合を使って自律 repeat する
   -> 現在画像を topomap と照合し、2回目以降を自律歩行
```

`2` と `3` は変換処理なので GUI は出ません。GUI が出るのは `1` と `5` の MuJoCo 側です。

## いつもの使い方

普段はこの3つだけ使えば大丈夫です。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate
```

1回目のテレオペ録画:

```bash
python3 mujoco_sim/run.py teach
```

GUI が出ます。`W/S` で前後、`A/D` で左右、`Q/E` で旋回、`Esc` で録画終了です。

録画から topomap と VPR DB を作る:

```bash
python3 mujoco_sim/run.py build
```

`build` のあと、撮影画像と選ばれたキーフレームを確認するためのフォルダが自動で作られます。

```text
/home/daisuke/research/local/guidenav_mujoco/latest/
  index.html                         ブラウザで見る確認ページ
  recorded_images/                   テレオペ中に撮影した画像
  keyframes/                         topomap に選ばれたキーフレーム画像
  recorded_images_contact_sheet.jpg  撮影画像の一覧
  keyframes_contact_sheet.jpg        キーフレームの一覧
```

ブラウザで開くなら:

```bash
xdg-open /home/daisuke/research/local/guidenav_mujoco/latest/index.html
```

2回目以降の自律走行:

```bash
python3 mujoco_sim/run.py repeat
```

自律走行のGUI右側には、次の画像パネルが出ます。

```text
current camera      現在のMuJoCoカメラ画像
VPR match #N        CosPlace が現在地として選んだキーフレーム
subgoal #M          ReLoc3R が相対姿勢を推定する目標キーフレーム
```

`repeat` は CUDA が使えるなら自動で `cuda` を選びます。明示したい場合は次です。

```bash
python3 mujoco_sim/run.py repeat --device cuda
```

軽めにしたい場合:

```bash
python3 mujoco_sim/run.py repeat --device cuda --image-width 160 --image-height 120 --vpr-fps 1
```

環境確認:

```bash
python3 mujoco_sim/run.py check
```

## まずコピペするコマンド

Ubuntu のターミナルを開いて、まずこの2行を実行します。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate
```

仮想環境が有効になると、プロンプトの先頭に `(.venv)` が出ます。確認するなら次を実行します。

```bash
which python
python --version
```

`which python` が次を指していれば OK です。

```text
/home/daisuke/research/projects/go2-mujoco/.venv/bin/python
```

## 0. 環境確認

まず不足しているものを確認します。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate
python mujoco_sim/check_env.py
```

教示走行の録画だけなら、MuJoCo、NumPy、ONNX Runtime、GLFW、PyYAML、Go2 model、Go2 policy があれば動きます。

公式 `guidenav/navigate.py` まで動かすには、追加で OpenCV、pandas、PyTorch、ROS2 Humble、`cv_bridge`、GuideNav の model weights が必要です。

Python 側の依存関係を入れ直す場合は、次を実行します。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

python -m pip install --upgrade opencv-python-headless pandas h5py matplotlib tqdm scipy pillow gdown
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

ROS2 Humble と `cv_bridge` は pip だけでは入りません。`/opt/ros/humble/setup.bash` が存在しない環境では、公式 `guidenav/navigate.py` の ROS2 実行はまだできません。

## 1. MuJoCo GUI で教示走行を録画

これが最初に GUI が出るコマンドです。屋外風の道路、歩道、車、歩行者、工事障害物がある scene で起動します。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

RUN="$PWD/data/mujoco_teach/raw/run_$(date +%Y%m%d_%H%M%S)"

python mujoco_sim/record_teach_run.py \
  --real-time \
  --scene-config mujoco_sim/scenes/outdoor_city_route.json \
  --scene-preset sunny_morning \
  --image-width 320 \
  --image-height 180 \
  --camera-fps 2 \
  --output-dir "$RUN"
```

複数行の貼り付けが崩れる場合は、次の一行版を使ってください。これだけで新しい保存先 `RUN` を作ってから起動します。

```bash
RUN="$PWD/data/mujoco_teach/raw/run_$(date +%Y%m%d_%H%M%S)" && python mujoco_sim/record_teach_run.py --real-time --scene-config mujoco_sim/scenes/outdoor_city_route.json --scene-preset sunny_morning --image-width 320 --image-height 180 --camera-fps 2 --output-dir "$RUN"
```

操作キー:

```text
W/S       前進/後退
A/D       左右移動
Q/E       左右旋回
Shift     少し速く移動
R         初期姿勢に戻す
Esc       録画終了
```

録画が終わったら、同じターミナルで保存先を確認できます。

```bash
echo "$RUN"
```

保存されるもの:

```text
$RUN/
  color/      RGB画像
  depth/      depth画像
  odom.csv    位置・姿勢ログ
```

もし GUI が一瞬で閉じた場合は、起動に失敗しています。次で直近のログを確認してください。

```bash
cd /home/daisuke/research/projects/GuideNav

LATEST_RUN=$(find data/mujoco_teach/raw -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
echo "$LATEST_RUN"
cat "$LATEST_RUN/run.log"
```

`color=0` のように画像が保存されていない場合も、その run は教示データとして使わず、ログを確認してから新しい `RUN` で録画し直してください。

## 2. 公式 GuideNav 用 topomap を作る

ここは GUI は出ません。`1` で保存した画像と odometry から、公式 GuideNav が読む keyframe 画像列を作ります。

`1` を実行した同じターミナルで、そのまま次を実行してください。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

python sensor/build_topomap.py \
  "$RUN" \
  data/mujoco_teach/topomap \
  --distance 0.35 \
  --yaw 14
```

成功すると、だいたい次のようなログが出ます。

```text
Loading odometry data...
Getting image timestamps...
Found ... RGB images and ... depth images
Found ... aligned data points
Keyframe 0: ...
Selected ... keyframes
Keyframe extraction completed!
```

出力確認:

```bash
find data/mujoco_teach/topomap/topo -maxdepth 1 -name '*.png' | sort | head
```

出力されるもの:

```text
data/mujoco_teach/topomap/
  color/
  depth/
  topo/
    0.png
    1.png
    2.png
    ...
  odom.csv
```

## 3. GUI が出ない環境で短く確認（任意）

GUI やキーボード操作なしで、scripted teacher に自動走行させる確認コマンドです。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

python mujoco_sim/record_teach_run.py \
  --headless \
  --scripted-teacher \
  --duration 10 \
  --scene-config mujoco_sim/scenes/outdoor_city_route.json \
  --scene-preset rainy_evening \
  --image-width 160 \
  --image-height 120 \
  --camera-fps 2 \
  --output-dir /tmp/guidenav_mujoco_raw_test

python sensor/build_topomap.py \
  /tmp/guidenav_mujoco_raw_test \
  /tmp/guidenav_mujoco_topomap_test \
  --distance 0.35 \
  --yaw 14
```

## 4. CosPlace の VPR DB を作る

`2` で作った topomap 画像を、公式 GuideNav の place recognition が読む HDF5 DB に変換します。初回だけ `gmberton/CosPlace` から CosPlace 重みをダウンロードします。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

python mujoco_sim/build_vpr_db.py \
  --topomap-dir data/mujoco_teach/topomap \
  --pr-model cosplace_hub \
  --overwrite
```

成功すると、次が作られます。

```text
data/mujoco_teach/topomap/topo/global-feats-cosplace_hub.h5
```

さらに `run.py build` を使った場合は、撮影画像とキーフレームを確認しやすいように次も作ります。

```text
/home/daisuke/research/local/guidenav_mujoco/latest/index.html
```

一行版:

```bash
python mujoco_sim/build_vpr_db.py --topomap-dir data/mujoco_teach/topomap --pr-model cosplace_hub --overwrite
```

## 5. GUI で CosPlace 自律走行を確認する

`4` で作った DB を使い、MuJoCo カメラ画像を CosPlace で topomap に照合しながら自律走行します。GUI が出ます。
GUI 右側には、現在カメラ画像、CosPlace の VPR match、ReLoc3R の subgoal keyframe が表示されます。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

python mujoco_sim/replay_cosplace.py \
  --real-time \
  --topomap-dir data/mujoco_teach/topomap \
  --pr-model cosplace_hub \
  --feature-matching reloc3r \
  --scene-config mujoco_sim/scenes/outdoor_city_route.json \
  --scene-preset sunny_morning \
  --image-width 320 \
  --image-height 180 \
  --vpr-fps 2
```

一行版:

```bash
python mujoco_sim/replay_cosplace.py --real-time --topomap-dir data/mujoco_teach/topomap --pr-model cosplace_hub --feature-matching reloc3r --scene-config mujoco_sim/scenes/outdoor_city_route.json --scene-preset sunny_morning --image-width 320 --image-height 180 --vpr-fps 2
```

この実装で使っている深層学習部分は、CosPlace の VPR と ReLoc3R の相対姿勢推定です。移動先は topomap の odom 座標ではなく、次の流れで決めます。

```text
現在のMuJoCoカメラ画像
  -> CosPlace で近い topomap keyframe を探す
  -> lookahead 先の subgoal keyframe を選ぶ
  -> ReLoc3R で 現在画像 + subgoal画像 の相対姿勢 x,y,yaw を推定
  -> 公式 control.vtr_controller で v,w に変換
  -> Go2 velocity policy に [v, 0, w] を渡す
```

CPU でも動きますが、ReLoc3R は重いのでリアルタイムより遅くなることがあります。GPU が使える環境なら `--device cuda` と CUDA 版 PyTorch の利用を検討してください。

## 6. scene の見た目を変える

`--scene-preset` で時間帯・天候の見た目を変えられます。

```text
sunny_morning
cloudy_noon
rainy_evening
night
```

例:

```bash
python mujoco_sim/record_teach_run.py \
  --real-time \
  --scene-preset night \
  --cycle-appearance \
  --cycle-period 15 \
  --image-width 320 \
  --image-height 180 \
  --camera-fps 2 \
  --output-dir "$RUN"
```

`--cycle-appearance` を付けると、実行中に morning/noon/evening/night の光源と haze を周期的に切り替えます。

注意: これは MuJoCo の幾何形状と簡易マテリアルによる屋外風 scene です。写真のような都市環境、信号、人混み、車の複雑な挙動まで評価したい場合は、CARLA、Isaac Sim、Unreal 系のフォトリアル環境に接続する方が適切です。この MuJoCo scene は、まず GuideNav のデータ経路と Go2 歩行制御を確認するための軽量環境です。

## 7. GUI で odom だけの自律走行を確認する

`2` で作った `data/mujoco_teach/topomap/odom.csv` を読み、MuJoCo 上の Go2 が keyframe 列を自律追従します。GUI が出ます。

```bash
cd /home/daisuke/research/projects/GuideNav
source ../go2-mujoco/.venv/bin/activate

python mujoco_sim/replay_topomap.py \
  --real-time \
  --topomap-dir data/mujoco_teach/topomap \
  --scene-config mujoco_sim/scenes/outdoor_city_route.json \
  --scene-preset sunny_morning \
  --image-width 320 \
  --image-height 180
```

複数行の貼り付けが崩れる場合は、次の一行版を使ってください。

```bash
python mujoco_sim/replay_topomap.py --real-time --topomap-dir data/mujoco_teach/topomap --scene-config mujoco_sim/scenes/outdoor_city_route.json --scene-preset sunny_morning --image-width 320 --image-height 180
```

これは ROS2/model weights なしで動く MuJoCo 側の最小プレビューです。CosPlace を使う場合は `5` の `replay_cosplace.py` を使ってください。

## 8. 公式 GuideNav で自律 repeat する

ここからは ROS2 Humble と GuideNav の model weights が必要です。現環境で `mujoco_sim/check_env.py` が `rclpy`、`cv_bridge`、`torch`、model weights を `MISSING` と出す場合、この段階はまだ動きません。

ターミナル 1: MuJoCo bridge を起動します。

```bash
cd /home/daisuke/research/projects/GuideNav
source /opt/ros/humble/setup.bash
source ../go2-mujoco/.venv/bin/activate

python mujoco_sim/ros_bridge.py replay \
  --real-time \
  --scene-config mujoco_sim/scenes/outdoor_city_route.json \
  --scene-preset sunny_morning \
  --image-width 320 \
  --image-height 180 \
  --camera-fps 4
```

ターミナル 2: 公式 GuideNav を起動します。

```bash
cd /home/daisuke/research/projects/GuideNav
source /opt/ros/humble/setup.bash
source ../go2-mujoco/.venv/bin/activate

python guidenav/navigate.py \
  --robot go2 \
  --robot-config-path config/robots.yaml \
  --topomap-base-dir data/mujoco_teach/topomap \
  --topomap-dir topo \
  --model-weight-dir model_weights \
  --model-config-path config/models.yaml \
  --pr-model cosplace_hub \
  --feature-matching reloc3r \
  --device cpu
```

このときの接続は次です。

```text
MuJoCo camera/depth/odom
  -> /d435i/color/image_raw
  -> /d435i/aligned_depth_to_color/image_raw
  -> /visual_slam/tracking/odometry
  -> 公式 guidenav/navigate.py
  -> /cmd_vel
  -> MuJoCo Go2 velocity policy
```

## 追加した MuJoCo ファイル

```text
mujoco_sim/
  check_env.py                         環境確認
  common.py                            MuJoCo scene/runtime 共通処理
  build_vpr_db.py                      topomap画像から公式VPR DBを作成
  record_teach_run.py                  教示走行を公式 topomap 入力形式で保存
  replay_cosplace.py                   CosPlace照合 + ReLoc3R相対姿勢でGUI自律repeat
  replay_topomap.py                    topomap/odom.csv をGUIで自律追従
  run.py                               teach/build/repeat を簡単に実行する入口
  ros_bridge.py                        公式 navigate.py と MuJoCo を ROS2 topic で接続
  scenes/outdoor_city_route.json       屋外風2回曲がりルートと動的 actor 設定
```

## 公式 GuideNav の構成

```text
guidenav/       公式 navigation 本体
sensor/         公式データ抽出・topomap 作成スクリプト
topogen/        公式 topomap 生成補助
config/         公式 robot/model 設定
model_weights/  学習済み重み配置先
```

## Citation

```bibtex
@inproceedings{hwang2026guidenav,
  title={Guidenav: User-informed development of a vision-only robotic navigation assistant for blind travelers},
  author={Hwang, Hochul and Yang, Soowan and Monon, Jahir Sadik and Giudice, Nicholas A and Lee, Sunghoon Ivan and Biswas, Joydeep and Kim, Donghyun},
  booktitle={Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction},
  pages={1129--1139},
  year={2026}
}
```

## License

GuideNav is released under the MIT License. See `LICENSE` for details.
