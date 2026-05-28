# Go2 ガイドロボット研究ロードマップ

## 1. この研究で目指すもの

Go2 を「盲導犬そのもの」として扱うのではなく、視覚障害者の移動支援を研究するための四脚ガイドロボット基盤として発展させる。

最初の到達点は、実機ではなく MuJoCo 上で、屋内の単純な誘導タスクを安全に評価できること。

現時点の主目的は次の 1 つに絞る。

```text
Go2 が低速で 2D ウェイポイントを追従し、転倒・接触・危険姿勢を避けながらゴールできるかを、ヘッドレス評価で繰り返し測る。
```

## 2. 現状

できていること:

- `projects/go2-mujoco/scripts/go2_teleop.py` で Go2 を MuJoCo 上に表示できる。
- 学習済み ONNX ポリシーで歩行できる。
- WASD/QE/Shift/Space で平面移動、旋回、ダッシュ、ジャンプ補助を試せる。
- `--headless --duration` で短時間のスモークテストを実行できる。

まだできていないこと:

- 誘導タスクの自動評価。
- ウェイポイント追従。
- 障害物や廊下を含む評価シナリオ。
- 接触、転倒、ゴール到達、停止理由のログ化。
- ユーザー想定位置、人との距離、安全停止、音声/触覚インターフェース。

## 3. 直近で作るもの

次に作るべきものは、実機連携や高度な AI ではなく、次のコマンドで動く評価基盤。

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/evaluate_guidance_sim.py --scenario straight_corridor --episodes 10
```

このコマンドで、固定シナリオを 10 エピソード実行し、各エピソードの結果を標準出力とログファイルに出す。

最初の成功条件:

```text
10 エピソードすべてでゴール到達
転倒 0 回
障害物接触 0 回
危険姿勢による停止 0 回
平均速度が誘導用途として低速範囲に収まる
```

最初の出力イメージ:

```text
scenario=straight_corridor episodes=10
success=10 fall=0 contact=0 timeout=0 unsafe_stop=0
mean_time=8.4s mean_path_error=0.12m
log=logs/guidance/straight_corridor_YYYYMMDD_HHMMSS.jsonl
```

## 4. 実装順序

### ステップ 0: 現在のスモークテストを維持する

目的: 既存の歩行ポリシーが壊れていないことを確認する。

実行コマンド:

```bash
cd projects/go2-mujoco
.venv/bin/python scripts/go2_teleop.py --headless --duration 0.05
```

このコマンドは、今後も最小の生存確認として使う。

### ステップ 1: 歩行制御ループを評価用に再利用できる形に分ける

目的: `go2_teleop.py` に入っている「MuJoCo 読み込み、ONNX ポリシー、PD 制御、1 ステップ更新」を、キーボード操作なしでも使えるようにする。

実装候補:

```text
projects/go2-mujoco/scripts/go2_sim_runtime.py
```

入れる内容:

- シーン XML とポリシーディレクトリを読み込む関数。
- Go2 の初期姿勢をセットする関数。
- `command = [vx, vy, yaw]` を受け取り、一定時間 MuJoCo を進める関数。
- 現在のベース位置、姿勢、速度、直立度を返す関数。
- GUI、キーボード、ダッシュ、ジャンプ補助に依存しない最小 API。

完了条件:

- `go2_teleop.py` の既存ヘッドレステストが通る。
- 新しい評価コードから同じ歩行ポリシーを呼べる。
- `external/` 配下は編集しない。

### ステップ 2: 最小のウェイポイント追従コントローラを作る

目的: ロボットの現在位置と次のウェイポイントから、低速の速度指令を出す。

実装候補:

```text
projects/go2-mujoco/scripts/guidance_controller.py
```

最初の仕様:

- 入力: 現在の `x, y, yaw` とウェイポイント列。
- 出力: `vx, vy, yaw_rate`。
- 速度上限は誘導用途として低くする。
- ゴール付近では減速する。
- ダッシュ、ジャンプ、急旋回は使わない。

初期パラメータ:

```text
通常前進速度: 0.3 m/s
最大前進速度: 0.5 m/s
最大横速度: 0.2 m/s
最大旋回速度: 0.5 rad/s
ゴール半径: 0.3 m
ウェイポイント到達半径: 0.25 m
```

完了条件:

- 平面上の `[(0, 0), (2, 0)]` のような直線ウェイポイントに追従できる。
- 速度指令が常に上限内に収まる。
- ゴール到達時に `vx=0, vy=0, yaw_rate=0` を返す。

### ステップ 3: 評価シナリオを定義する

目的: 手動操作ではなく、同じ条件を何度も実行できるようにする。

実装候補:

```text
projects/go2-mujoco/scripts/guidance_scenarios.py
```

最初に作るシナリオ:

```text
straight_corridor
turning_corridor
static_obstacle_avoidance
```

最初は複雑な 3D 環境を作り込みすぎない。まずは平面シーン上で、ウェイポイント、仮想壁、仮想障害物を Python 側の 2D 幾何として持てばよい。

各シナリオに含める情報:

- 初期位置。
- ゴール位置。
- ウェイポイント列。
- 障害物の位置と半径、または壁の線分。
- 制限時間。
- 接触判定距離。
- 成功条件と失敗条件。

完了条件:

- シナリオ名を指定して読み込める。
- シナリオごとにウェイポイントと安全判定条件が取得できる。
- MuJoCo シーン編集なしでも最初の評価が動く。

### ステップ 4: 安全シールドを作る

目的: コントローラが危険な指令を出しても、最後に速度を制限または停止する。

実装候補:

```text
projects/go2-mujoco/scripts/guidance_safety.py
```

最初の安全条件:

- 速度指令の上限を超えたらクリップする。
- 直立度が低い場合は停止する。
- ベース高さが低すぎる場合は転倒扱いにする。
- 障害物や壁に近づきすぎたら停止する。
- 制限時間を超えたらタイムアウトにする。

初期しきい値:

```text
最小ベース高さ: 0.18 m
最小直立度: 0.75
障害物停止距離: 0.35 m
壁停止距離: 0.25 m
制限時間: シナリオごとに 20-60 秒
```

完了条件:

- 危険時に停止理由を返せる。
- 速度指令を安全範囲へ制限できる。
- 評価ログに停止理由が残る。

### ステップ 5: ヘッドレス評価スクリプトを作る

目的: 複数エピソードを自動実行し、結果を比較できるようにする。

実装候補:

```text
projects/go2-mujoco/scripts/evaluate_guidance_sim.py
```

CLI 仕様:

```bash
.venv/bin/python scripts/evaluate_guidance_sim.py --scenario straight_corridor --episodes 10
.venv/bin/python scripts/evaluate_guidance_sim.py --scenario turning_corridor --episodes 10
.venv/bin/python scripts/evaluate_guidance_sim.py --scenario static_obstacle_avoidance --episodes 10
```

記録する指標:

- 成功/失敗。
- 停止理由。
- ゴール到達時間。
- 走行距離。
- ウェイポイントからの平均誤差。
- 最大速度。
- 最小ベース高さ。
- 最小直立度。
- 障害物や壁への最小距離。

ログ形式:

```text
projects/go2-mujoco/logs/guidance/*.jsonl
```

`logs/` は実験出力なので Git に入れない。

完了条件:

- `--scenario` と `--episodes` を指定して実行できる。
- 1 エピソードごとの JSONL と、最後の集計サマリを出せる。
- `straight_corridor` で 10/10 成功を確認できる。

## 5. 最初の実装タスク一覧

優先順位順に進める。

1. `go2_teleop.py` の歩行制御部分を `go2_sim_runtime.py` へ分離する。
2. `go2_teleop.py` が新しい runtime を使って今まで通り動くようにする。
3. `guidance_controller.py` に直線ウェイポイント追従を実装する。
4. `guidance_scenarios.py` に `straight_corridor` を定義する。
5. `guidance_safety.py` に速度制限、転倒判定、タイムアウト判定を入れる。
6. `evaluate_guidance_sim.py` で 1 エピソードだけ実行できるようにする。
7. `evaluate_guidance_sim.py --episodes 10` で集計を出す。
8. `turning_corridor` を追加する。
9. `static_obstacle_avoidance` を追加する。
10. `projects/go2-mujoco/README.md` に評価コマンドを追記する。

## 6. フェーズ別の研究計画

### フェーズ 1: 誘導タスクの定義

「何をできればガイドと言えるか」を、まず屋内限定で定義する。

初期対象:

- 目的地までの低速誘導。
- 静的障害物の回避または安全停止。
- 曲がり角。
- 狭い通路。
- ユーザー想定位置との距離維持。

初期対象外:

- 公道。
- 横断歩道。
- 駅。
- 階段。
- 群衆。
- 自転車や車などの動的交通環境。

### フェーズ 2: シミュレーション評価基盤

`evaluate_guidance_sim.py` を中心に、誘導タスクを自動評価する。

このフェーズの成果物:

- ヘッドレス評価スクリプト。
- 3 つ以上の固定シナリオ。
- JSONL ログ。
- 成功率、転倒率、接触率、停止理由の集計。

### フェーズ 3: ナビゲーション層

歩行ポリシーの上に、誘導用途の低速な速度指令を出す層を作る。

このフェーズの成果物:

- ウェイポイント追従。
- 減速制御。
- 障害物や狭路での停止。
- 急加速、急旋回、ダッシュ、ジャンプの無効化。

### フェーズ 4: 人とロボットのチームモデル

ロボット単体ではなく、人とロボットを 1 つの移動チームとして評価する。

最初は物理的な人間モデルを作り込みすぎず、2D 点または簡易剛体でよい。

見る指標:

- ロボットとユーザー想定位置の距離。
- 相対角度。
- 相対速度。
- ユーザーが停止した場合のロボット停止。
- ユーザーが遅れた場合の減速。

### フェーズ 5: 安全シールド

将来センサや実機に移る前の必須条件として、安全層を独立させる。

止める条件:

- 速度、旋回速度、加速度が上限を超える。
- 障害物や壁に近すぎる。
- 接触した。
- 転倒した、または転倒寸前。
- ユーザー想定位置との距離が離れすぎた。

### フェーズ 6: インタラクション

ロボットが人へ意図を伝える方法を決める。

候補:

- 触覚: ハーネス、リード、グリップへの方向提示。
- 音声: 停止理由、目的地、次の動作。
- 操作: 停止、再開、目的地変更、緊急停止。

音声だけに依存せず、触覚、停止挙動、速度変化で伝わる設計を優先する。

### フェーズ 7: 実世界移行

シミュレーションで安全性が十分に見えてから、実機に進む。

段階:

```text
1. 実機 Go2 を人なしで低速歩行
2. 障害物ありで無人テスト
3. 安全担当者が近くにいる状態で晴眼者がハンドルを持つ
4. 晴眼者の目隠しテスト
5. 倫理・安全レビュー後に視覚障害者協力者との限定評価
```

視覚障害者ユーザーを巻き込む段階では、研究倫理、安全管理、同意、保険、法規、緊急停止手順を先に整える。

## 7. 安全原則

- 安全を最優先する。実機・人間・視覚障害者ユーザーを巻き込む前に、シミュレーションで危険行動を潰す。
- ユーザー中心で設計する。盲導犬ユーザー、白杖ユーザー、歩行訓練士、アクセシビリティ専門家の知見を前提にする。
- 過大な主張をしない。現段階ではサービスアニマルや医療機器ではなく、研究用の移動支援ロボットとして扱う。
- 屋内から始める。初期対象は屋内の廊下、曲がり角、ドア、静的障害物に限定する。
- ロボットはユーザーに従属させる。ユーザーの停止、拒否、緊急停止を常に優先する。
- `projects/go2-mujoco/external/` は読み取り専用として扱う。
- `projects/go2-mujoco/.venv/`、`projects/go2-mujoco/external/`、`papers/pdfs/`、`__pycache__/`、`.DS_Store`、`._*`、実験ログは Git に入れない。

## 8. 現時点で扱わないこと

- いきなり実機で視覚障害者を誘導しない。
- 公道、横断歩道、駅、階段、混雑環境は扱わない。
- 「盲導犬の代替」として宣伝しない。
- `projects/go2-mujoco/external/` のモデルやポリシーを研究コードとして編集しない。
- 大規模な LLM 会話機能や視覚言語モデルを最初から入れない。
- ロボットの自律判断を人の停止・拒否・緊急停止より優先しない。

## 9. 参考資料の置き場所

論文 PDF とメモは `papers/` 配下で管理する。

```text
papers/references.bib
papers/pdfs/
papers/notes/
```

引用メタデータは `references.bib`、ローカル PDF コピーは `pdfs/`、論文ごとの Markdown 要約は `notes/` に置く。

参考リンク:

- ADA.gov Service Animals: https://www.ada.gov/topics/service-animals/
- Transforming a Quadruped into a Guide Robot for the Visually Impaired: https://arxiv.org/abs/2306.14055
- System Configuration and Navigation of a Guide Dog Robot: https://arxiv.org/abs/2210.13368
- Robotic Guide Dog with Leash-Guided Hybrid Physical Interaction: https://hybrid-robotics.berkeley.edu/publications/ICRA2021_GuideDog.pdf
