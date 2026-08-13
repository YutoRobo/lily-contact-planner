# 最新探索結果の再生・GIF表示

`results/latest_search_report.json` に保存された探索結果から、接触系列を再探索せずに再生用軌道を復元し、GIFを生成できます。

## 1. 可視化依存を入れる

```bash
python3 -m pip install -r requirements-visualization.txt
```

## 2. 保存済み探索結果をリプレイする

```bash
python3 scripts/replay_latest.py
```

生成物:

```text
results/latest_trajectory.npz
results/latest_switch_states.npz
```

`latest_trajectory.npz` には 1°刻みの

- `angles_deg`
- `body_t`
- `body_R`
- `joint_q`
- `support_mask`

が入ります。

`latest_switch_states.npz` には接触切替直前・直後の状態を保存します。

この処理は新しい接触系列を探索しません。`latest_search_report.json` に記録された `events` を順に適用し、連続関節状態だけを同じ運動学 solver で再構成します。

## 3. GIFを生成する

```bash
python3 scripts/render_latest_gif.py
```

生成物:

```text
results/latest.gif
```

接触切替は必ず

```text
旧支持維持
-> 新脚 touchdown approach
-> 新脚 touchdown
-> 新旧両支持
-> 支持移行
-> 旧脚 liftoff
```

の順で表示します。

## 表示速度・フレーム数を変える

デフォルトでは通常状態を2°おきに表示し、接触切替部分は5段階で必ず挿入します。

全1°状態を表示:

```bash
python3 scripts/render_latest_gif.py --stride 1
```

軽量化:

```bash
python3 scripts/render_latest_gif.py --stride 4
```

FPS変更:

```bash
python3 scripts/render_latest_gif.py --fps 20
```

出力ファイル名変更:

```bash
python3 scripts/render_latest_gif.py --out results/my_motion.gif
```

## 注意

接触切替時に挿入する touchdown / dual-support / transfer / liftoff の中間フレームは表示用補間です。planner が有限時間の接触遷移として最適化した軌道ではありません。

また、現在の `replay_latest.py` は `scripts/run_rollwalk_720.py` で生成した `ForwardRollTask` の `latest_search_report.json` を対象にしています。任意タスクについては、今後 report 内に task metadata を保存して同じ replay entry point から再構成できるよう一般化する予定です。
