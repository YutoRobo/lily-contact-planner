# Lily Contact Planner

8脚ロボット Lily を対象とした、**ハイブリッド幾何・運動学ベースの接触計画**の研究用プロトタイプです。

現在の基準マイルストーンは、前進しながら 0°→720° 回転する課題に対して、**単一の接触探索アルゴリズム**を適用した結果です。planner には接触切替角度や歩容・接触系列を事前に与えません。初期関節状態と初期支持脚集合だけから開始し、現在の支持状態で進めるところまで進み、実行不能になったら touchdown 候補を生成し、局所的な接触 add/remove を探索し、先で行き止まりになれば backtracking します。

## 現在の基準タスク

胴体幾何中心の高さを 0.35 m に固定し、基準タスクでは

- roll: 0° → 720°
- 前進量: `x = roll_deg / 300`（720°で 2.4 m）
- `y = 0`
- pitch = yaw = 0

としています。

この胴体軌道は**タスク定義**であり、接触歩容をハードコードしているわけではありません。接触切替は planner が自動で決定します。

## 任意の並進・回転の設定方法

contact planner が必要とするのは、経路パラメータ `s` に対して目標胴体姿勢

```text
(t(s), R(s))
```

を返すタスク定義だけです。

ここで

- `t(s) = [x, y, z]`：**ワールド座標系**での胴体幾何中心位置
- `R(s)`：胴体姿勢を表す回転行列 `SO(3)`

です。

### 任意方向への並進

並進はワールド座標系で直接指定します。

初期位置 `p0` から、ワールド座標系の任意方向 `d` へ直線移動させる場合は、例えば

```python
p = p0 + distance(s) * d / np.linalg.norm(d)
```

とします。

`d` は3次元の任意方向で構いません。曲線軌道にしたい場合は、`x(s)`, `y(s)`, `z(s)` を個別に定義します。

例:

```python
# +x 方向へ並進
p = np.array([distance, 0.0, 0.35])

# +y 方向へ並進
p = np.array([0.0, distance, 0.35])

# xy 平面45°方向へ並進
p = np.array([
    distance / np.sqrt(2.0),
    distance / np.sqrt(2.0),
    0.35,
])
```

### ワールド座標系の任意軸まわりの回転

回転は、ワールド座標系で定義した単位回転軸

```text
nW = [nx, ny, nz],  ||nW|| = 1
```

と回転量 `theta` で指定します。

姿勢更新は

```text
R(theta) = Exp([nW]x theta) R0
```

です。

`[nW]x` は `nW` の歪対称行列です。

重要なのは、**増分回転を左から掛けること**です。これにより、回転軸は機体姿勢に追従せず、常にワールド座標系に固定されます。

SciPy では次のように書けます。

```python
from scipy.spatial.transform import Rotation

nW = np.asarray(nW, dtype=float)
nW = nW / np.linalg.norm(nW)

R_inc = Rotation.from_rotvec(theta_rad * nW).as_matrix()
R = R_inc @ R0
```

yaw / pitch / roll は、この任意軸回転の特殊例です。

```text
+yaw   : nW = [ 0,  0,  1]
+pitch : nW = [ 0,  1,  0]
-roll  : nW = [-1,  0,  0]
```

例えば、ワールド `xy` 平面内の45°方向を回転軸にしたい場合は

```python
nW = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
```

とします。

### 任意の並進と任意軸回転を同時に行う

並進と回転は、目標胴体姿勢の独立した要素として設定できます。

そのため、任意の並進方向と、任意の回転軸を同時に指定できます。

```python
def pose(s):
    # ワールド座標系での任意並進
    p = p0 + distance(s) * direction_world

    # ワールド座標系での任意軸回転
    theta = angle(s)
    R_inc = Rotation.from_rotvec(theta * axis_world).as_matrix()
    R = R_inc @ R0

    return p, R
```

`direction_world` と `axis_world` は平行である必要はありません。

例えば、斜め方向へ並進しながら、それとは無関係な3次元方向を軸として回転させることも可能です。

### 複数区間の動作を連結する

長い動作は、複数の pose 区間をつなげて定義できます。

区間 `j` の開始姿勢を `R_start` とすると、ワールド座標系の回転軸 `nW_j` に対して

```text
R_j(theta) = Exp([nW_j]x theta) R_start
```

とし、その区間の終端姿勢を次区間の初期姿勢として使用します。

現在の実験タスクでは、例えば

```text
その場 +yaw 45°
-> +x 並進しながら +pitch 480°
-> +y 並進しながら -roll 480°
```

を連結しており、各回転軸はすべてワールド座標系で定義しています。

### 将来の joystick 指令

将来的には、上位層から

```text
v_des^W, omega_des^W
```

を与える構成を想定できます。

ここで

- `v_des^W`：ワールド座標系での目標並進速度
- `omega_des^W`：ワールド座標系での目標角速度

です。

微小ステップでは

```text
R_{k+1} = Exp([omega_des^W]x Delta s) R_k
```

と更新できます。

なお、**現在の planner は胴体の並進・姿勢軌道そのものを最適化しているわけではありません**。上位から与えられた胴体目標軌道に対して、それを実現するための接触系列と脚関節状態を自動探索します。

## 統一接触探索アルゴリズム

各経路サンプルで同じルールを適用します。

1. 現在の支持脚足先をワールド座標系で固定し、次のタスク姿勢に対して支持脚 IK を解く。
2. 非支持脚について、現在姿勢から連続的に到達可能な範囲で床上クリアランスを確保する。
3. 現在の支持集合でこれ以上進めなくなったら、遊脚の touchdown 候補を生成する。
4. 現在の遊脚姿勢から ground-safe な連続関節軌道で到達できない touchdown 候補を除外する。
5. 少数脚の add/remove を組み合わせて局所的な接触変更候補を生成する。毎回 `2^8` 個の全支持状態を総当たりする方式ではない。
6. 各候補について支持脚 IK と有限 look-ahead を用いて、どれだけ先へ進めるかを評価する。
7. 接触イベント木を DFS/backtracking で探索する。
8. 目標終端まで到達するか、探索上限に達するまで繰り返す。

角度ごとの専用分岐、保存済み未来状態、事前指定した接触切替角度は使用しません。

## 現在使っている solver

チェックイン済みの 0°→720° 基準版は、全区間を一括で解く巨大な NLP でも QP でもありません。

2層構成です。

- **離散接触決定**：DFS + backtracking
- **連続関節状態**：`scipy.optimize.least_squares` を使った bounded nonlinear least-squares IK

要約すると

```text
接触系列: DFS / backtracking
+
連続関節状態: nonlinear least-squares IK
```

です。

主なコード位置は次の通りです。

- `src/lily_contact_planner/planner_search.py`
  - `plan()` — planner の入口
  - `_dfs()` — 接触イベントの DFS/backtracking
  - `_advance_to_stall()` — 現在の支持状態で進めるところまで前進
- `src/lily_contact_planner/planner_touchdown.py`
  - `_reachable_touchdowns()` — touchdown 候補の生成・絞り込み
  - `_rank_plans()` — 局所 add/remove 候補の順位付け
- `src/lily_contact_planner/planner_base.py`
  - `_solve_leg_to_anchor()` — 支持脚足先固定 IK
  - `_support_only()` — 支持状態の運動学的成立性確認
  - `_actual()` — 各数値状態の Level-1 成立性確認
  - `_predict_gain()` — look-ahead 進捗量の推定

詳細は [`docs/algorithm_solver.md`](docs/algorithm_solver.md) にまとめています。

## 基準結果

チェックイン済みの基準版では 720° に到達しています。

- DFS nodes: 29
- contact events: 28
- 最終支持脚集合: `[0, 4, 7]`

参照ファイル:

- `results/unified_rollwalk_720_search_summary.json` — 基準結果の要約
- `results/unified_rollwalk_720_contact_events.json` — 探索で得られた接触イベント系列
- `results/unified_rollwalk_720_terminal.npz` — 終端 joint/support/anchor 状態

接触イベント系列は**入力ではなく探索結果**です。

重要な制限として、この基準マイルストーンが確認しているのは、**接触探索系列と数値サンプル点での Level-1 成立性**です。全サンプル間の連続時間軌道を厳密に保証する dense continuous certification は別途必要です。また、self-collision は Level-1 checker で計測していますが、現在の stepping logic 検討段階では接触探索候補の reject 条件には使用していません。

## 現在の実験進捗 — 2026-08-13

0°→720° の結果は、歴史的な baseline として保持しています。

その後の実験では、DFS/backtracking を中心とした接触探索構造は維持しつつ、以下の変更を加えています。

- 高速実験用の3DOF脚解析 IK
- 接触切替後に、少なくとも次の1 task step へ実際に進めることを要求
- normal branch が失敗した場合の expanded touchdown fallback
- 実験用の軽量 search parameter

現在の多軸タスクでは、回転軸をすべて**ワールド座標系**で定義しています。

- 0°–45°: world `+z` 軸まわりに、その場 `+yaw`
- 45°–525°: `+x` 並進しながら world `+y` 軸まわりに `+pitch`
- 525°–1005°: `+y` 並進しながら world `+x` 軸まわりに `-roll`

再利用可能な `YawPitchRollWorldTask` を `src/lily_contact_planner/tasks.py` に追加しています。

fresh search の部分解として、total progress 255° = yaw 45° + pitch 210° まで到達しています。

- DFS nodes: 7
- contact events: 6
- 保存された 1° 刻み 256 状態で joint-limit violation: 0
- 保存状態すべてで support-region condition: OK

この world-frame 実験では、まだ roll 区間には到達していません。

可視化では、接触切替を

```text
旧支持維持
-> 新脚 touchdown
-> 新旧両支持
-> support transfer
-> 旧脚 liftoff
```

の順で表示します。

再利用可能な実装は `src/lily_contact_planner/visualization.py` にあり、仕様は [`docs/visualization.md`](docs/visualization.md)、順序のテストは `tests/test_visualization.py` にあります。

ただし、この中間 transition frame は現時点では**表示用再構成**です。有限時間の接触遷移軌道として planner が明示的に最適化した結果ではありません。

詳細は [`docs/progress_20260813.md`](docs/progress_20260813.md) と `results/yaw45_pitch480_world_partial_255_summary.json` を参照してください。

## ディレクトリ構成

- `src/lily_contact_planner/kinematics.py` — Lily の順運動学・Jacobian
- `src/lily_contact_planner/checker.py` — 独立 Level-1 幾何 checker
- `src/lily_contact_planner/tasks.py` — task path 定義
- `src/lily_contact_planner/planner_base.py` — 連続運動学成立性レイヤ
- `src/lily_contact_planner/planner_touchdown.py` — touchdown 生成・局所接触候補評価
- `src/lily_contact_planner/planner_search.py` — DFS/backtracking 接触探索
- `src/lily_contact_planner/visualization.py` — 表示補間と touchdown-before-liftoff の接触切替表示
- `src/lily_contact_planner/unified_planner.py` — public planner class
- `scripts/run_rollwalk_720.py` — 0°→720° 基準タスク再現スクリプト
- `docs/formulation.md` — Level-1 数理定式化
- `docs/algorithm_solver.md` — solver 構造とコード対応
- `docs/validated_baseline.md` — baseline と検証範囲
- `docs/progress_20260813.md` — 現在の実験変更点と world-frame task
- `docs/visualization.md` — 表示専用補間の規約と制約
- `tests/test_visualization.py` — touchdown/両支持が liftoff より先であることを確認
- `results/` — baseline・実験結果

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_rollwalk_720.py
```

720° の full search は意図的に計算量が大きい研究用 proof of concept であり、現時点では real-time planner ではありません。

## 現在位置

この repository では、**最初の角度非依存 contact-planning baseline**を保持しつつ、world-frame 多軸タスクの検討結果も記録しています。

今後の主な課題は

- dense continuous rollout / certification
- touchdown-before-liftoff を表示だけでなく planner 本体の有限時間接触遷移として扱うこと
- finite-thickness self-collision
- 一般 joystick/task command
- 各離散接触 branch 内での連続最適化の強化

です。
