# Trajectory Joiner — Documentation

Reconstruction of single-animal tracks from fragmented larva-tracking output.

---

## 1. What this is

When several larvae are tracked crawling on an arena, the tracker assigns each
animal an ID. Whenever it loses an animal — because two larvae collide or
occlude each other, because one crawls out of and back into the field of view,
or because one stops moving and is dropped then re-acquired — it gives the
*same* animal a **new** ID when it reappears. A single larva therefore ends up
split across several track fragments (`L0`, `L7`, `L12`, …).

This tool stitches those fragments back together, so that one physical larva is
represented by one continuous trajectory again. It does this by deciding, for
each fragment that *ends* somewhere mid-recording, which fragment that *starts*
shortly afterwards and nearby is most likely the same animal continuing.

The project has three Python files plus an environment spec:

| File | Role |
|------|------|
| `join_paths.py` | The engine. Contains `TrajectoryJoiner` and the matching/joining logic. |
| `trajectory_joiner_gui.py` | A single-window GUI: load a file, run automatic reconstruction *or* manual joins, save. |
| `run_join_paths.py` | A script driver for batch/Spyder use without the GUI. |
| `environment.yml` | Conda environment definition. |

Everything routes through `join_paths.py`; the GUI and the script are just
front-ends to it.

---

## 2. Installation

The only awkward dependency is `dtaidistance` (Dynamic Time Warping, compiled in
C). Install it and `pyarrow` from **conda-forge** so you get precompiled
binaries and never need a C compiler.

```bash
conda env create -f environment.yml
conda activate join_paths
```

or as a one-liner:

```bash
conda create -n join_paths -c conda-forge python=3.11 \
    numpy pandas scipy joblib dtaidistance openpyxl pyarrow
conda activate join_paths
```

`tkinter` (used by the GUI) ships with the conda `python` package, so there is
nothing extra to install for the GUI.

After install, confirm the fast DTW path is active:

```bash
python -c "from dtaidistance import dtw; dtw.try_import_c(verbose=True)"
```

If it reports the C library is available you get the fast path. If not, the code
still runs on the pure-Python fallback — just slower.

---

## 3. Input and output data format

### 3.1 Internal representation

Internally the data is a pandas `DataFrame` with a 5-level `MultiIndex` and two
columns, `x` and `y`. Each row is one trajectory; each `x`/`y` cell holds a full
**NumPy array** the length of the recording, with `NaN` at frames where that
larva was not tracked.

```
index levels: (condition, substrate, other, experiment, larva)
columns:       x, y           # each is a 1-D array of length n_frames

                                  x                         y
condition substrate other  experiment larva
Rough     Sand      Default Exp1       L0     [12.3, 12.9, …, nan, nan]  [4.1, 4.0, …]
                                       L1     [nan, nan, …, 40.2, 40.8]  [nan, …, 9.9]
```

Only the **experiment** level and the **larva** level matter for matching. The
first three levels (`condition`, `substrate`, `other`) are carried along but are
**not** used to decide joins — the tool works for **any arena type**. (Earlier
versions silently restricted processing to `condition == 'Homogeneous'` and
`substrate == 'Agar'`; that restriction has been removed.)

Larva labels must be of the form `L<integer>` (`L0`, `L1`, …) because the code
sorts and references them by their integer part.

### 3.2 Supported file types

`load_input_dataframe()` accepts:

- `.pkl`, `.pkl.xz`, `.pickle` — a pickled DataFrame already in the internal
  format above.
- `.csv` — a "plotting-format" table (see below), converted automatically.
- `.xlsx`, `.xls` — same plotting-format table in a spreadsheet.

The **plotting CSV format** has one column per larva (`larva(0)`, `larva(1)`, …)
and rows labelled `mom_x(0)`, `mom_x(1)`, … followed by `mom_y(0)`, `mom_y(1)`,
…, i.e. the x-coordinates over time for every larva, then the y-coordinates:

```
Index,larva(0),larva(1),larva(2)
mom_x(0),12.3,nan,80.1
mom_x(1),12.9,nan,80.4
...
mom_y(0),4.1,nan,5.0
mom_y(1),4.0,nan,5.2
...
```

When you load a CSV/XLSX it is given placeholder index levels
(`Homogeneous, Agar, Default, Exp1`) — these are just labels and have no effect
on matching.

### 3.3 Outputs

Running a job writes, into the output folder:

| File | Contents |
|------|----------|
| `<name>_joined.<ext>` | The reconstructed trajectories, in the chosen save format. |
| `join_history_<name>_joined.json` | Machine-readable record of every join (larva pair, frame, distance, similarity). |
| `<name>_summary.txt` | Human-readable per-experiment summary: counts, join list, suggestions, parameters used. |
| `<name>_join_pairs.csv` | One row per join with its metrics. |

Save formats: `csv` (plotting format, only the surviving larvae as columns),
`xlsx`, `pkl.xz`, `parquet`.

---

## 4. How it works

For each experiment independently, the engine runs this pipeline.

### 4.1 Arena boundary detection

`detect_arena_boundaries()` pools all valid coordinates in the experiment and
takes the 0.5th and 99.5th percentiles of x and y (plus a small margin) as the
arena extent. This is used only to *label* whether a fragment ended near the
edge (`at_boundary`), which in turn feeds the join-type reporting. It does not by
itself decide joins.

### 4.2 Candidate generation

For every fragment the engine records its **last valid point** (where it ends)
and **first valid point** (where it starts). A `KDTree` is built over all start
points. Then for each fragment `i` that ends mid-recording, it queries all
fragments `j` whose start lies within `max(proximity_threshold,
collision_distance)` of `i`'s end. A pair `(i → j)` survives as a candidate only
if the temporal gap is forward and bounded:

```
0 < (start_frame_of_j − end_frame_of_i) ≤ time_window
```

So a candidate is "fragment `i` ends, then a little later and nearby fragment `j`
begins." (Pairs that *overlap* in time — `j` starts before `i` ends — are not
currently considered; see Limitations.)

### 4.3 Cost of a candidate join

Each surviving candidate is scored by a weighted cost combining three signals:

```
cost(i→j) =  w_dist · (distance     / proximity_threshold)
           + w_time · (temporal_gap / time_window)
           + w_dtw  · (DTW          / DTW_scale)
```

- **distance** — Euclidean distance between `i`'s end point and `j`'s start
  point. Smaller is better.
- **temporal_gap** — number of frames between them. Smaller is better.
- **DTW** — Dynamic Time Warping distance between the *shape* of the tail of `i`
  (the frames leading up to its end) and the head of `j` (the frames after its
  start), computed separately on x and y and summed. A genuine continuation
  tends to keep heading the same way, so a lower DTW means the shapes line up.
  `DTW_scale` is the **median** DTW across all candidates in that experiment, so
  "better than typical shape" lowers the cost and "worse than typical" raises
  it. If DTW cannot be computed for a pair (too few points), the term is
  replaced by a fixed `dtw_missing_penalty` instead of being treated as a
  perfect match.

A candidate whose total cost exceeds `cost_threshold` is discarded.

### 4.4 Global assignment (Hungarian algorithm)

The surviving costs form an N×N matrix (rows = fragments that end, columns =
fragments that start). `scipy.optimize.linear_sum_assignment` finds the set of
one-to-one joins with the **lowest total cost** across the whole experiment at
once. This is better than greedily joining the closest pair first, because it
avoids one fragment "stealing" a partner that is a much better match for another
fragment. If the assignment fails for any reason, the engine falls back to a
maximum bipartite matching over the feasible pairs.

### 4.5 Performing a join

`join_trajectories()` merges fragment `j` (source) into fragment `i` (target):
it takes the union of their frame indices and, frame by frame, keeps the
target's value where present, otherwise the source's, otherwise `NaN`. The
source row is then dropped. The join is recorded in the history with a type:

- `collision` — distance below `collision_distance` and gap within
  `collision_time_gap`.
- `boundary` — the target ended near the arena edge.
- `manual` — created by you via manual joins.
- `proximity` — everything else.

### 4.6 Iteration

One physical larva can be split into three or more fragments. After applying a
round of joins, the engine **re-runs the whole matching** on the reduced set and
keeps going until no acceptable joins remain. Within a single round, no fragment
is allowed to take part in more than one join, so chains are built up one link
per round.

### 4.7 Stopped and left-arena detection

After joining, the engine classifies every surviving trajectory that **ends
early** — more than `termination_end_margin` frames before the end of the
recording (`analyze_terminations`). A track that runs essentially to the end is
considered complete and ignored. Of the early-ending tracks:

- if the last tracked position is at the arena boundary → **left_arena** (the
  larva crawled out of the field and did not return);
- otherwise, if the larva was nearly stationary in the `stop_window` frames
  *leading up to* its last tracked frame → **stopped** (it stopped moving inside
  the arena);
- otherwise it is left unclassified (e.g. a brief, unexplained loss).

The stationary test deliberately looks **backward** from the last tracked frame.
(An earlier version looked forward from that frame, into frames that are `NaN`
by definition, so it almost never flagged anything — the stopped-larva report
was effectively always empty. That is fixed.)

For each **stopped** larva the engine also characterises the stop itself
(`_analyze_stop`), reporting:

- `moving_frames_before_stop` — how many frames the larva was moving before it
  first stopped;
- `stopped_frames` — how long it then stayed still (until it resumed, or until
  its last tracked frame if it never did);
- `resumed` — whether sustained movement occurred again after the stop;
- `stop_frame` / `stop_position` — when and where it first stopped.

"Sustained" is judged on a short rolling-average of speed rather than each raw
frame, using the same averaging that decides the stopped flag — so a few jittery
frames at the resting spot neither break the stop nor read as a resumption, and
these durations are always reported for a flagged larva (never blank). A larva
that crawls and then halts for good reports `resumed = False` with a long
`stopped_frames`, while one that paused and then moved on reports `resumed = True`
with the duration of that pause. A resumption requires genuinely sustained motion
(a real crawl-away), not a brief noise spike. These appear in the summary and in
the GUI review list (e.g. `moved 99f -> still 51f, no resume`).

This classification runs on the **joined** data, so a fragment whose early end
was merely an occlusion that has since been reconnected is not mislabelled as a
stop or an exit. The results appear in the summary and drive the GUI's
**Review & remove** panel (§7.1). They are *diagnostic*: detection never deletes
anything on its own — removal is an explicit, separate step you take after
inspecting the list.

---

## 5. Parameters

All parameters are attributes of `TrajectoryJoiner` and can be set in the GUI,
passed to `run_join_paths.py`, or given to the constructor directly. They split
into those that **affect which joins happen** and those that only affect
**labelling / reporting**.

### 5.1 Matching parameters (these change the result)

| Parameter | Default | What it does | Increase it to… | Decrease it to… |
|-----------|---------|--------------|-----------------|-----------------|
| `proximity_threshold` | 8.0 | Max spatial gap (in your coordinate units, usually px) for a pair to be considered, and the scale that normalises the distance term. | Allow joins across bigger jumps (more joins, more risk of wrong ones). | Only join very close fragments (fewer, safer joins). |
| `time_window` | 250 | Max frame gap to bridge; also the DTW window length and the scale for the time term. | Reconnect fragments separated by longer drop-outs. | Only reconnect quickly-reappearing fragments. |
| `collision_distance` | 4.0 | Widens the candidate search radius if larger than `proximity_threshold`; also flags a join as a "collision". | Capture short separations right after collisions. | Treat fewer joins as collisions. |
| `min_overlap` | 5 | Minimum number of valid frames each fragment must have near the junction for DTW to be computed. | Demand more evidence before trusting a shape match. | Allow shape scoring on shorter fragments. |
| `cost_threshold` | 4.0 | Maximum total cost for a join to be accepted. **The main "how permissive am I" dial.** | Allow more (and weaker) joins. | Only accept high-confidence joins. |
| `w_dist` | 1.0 | Weight of the distance term in the cost. | Make spatial closeness matter more. | Make it matter less. |
| `w_time` | 1.0 | Weight of the temporal-gap term. | Penalise long gaps more. | Tolerate long gaps. |
| `w_dtw` | 1.0 | Weight of the DTW shape term. | Insist that the motion shapes line up. | Rely mostly on space and time. |
| `dtw_missing_penalty` | 3.0 | Cost stand-in when DTW can't be computed for a pair. | Make un-scoreable pairs less likely to join. | Be lenient about un-scoreable pairs. |
| `collision_time_gap` | 10 | Max frame gap for a join to be labelled (and prioritised as) a collision. | Treat slightly longer gaps as collisions. | Restrict the collision label. |

### 5.2 Reporting / labelling parameters (these don't change which joins happen)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `boundary_margin` | 5.0 | How close to the arena edge counts as "at boundary". Affects the `boundary` join label, the `at_boundary` flag, and the **left_arena** classification (§4.7). |
| `stop_threshold` | 0.5 | Total movement (over the `stop_window` frames before a track ends) below this marks a larva as "stopped" (§4.7). |
| `termination_end_margin` | 100 | A track ending within this many frames of the recording's end counts as "tracked to the end" and is not flagged as stopped or left-arena. |
| `stop_window` | 10 | Number of frames before a track ends used to test whether the larva had become stationary. |
| `save_format` | csv | Output file format: `csv`, `xlsx`, `pkl.xz`, `parquet`. |

### 5.3 Legacy parameters (currently inert in the main pipeline)

`boundary_time_gap`, `boundary_distance`, and `max_stop_gap` are read and stored,
but they are only used by helper methods (`check_boundary_reentry`,
`detect_larva_resumption`) that are **not called** by the Hungarian matching
path. Setting them has no effect on reconstruction today. They are kept for
backward compatibility and possible future use.

---

## 6. How to choose parameters

You will not get the right numbers by guessing; choose them from your own data.

### 6.1 Recommended workflow

1. **Load your file** and click **Preview / Recommend** (GUI) — or call
   `joiner.analyze_parameter_sensitivity(df, exp)`. This inspects the actual
   distribution of distances and time gaps between fragment ends and starts in
   your recording and prints suggested values plus an estimate of how many joins
   they would produce.
2. **Apply recommended** to load those values, then **Run automatic
   reconstruction**.
3. **Read the summary** (`*_summary.txt`) and the join list. Check a handful of
   joins against the raw video or a plot. Are they real continuations?
4. **Adjust and repeat:**
   - Too few joins (many fragments left, animals you know are continuous are
     still split) → raise `cost_threshold` first; then raise
     `proximity_threshold` and/or `time_window`. Lowering `w_dtw` also helps if
     shapes are noisy.
   - Too many / wrong joins (fragments stitched that clearly aren't the same
     animal) → lower `cost_threshold`; tighten `proximity_threshold` and
     `time_window`; raise `w_dtw` so motion shape has to agree.

### 6.2 Rules of thumb

- `proximity_threshold` should be a little larger than how far a larva can
  travel during a typical drop-out — roughly (max crawl speed) × (typical gap in
  frames). If most real gaps in your data are a few frames and a larva moves a
  few px per frame, a threshold of a few tens of px is usually sensible.
- `time_window` should comfortably cover your typical drop-out length but not be
  so large that two genuinely different animals get linked across a long gap.
- `cost_threshold` is the dial you will touch most. Start near the recommended
  value and move it up or down in small steps, re-checking the join list each
  time.
- The three weights only matter *relative* to each other. If your DTW coverage
  is low (the preview reports what fraction of pairs could be DTW-scored), lower
  `w_dtw` and lean on distance and time.

### 6.3 The Preview report

The preview prints, per experiment: the number of larvae, the number of valid
sequential pairs, the recommended parameters, the estimated number of joins, and
the spatial/temporal distributions (median and 90th percentile). Use the
percentiles directly: a `proximity_threshold` near the 90th percentile of
observed gap distances will admit most genuine continuations while excluding the
long tail.

---

## 7. Usage

### 7.1 GUI

```bash
python trajectory_joiner_gui.py
```

The window has three numbered sections:

1. **Files** — browse to the input file, pick an output folder, choose the save
   format.
2. **Parameters** — the five core parameters are always visible; the rest are
   behind the **Advanced parameters** toggle. Hover any field for a one-line
   explanation. Use **Preview / Recommend**, **Apply recommended**, and **Reset
   defaults** here.
3. **Run** — two actions:
   - **Run automatic reconstruction** — detects and joins fragments with the
     parameters above, then saves.
   - **Apply manual joins & save** — applies exactly the joins you type, saving
     to a separate `*_manual_joined` file so it does not overwrite automatic
     output.
4. **Review & remove larvae** — after a reconstruction run this list
   auto-populates with every **stopped** and **left-arena** larva (§4.7); you can
   also press **Detect stopped / left-arena larvae** at any time. Inspect the
   entries (each shows the larva, why it was flagged, the frame, and the
   position), select the rows you want to drop (and/or type extra IDs such as
   `L3, L9`), then press **Remove selected & save cleaned file**. This writes a
   new `*_cleaned_joined.<ext>` file with those larvae removed — your original
   reconstruction output is left untouched. The list refreshes against the
   cleaned data after removal.

Progress and a full log appear at the bottom.

**Manual join syntax.** In the manual-joins box, each group is
`target:source,source,…` and groups are separated by `;`. Sources are merged
into the target. Example:

```
0:3,6,15 ; 1:5,7
```

means: merge `L3`, `L6`, `L15` into `L0`, and merge `L5`, `L7` into `L1`.

### 7.2 Script (`run_join_paths.py`)

Edit the paths and the `processing_config` dictionary near the top, then run it.
It calls `run_trajectory_processing(...)` and prints a detailed per-experiment
summary. Use this for batch jobs or when you prefer not to open the GUI.

```python
processing_config = {
    'proximity_threshold': 75.0,
    'collision_distance': 50.0,
    'time_window': 300,
    'min_overlap': 5,
    'cost_threshold': 4.0,
    'w_dist': 1.0, 'w_time': 1.0, 'w_dtw': 1.0,
}
df_processed, join_history = run_trajectory_processing(
    input_path=input_path,
    output_path=output_path,
    save_format='pkl.xz',
    **processing_config,
)
```

### 7.3 As a library

```python
from join_paths import TrajectoryJoiner, load_input_dataframe
from pathlib import Path

df = load_input_dataframe(Path("data.csv"))
joiner = TrajectoryJoiner(
    proximity_threshold=20.0, time_window=200,
    min_overlap=5, cost_threshold=4.0,
    w_dist=1.0, w_time=1.0, w_dtw=1.0,
)
df_joined = joiner.process_all_data(df, output_path=Path("out"))
joiner.save_results(df_joined, Path("out"), "data.csv", save_format="csv")

# Inspect what happened
for exp, joins in joiner.join_history.items():
    for j in joins:
        print(exp, j["larva2"], "->", j["larva1"], j["join_type"])

# Detect stopped / left-arena larvae and remove them, then save a cleaned file
to_remove = []
for exp in df_joined.index.get_level_values(3).unique():
    exp_df = df_joined[df_joined.index.get_level_values(3) == exp]
    term = joiner.analyze_terminations(exp_df, exp)   # {'stopped':..., 'left_arena':...}
    to_remove += [(exp, lar) for lar in term["stopped"]]
    to_remove += [(exp, lar) for lar in term["left_arena"]]

clean_df, removed = joiner.remove_larvae(df_joined, to_remove)
joiner.save_results(clean_df, Path("out"), "data_cleaned.csv", save_format="csv")
```

---

## 8. Reading the log

A normal automatic run logs, per experiment:

- `[DTW] scale (median similarity) = …` — the shape-normalisation scale; if this
  is reported over very few candidates, shape scoring is sparse and you may want
  a lower `w_dtw`.
- `[Feature stats] Spatial distance / Temporal gap / DTW similarity …` — the
  distributions of the candidate pairs, useful for sanity-checking your
  thresholds.
- `Cost matrix stats …` — the spread of accepted costs.
- `[Unjoinable] …` — fragments that had no acceptable partner.
- `[Join Loop] Iteration k: Applied n joins` — how many joins each round made.

If you see `No valid matches found … (cost matrix is infeasible)`, no candidate
passed your filters: relax `cost_threshold`, `proximity_threshold`, or
`time_window`, or lower `w_dtw`.

---

## 9. Limitations and notes

- **Overlapping fragments are not joined.** A pair is only a candidate when the
  second fragment starts *after* the first ends. Fragments that overlap in time
  (common at the exact moment of a collision, where the tracker briefly holds
  both IDs) are skipped.
- **Overlap is resolved in favour of the target.** When a join does merge frames
  that both fragments cover, the target larva's coordinates are kept and the
  source's are dropped, without checking that the two agree. A wrong join is
  therefore not flagged by a coordinate clash.
- **`min_overlap` is a window length, not a temporal overlap.** Despite the
  name, it sets the minimum number of frames needed for DTW scoring, not a
  required overlap between the two fragments.
- **Frames are assumed 0-based and aligned.** Joining assumes both fragments use
  the same frame numbering (0 … n−1), which holds for the standard input format.
- **One experiment at a time.** Matching never crosses experiment boundaries,
  which is correct, but means fragments mislabelled into the wrong experiment
  will not be reconnected.
- **Manual joins on multi-experiment files** are matched within whichever
  experiment the target larva belongs to; if the same `L<n>` exists in more than
  one experiment, check the log to confirm the intended one was used.

---

## 10. Quick reference

```text
Pipeline (per experiment):
  detect boundaries
  → find candidate end→start pairs   (KDTree within max(proximity, collision);
                                       0 < gap ≤ time_window; ≥ min_overlap frames)
  → score each:  cost = w_dist·(dist/proximity)
                      + w_time·(gap/time_window)
                      + w_dtw ·(DTW/median_DTW)        (drop if cost > cost_threshold)
  → Hungarian assignment (global lowest-cost one-to-one set)
  → merge each chosen pair (target kept, source merged in & dropped)
  → repeat until no acceptable joins remain

Outputs:
  <name>_joined.<ext>                 reconstructed trajectories
  join_history_<name>_joined.json     every join, machine-readable
  <name>_summary.txt                  human-readable summary
  <name>_join_pairs.csv               per-join metrics
```
