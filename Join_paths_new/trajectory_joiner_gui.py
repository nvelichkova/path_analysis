# -*- coding: utf-8 -*-
"""
Unified GUI for larva trajectory joining.

One window that lets you:
  1. Load an input file (.pkl.xz / .csv / .xlsx),
  2. Run automatic reconstruction (global cost-based joining), OR
     apply manual joins by larva number,
  3. Save the result (csv / xlsx / pkl.xz / parquet) with a join summary.

Drop this next to the patched `join_paths.py` and run:
    python trajectory_joiner_gui.py
"""

import os
import sys
import threading
import traceback
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

# Reuse the patched engine (single source of truth)
from join_paths import (
    run_trajectory_processing,
    TrajectoryJoiner,
    load_input_dataframe,
)

# ---------------------------------------------------------------------------
# Parameter definitions. Every key here is understood by TrajectoryJoiner and
# flows through run_trajectory_processing -> config -> load_config.
# ---------------------------------------------------------------------------
CORE_PARAMS = [
    ('proximity_threshold', 8.0, float, 'Max gap (px) between an end and a start to consider joining'),
    ('collision_distance',  4.0, float, 'Below this gap a join is treated as a collision'),
    ('time_window',         250, int,   'Max frame gap to bridge / DTW window length'),
    ('min_overlap',         5,   int,   'Min frames of trajectory needed for DTW shape scoring'),
    ('cost_threshold',      4.0, float, 'Reject candidate joins whose total cost exceeds this'),
]
ADVANCED_PARAMS = [
    ('w_dist',              1.0, float, 'Weight of spatial distance in the cost'),
    ('w_time',              1.0, float, 'Weight of temporal gap in the cost'),
    ('w_dtw',               1.0, float, 'Weight of DTW shape similarity in the cost'),
    ('dtw_missing_penalty', 3.0, float, 'Cost stand-in when DTW cannot be computed'),
    ('collision_time_gap',  10,  int,   'Max frame gap to label a join "collision"'),
    ('stop_threshold',      0.5, float, 'Movement below this counts as a stopped larva'),
    ('boundary_margin',     5.0, float, 'How close to the arena edge counts as "at boundary"'),
    ('boundary_time_gap',   200, int,   'Max frames between a boundary exit and re-entry'),
    ('boundary_distance',   150.0, float, 'Max distance for a boundary re-entry match'),
]
SAVE_FORMATS = ['csv', 'xlsx', 'pkl.xz', 'parquet']


def parse_manual_joins(text):
    """Parse manual joins of the form '0:3,6,15; 1:5,7'.

    Each group 'target:source,source,...' means join every source into target.
    Returns a list of (target:int, source:int) tuples.
    """
    joins = []
    text = (text or "").strip()
    if not text:
        return joins
    for group in (g.strip() for g in text.split(';') if g.strip()):
        if ':' not in group:
            continue
        left, right = group.split(':', 1)
        left = left.strip()
        if not left.isdigit():
            continue
        target = int(left)
        for src in (s.strip() for s in right.split(',') if s.strip()):
            if src.isdigit():
                joins.append((target, int(src)))
    return joins


def apply_manual_joins(df, manual_joins, joiner, log=print):
    """Apply a list of (target, source) manual joins to df using the engine.

    Returns the processed DataFrame. UI-free so it can be unit-tested.
    """
    df_processed = df.copy()
    experiments = list(pd.unique(df_processed.index.get_level_values(3)))
    if len(experiments) > 1:
        log(f"[Manual] Note: {len(experiments)} experiments present; manual joins "
            f"are matched within whichever experiment a larva belongs to.")

    for k, (target, source) in enumerate(manual_joins, start=1):
        larva1, larva2 = f"L{target}", f"L{source}"
        mask1 = df_processed.index.get_level_values(4) == larva1
        mask2 = df_processed.index.get_level_values(4) == larva2
        if not mask1.any() or not mask2.any():
            present = sorted(set(df_processed.index.get_level_values(4)))
            log(f"[Manual] Skipped {larva2}->{larva1}: not found. Present: {present}")
            continue

        # Resolve the experiment from the target larva's own index row.
        exp_name = df_processed[mask1].index.get_level_values(3)[0]

        traj1 = df_processed[mask1].iloc[0]
        traj2 = df_processed[mask2].iloc[0]
        v1 = ~pd.isna(traj1['x']) & ~pd.isna(traj1['y'])
        v2 = ~pd.isna(traj2['x']) & ~pd.isna(traj2['y'])
        idx1 = int(np.where(v1)[0][-1]) if v1.any() else 0   # last valid of target
        idx2 = int(np.where(v2)[0][0]) if v2.any() else 0    # first valid of source

        match = {
            'larva1': larva1, 'larva2': larva2,
            'distance': float('nan'), 'similarity': float('nan'),
            'idx1': idx1, 'idx2': idx2,
            'is_collision': False, 'is_resumption': False, 'at_boundary': False,
            'end_point': None, 'start_point': None, 'match_type': 'manual',
        }
        df_processed = joiner.join_trajectories(df_processed, exp_name, match)
        log(f"[Manual] Joined {larva2} -> {larva1}")
    return df_processed


def collect_termination_flags(df, joiner):
    """Flat list of flagged larvae across all experiments (UI-free, testable).

    Each entry: {'experiment', 'larva', 'status', 'frame', 'position'} where
    status is 'stopped' or 'left_arena'.
    """
    entries = []
    for exp in pd.unique(df.index.get_level_values(3)):
        term = joiner.analyze_terminations(
            df[df.index.get_level_values(3) == exp], exp)
        for lar, info in term['stopped'].items():
            entries.append({'experiment': exp, 'larva': lar, 'status': 'stopped',
                            'frame': info['stop_frame'], 'position': info['stop_position'],
                            'moving_frames_before_stop': info.get('moving_frames_before_stop'),
                            'stopped_frames': info.get('stopped_frames'),
                            'resumed': info.get('resumed', False)})
        for lar, info in term['left_arena'].items():
            entries.append({'experiment': exp, 'larva': lar, 'status': 'left_arena',
                            'frame': info['exit_frame'], 'position': info['exit_position']})
    # stable, readable order
    entries.sort(key=lambda e: (str(e['experiment']), e['status'],
                                int(str(e['larva']).lstrip('L') or 0)))
    return entries


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class _GuiLogHandler(logging.Handler):
    """Routes logging records into the GUI log box (thread-safe via gui.log)."""
    def __init__(self, gui):
        super().__init__()
        self.gui = gui

    def emit(self, record):
        try:
            self.gui.log(self.format(record))
        except Exception:
            pass


class TrajectoryJoinerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trajectory Joiner")
        self.geometry("780x860")
        self.minsize(720, 760)

        self.param_vars = {}        # name -> (StringVar, type)
        self.last_recommendations = {}
        self.last_result_df = None      # most recent reconstructed/loaded DataFrame
        self.last_input_name = None     # original input filename (for output naming)
        self._detected = []             # parallel to the detect listbox: (exp, larva) tuples
        self._busy = False

        self._build_widgets()

    # ---------- layout ----------
    def _build_widgets(self):
        pad = dict(padx=6, pady=4)

        # --- Files ---
        files = ttk.LabelFrame(self, text="1. Files")
        files.pack(fill='x', **pad)

        ttk.Label(files, text="Input file:").grid(row=0, column=0, sticky='e', padx=4, pady=4)
        self.input_entry = ttk.Entry(files, width=64)
        self.input_entry.grid(row=0, column=1, sticky='we', padx=4, pady=4)
        ttk.Button(files, text="Browse…", command=self._browse_input).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(files, text="Output folder:").grid(row=1, column=0, sticky='e', padx=4, pady=4)
        self.output_entry = ttk.Entry(files, width=64)
        self.output_entry.grid(row=1, column=1, sticky='we', padx=4, pady=4)
        ttk.Button(files, text="Browse…", command=self._browse_output).grid(row=1, column=2, padx=4, pady=4)

        ttk.Label(files, text="Save format:").grid(row=2, column=0, sticky='e', padx=4, pady=4)
        self.format_var = tk.StringVar(value='csv')
        ttk.Combobox(files, textvariable=self.format_var, values=SAVE_FORMATS,
                     state='readonly', width=12).grid(row=2, column=1, sticky='w', padx=4, pady=4)
        files.columnconfigure(1, weight=1)

        # --- Parameters ---
        params = ttk.LabelFrame(self, text="2. Parameters (automatic reconstruction)")
        params.pack(fill='x', **pad)

        core = ttk.Frame(params)
        core.pack(fill='x', padx=4, pady=2)
        self._add_param_rows(core, CORE_PARAMS)

        # Advanced (collapsible)
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_btn = ttk.Button(params, text="▸ Advanced parameters", command=self._toggle_advanced)
        self._adv_btn.pack(anchor='w', padx=4, pady=(2, 0))
        self._adv_frame = ttk.Frame(params)
        self._add_param_rows(self._adv_frame, ADVANCED_PARAMS)

        btns = ttk.Frame(params)
        btns.pack(fill='x', padx=4, pady=4)
        ttk.Button(btns, text="Preview / Recommend", command=self._preview).pack(side='left', padx=3)
        ttk.Button(btns, text="Apply recommended", command=self._apply_recommended).pack(side='left', padx=3)
        ttk.Button(btns, text="Reset defaults", command=self._reset_defaults).pack(side='left', padx=3)

        # --- Actions ---
        actions = ttk.LabelFrame(self, text="3. Run")
        actions.pack(fill='x', **pad)

        auto = ttk.Frame(actions)
        auto.pack(fill='x', padx=4, pady=4)
        self.auto_btn = ttk.Button(auto, text="▶  Run automatic reconstruction",
                                   command=self._run_auto)
        self.auto_btn.pack(side='left', padx=3)
        ttk.Label(auto, text="(detect & join fragments using the parameters above, then save)").pack(side='left', padx=6)

        ttk.Separator(actions, orient='horizontal').pack(fill='x', pady=4)

        ttk.Label(actions, text="Manual joins  —  format:  target:source,source ;  e.g.  0:3,6 ; 1:5",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor='w', padx=4)
        self.manual_text = scrolledtext.ScrolledText(actions, width=80, height=4,
                                                      font=("Consolas", 10))
        self.manual_text.pack(fill='x', padx=4, pady=4)
        self.manual_btn = ttk.Button(actions, text="▶  Apply manual joins & save",
                                     command=self._run_manual)
        self.manual_btn.pack(anchor='w', padx=4, pady=(0, 4))

        # --- Review & remove ---
        cleanup = ttk.LabelFrame(self, text="4. Review & remove larvae (after inspection)")
        cleanup.pack(fill='x', **pad)

        top = ttk.Frame(cleanup)
        top.pack(fill='x', padx=4, pady=2)
        ttk.Button(top, text="Detect stopped / left-arena larvae",
                   command=self._detect_terminations).pack(side='left', padx=3)
        ttk.Label(top, text="(auto-fills after a reconstruction run; "
                            "select rows below, then remove)").pack(side='left', padx=6)

        listwrap = ttk.Frame(cleanup)
        listwrap.pack(fill='x', padx=4, pady=2)
        sb = ttk.Scrollbar(listwrap, orient='vertical')
        self.detect_listbox = tk.Listbox(listwrap, selectmode='extended', height=6,
                                         font=("Consolas", 9), yscrollcommand=sb.set)
        sb.config(command=self.detect_listbox.yview)
        self.detect_listbox.pack(side='left', fill='x', expand=True)
        sb.pack(side='right', fill='y')

        man = ttk.Frame(cleanup)
        man.pack(fill='x', padx=4, pady=2)
        ttk.Label(man, text="Also remove these IDs (comma-sep, e.g. L3, L9):").pack(side='left')
        self.manual_remove_entry = ttk.Entry(man, width=30)
        self.manual_remove_entry.pack(side='left', padx=6)

        self.remove_btn = ttk.Button(cleanup, text="✖  Remove selected & save cleaned file",
                                     command=self._remove_and_save)
        self.remove_btn.pack(anchor='w', padx=4, pady=(2, 4))

        # --- Progress + log ---
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(self, variable=self.progress_var, maximum=100).pack(fill='x', **pad)

        logframe = ttk.LabelFrame(self, text="Log")
        logframe.pack(fill='both', expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(logframe, state='disabled',
                                                   font=("Consolas", 9), height=14)
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)

        self._reset_defaults(quiet=True)

    def _add_param_rows(self, parent, spec):
        ncols = 2
        for i, (name, default, ptype, help_text) in enumerate(spec):
            r, c = divmod(i, ncols)
            cell = ttk.Frame(parent)
            cell.grid(row=r, column=c, sticky='we', padx=6, pady=2)
            ttk.Label(cell, text=name, width=18).pack(side='left')
            var = tk.StringVar(value=str(default))
            ent = ttk.Entry(cell, textvariable=var, width=10)
            ent.pack(side='left')
            self.param_vars[name] = (var, ptype)
            self._add_tooltip(ent, help_text)
        for c in range(ncols):
            parent.columnconfigure(c, weight=1)

    def _add_tooltip(self, widget, text):
        # lightweight hover tooltip
        tip = {'win': None}

        def show(_):
            if tip['win'] or not text:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 24
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=text, background="#ffffe0", relief='solid', borderwidth=1,
                     justify='left', wraplength=320, font=("TkDefaultFont", 8)).pack()
            tip['win'] = tw

        def hide(_):
            if tip['win']:
                tip['win'].destroy()
                tip['win'] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _toggle_advanced(self):
        if self._adv_open.get():
            self._adv_frame.forget()
            self._adv_btn.config(text="▸ Advanced parameters")
            self._adv_open.set(False)
        else:
            self._adv_frame.pack(fill='x', padx=4, pady=2)
            self._adv_btn.config(text="▾ Advanced parameters")
            self._adv_open.set(True)

    # ---------- thread-safe helpers ----------
    def log(self, msg):
        self.after(0, self._log_main, str(msg))

    def _log_main(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def set_progress(self, pct):
        self.after(0, self.progress_var.set, float(pct))

    def _set_busy(self, busy):
        self._busy = busy
        state = 'disabled' if busy else 'normal'
        self.after(0, lambda: (self.auto_btn.config(state=state),
                               self.manual_btn.config(state=state),
                               self.remove_btn.config(state=state)))

    # ---------- file pickers ----------
    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[("Trajectory files", "*.pkl.xz *.pkl *.csv *.xlsx *.xls"),
                       ("All files", "*.*")])
        if path:
            self.input_entry.delete(0, 'end')
            self.input_entry.insert(0, path)
            # default the output folder to the input's folder if empty
            if not self.output_entry.get().strip():
                self.output_entry.insert(0, str(Path(path).parent))

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_entry.delete(0, 'end')
            self.output_entry.insert(0, path)

    # ---------- params ----------
    def get_params(self):
        out = {}
        for name, (var, ptype) in self.param_vars.items():
            raw = var.get().strip()
            try:
                out[name] = ptype(raw)
            except Exception:
                raise ValueError(f"Invalid value for '{name}': {raw!r}")
        return out

    def _reset_defaults(self, quiet=False):
        for name, default, ptype, _ in (CORE_PARAMS + ADVANCED_PARAMS):
            if name in self.param_vars:
                self.param_vars[name][0].set(str(default))
        if not quiet:
            self.log("[Params] Reset to defaults.")

    def _apply_recommended(self):
        if not self.last_recommendations:
            messagebox.showinfo("No recommendations",
                                "Run 'Preview / Recommend' first.")
            return
        applied = []
        for name, val in self.last_recommendations.items():
            if name in self.param_vars:
                self.param_vars[name][0].set(str(val))
                applied.append(f"{name}={val}")
        self.log("[Params] Applied recommended: " + ", ".join(applied))

    # ---------- validation ----------
    def _validate_io(self, need_output=True):
        infile = self.input_entry.get().strip()
        outdir = self.output_entry.get().strip()
        if not infile or not os.path.isfile(infile):
            messagebox.showerror("Error", "Please select a valid input file.")
            return None
        if need_output and (not outdir or not os.path.isdir(outdir)):
            messagebox.showerror("Error", "Please select a valid output folder.")
            return None
        return infile, outdir

    # ---------- preview ----------
    def _preview(self):
        if self._busy:
            return
        io = self._validate_io(need_output=False)
        if not io:
            return
        infile, _ = io
        self._set_busy(True)
        self.log("\n" + "=" * 60)
        self.log("Parameter preview / recommendation")
        threading.Thread(target=self._preview_worker, args=(infile,), daemon=True).start()

    def _preview_worker(self, infile):
        handler = self._attach_log_handler()
        try:
            params = self.get_params()
            df = load_input_dataframe(Path(infile))
            self.log(f"Loaded {df.shape[0]} trajectories; "
                     f"experiments: {list(pd.unique(df.index.get_level_values(3)))}")
            joiner = TrajectoryJoiner(**{k: v for k, v in params.items()
                                         if k in TrajectoryJoiner.__init__.__code__.co_varnames})
            recs = {}
            for exp in pd.unique(df.index.get_level_values(3)):
                result = joiner.analyze_parameter_sensitivity(df, exp)
                rec = result.get('recommendations', {})
                if 'error' in rec:
                    self.log(f"[{exp}] {rec['error']}")
                    continue
                an = result.get('analysis', {})
                self.log(f"\n[{exp}] larvae={an.get('total_larvae')}, "
                         f"valid pairs={an.get('valid_sequential_pairs')}, "
                         f"est. joins={an.get('potential_joins_with_recommended_params')}")
                for k, v in rec.items():
                    self.log(f"    {k}: {v}")
                recs = rec  # keep last experiment's recommendations
            self.last_recommendations = recs
            self.log("\nPreview complete. Use 'Apply recommended' to load these values.")
        except Exception as e:
            self.log(f"[Preview error] {e}\n{traceback.format_exc()}")
        finally:
            self._detach_log_handler(handler)
            self._set_busy(False)
            self.set_progress(0)

    # ---------- automatic run ----------
    def _run_auto(self):
        if self._busy:
            return
        io = self._validate_io()
        if not io:
            return
        infile, outdir = io
        try:
            params = self.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        save_format = self.format_var.get()
        self._set_busy(True)
        self.set_progress(0)
        self.log("\n" + "=" * 60)
        self.log("Automatic reconstruction started…")
        threading.Thread(target=self._auto_worker,
                         args=(infile, outdir, params, save_format), daemon=True).start()

    def _auto_worker(self, infile, outdir, params, save_format):
        handler = self._attach_log_handler()
        try:
            result = run_trajectory_processing(
                input_path=infile,
                output_path=outdir,
                save_format=save_format,
                original_filename=Path(infile).name,
                progress_callback=self.set_progress,
                **params,
            )
            self.set_progress(100)
            if isinstance(result, tuple):
                df_out, history = result
                n_final = len(df_out)
                n_joins = sum(len(v) for v in history.values())
                self.log(f"\nDone. Final trajectories: {n_final}; total joins: {n_joins}.")
                for exp, joins in history.items():
                    if joins:
                        self.log(f"  [{exp}] {len(joins)} join(s):")
                        for j in joins:
                            self.log(f"      {j['larva2']} -> {j['larva1']} "
                                     f"(frame {j['frame']}, {j.get('join_type','?')}, "
                                     f"dist {j['distance']:.1f})")
                # Keep the joined result and surface stopped / left-arena larvae
                # for review, computed on the JOINED data.
                self.last_result_df = df_out
                self.last_input_name = Path(infile).name
                self._detect_into_panel(df_out, params)
            else:
                self.log("Done. Multiple files processed; see the output folder.")
            self.log(f"Saved to: {outdir}")
        except Exception as e:
            self.log(f"[Run error] {e}\n{traceback.format_exc()}")
            self.after(0, lambda: messagebox.showerror("Processing error", str(e)))
        finally:
            self._detach_log_handler(handler)
            self._set_busy(False)
            self.set_progress(0)

    # ---------- manual run ----------
    def _run_manual(self):
        if self._busy:
            return
        io = self._validate_io()
        if not io:
            return
        infile, outdir = io
        manual = parse_manual_joins(self.manual_text.get("1.0", 'end'))
        if not manual:
            messagebox.showinfo("No manual joins",
                                "Enter at least one join, e.g.  0:3,6 ; 1:5")
            return
        try:
            params = self.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        save_format = self.format_var.get()
        self._set_busy(True)
        self.set_progress(0)
        self.log("\n" + "=" * 60)
        self.log(f"Manual joins started ({len(manual)} requested)…")
        threading.Thread(target=self._manual_worker,
                         args=(infile, outdir, manual, params, save_format), daemon=True).start()

    def _manual_worker(self, infile, outdir, manual, params, save_format):
        handler = self._attach_log_handler()
        try:
            df = load_input_dataframe(Path(infile))
            joiner = TrajectoryJoiner(**{k: v for k, v in params.items()
                                         if k in TrajectoryJoiner.__init__.__code__.co_varnames})
            df_out = apply_manual_joins(df, manual, joiner, log=self.log)
            # Reuse the engine's saver; tag the name so it doesn't overwrite auto output.
            name = f"{Path(infile).stem}_manual.x"
            out_file, hist_file, _ = joiner.save_results(
                df_out, Path(outdir), name, save_format)
            self.set_progress(100)
            self.log(f"\nDone. Saved to: {out_file}")
        except Exception as e:
            self.log(f"[Manual error] {e}\n{traceback.format_exc()}")
            self.after(0, lambda: messagebox.showerror("Manual join error", str(e)))
        finally:
            self._detach_log_handler(handler)
            self._set_busy(False)
            self.set_progress(0)

    # ---------- review & remove ----------
    def _detect_terminations(self):
        """Load the current input file and flag stopped / left-arena larvae."""
        if self._busy:
            return
        io = self._validate_io(need_output=False)
        if not io:
            return
        infile, _ = io
        try:
            params = self.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        self._set_busy(True)
        self.log("\n" + "=" * 60)
        self.log("Detecting stopped / left-arena larvae…")

        def worker():
            handler = self._attach_log_handler()
            try:
                df = load_input_dataframe(Path(infile))
                self.last_result_df = df
                self.last_input_name = Path(infile).name
                self._detect_into_panel(df, params)
            except Exception as e:
                self.log(f"[Detect error] {e}\n{traceback.format_exc()}")
            finally:
                self._detach_log_handler(handler)
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def _detect_into_panel(self, df, params):
        """Compute terminations on df and fill the review list (thread-safe)."""
        try:
            joiner = TrajectoryJoiner(**{k: v for k, v in params.items()
                                         if k in TrajectoryJoiner.__init__.__code__.co_varnames})
        except Exception:
            joiner = TrajectoryJoiner()

        entries = []  # (display_label, (experiment, larva))
        for exp in pd.unique(df.index.get_level_values(3)):
            exp_df = df[df.index.get_level_values(3) == exp]
            joiner.detect_arena_boundaries(exp_df, exp)
            term = joiner.analyze_terminations(exp_df, exp)
            for lar, info in term['stopped'].items():
                pos = tuple(round(c, 1) for c in info['stop_position'])
                moved = info.get('moving_frames_before_stop')
                still = info.get('stopped_frames')
                resumed = 'resumed' if info.get('resumed') else 'no resume'
                moved_s = f"{moved}f" if moved is not None else "?"
                still_s = f"{still}f" if still is not None else "?"
                entries.append(
                    (f"[{exp}] {lar:<5} STOPPED     @frame {info['stop_frame']:>5}  pos {pos}  "
                     f"(moved {moved_s} -> still {still_s}, {resumed})",
                     (exp, lar)))
            for lar, info in term['left_arena'].items():
                pos = tuple(round(c, 1) for c in info['exit_position'])
                entries.append((f"[{exp}] {lar:<5} LEFT ARENA  @frame {info['exit_frame']:>5}  pos {pos}",
                                (exp, lar)))

        self.after(0, self._populate_listbox, entries)
        self.log(f"[Detect] {len(entries)} stopped / left-arena larva(e) flagged for review.")

    def _populate_listbox(self, entries):
        self._detected = [tup for _, tup in entries]
        self.detect_listbox.delete(0, 'end')
        if not entries:
            self.detect_listbox.insert('end', "(none detected)")
            self._detected = []
            return
        for label, _ in entries:
            self.detect_listbox.insert('end', label)

    def _remove_and_save(self):
        if self._busy:
            return
        if self.last_result_df is None:
            messagebox.showinfo("Nothing loaded",
                                "Run automatic reconstruction or click "
                                "'Detect stopped / left-arena larvae' first.")
            return
        io = self._validate_io()
        if not io:
            return
        _, outdir = io
        try:
            params = self.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        selected = [self._detected[i] for i in self.detect_listbox.curselection()
                    if i < len(self._detected)]
        manual = []
        for tok in self.manual_remove_entry.get().split(','):
            tok = tok.strip()
            if not tok:
                continue
            manual.append(tok if tok.upper().startswith('L') else (f"L{tok}" if tok.isdigit() else tok))
        to_remove = list(selected) + manual
        if not to_remove:
            messagebox.showinfo("Nothing selected",
                                "Select rows in the list and/or type IDs to remove.")
            return

        save_format = self.format_var.get()
        self._set_busy(True)
        self.log("\n" + "=" * 60)
        self.log(f"Removing {len(to_remove)} larva(e) and saving cleaned file…")
        threading.Thread(target=self._remove_worker,
                         args=(outdir, to_remove, save_format, params), daemon=True).start()

    def _remove_worker(self, outdir, to_remove, save_format, params):
        handler = self._attach_log_handler()
        try:
            joiner = TrajectoryJoiner()
            clean, removed = joiner.remove_larvae(self.last_result_df, to_remove)
            self.last_result_df = clean
            stem = Path(self.last_input_name or "data").stem
            out_file, _, _ = joiner.save_results(clean, Path(outdir),
                                                 f"{stem}_cleaned.x", save_format)
            self.log(f"Removed {len(removed)} larva(e): {[l for _, l in removed]}")
            self.log(f"Cleaned file saved to: {out_file}")
            # refresh the list against the cleaned data so removed rows disappear
            self._detect_into_panel(clean, params)
        except Exception as e:
            self.log(f"[Remove error] {e}\n{traceback.format_exc()}")
            self.after(0, lambda: messagebox.showerror("Remove error", str(e)))
        finally:
            self._detach_log_handler(handler)
            self._set_busy(False)

    # ---------- logging plumbing ----------
    def _attach_log_handler(self):
        handler = _GuiLogHandler(self)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)
        return handler

    def _detach_log_handler(self, handler):
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:
            pass


if __name__ == "__main__":
    TrajectoryJoinerGUI().mainloop()
