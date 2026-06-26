# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:18:24 2024

@author: bsmsa18b
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from dtaidistance import dtw
import warnings
import logging
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional, Any
import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def setup_logger(output_dir=None):
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if output_dir is not None:
        log_path = Path(output_dir) / 'trajectory_processing.log'
        log_handlers.append(logging.FileHandler(log_path, mode='w'))
    else:
        log_handlers.append(logging.FileHandler('trajectory_processing.log', mode='w'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    logger = logging.getLogger(__name__)
    return logger

# Ensure logger is always defined for both script and GUI/preview use
logger = setup_logger()

# Helper: convert plotting table (CSV/XLSX) to expected multi-index DataFrame
def table_df_to_multiindex(df_table: pd.DataFrame) -> pd.DataFrame:
    # Ensure first column is string and subsequent numeric columns are floats
    try:
        first_col = df_table.columns[0]
        df_table[first_col] = df_table[first_col].astype(str)
    except Exception:
        pass
    
    for c in df_table.columns[1:]:
        try:
            df_table[c] = pd.to_numeric(df_table[c], errors='coerce')
        except Exception:
            pass
    
    # Standardize column names to strings
    df_table.columns = [str(c) for c in df_table.columns]
    
    # Identify larva columns
    larva_cols = [col for col in df_table.columns if str(col).lower().startswith('larva(')]
    if len(larva_cols) == 0:
        # Fallback: treat all columns except first as larvae
        larva_cols = [col for col in df_table.columns[1:]]
    
    # Separate x/y rows by label in first column
    x_rows = df_table[df_table.iloc[:, 0].str.contains('mom_x', case=False, na=False)]
    y_rows = df_table[df_table.iloc[:, 0].str.contains('mom_y', case=False, na=False)]
    
    records = []
    index = []
    
    for i, col in enumerate(larva_cols):
        x = x_rows[col].to_numpy()
        y = y_rows[col].to_numpy()
        maxlen = max(len(x), len(y))
        
        if len(x) < maxlen:
            x = np.pad(x, (0, maxlen - len(x)), constant_values=np.nan)
        if len(y) < maxlen:
            y = np.pad(y, (0, maxlen - len(y)), constant_values=np.nan)
        
        records.append({'x': x, 'y': y})
        index.append(('Homogeneous', 'Agar', 'Default', 'Exp1', f'L{i}'))
    
    midx = pd.MultiIndex.from_tuples(index, names=['condition', 'substrate', 'other', 'experiment', 'larva'])
    return pd.DataFrame(records, index=midx)

# Helper: convert internal multi-index DataFrame back to plotting CSV layout
def multiindex_df_to_plotting_csv(df: pd.DataFrame, output_file: Path) -> None:
    table = multiindex_df_to_plotting_table(df)
    table.to_csv(output_file, index=False)

# Helper: build plotting-format table (as DataFrame) from internal multi-index DataFrame
def multiindex_df_to_plotting_table(df: pd.DataFrame) -> pd.DataFrame:
    # Determine larva labels and numeric indices (L{n} -> n), preserving numeric order
    larva_labels = [idx[-1] for idx in df.index]
    larva_nums = []
    
    for label in larva_labels:
        try:
            larva_nums.append(int(str(label).replace('L', '')))
        except Exception:
            larva_nums.append(len(larva_nums))
    
    order = np.argsort(larva_nums)
    larva_nums_sorted = [larva_nums[i] for i in order]
    larva_rows = [df.iloc[i] for i in order]
    larva_cols = [f"larva({n})" for n in larva_nums_sorted]
    
    # Determine max length across larvae
    maxlen = 0
    for row in larva_rows:
        maxlen = max(maxlen, len(row['x']), len(row['y']))
    
    # Build data dict with padded arrays
    data = {}
    for col_name, row in zip(larva_cols, larva_rows):
        x = row['x']
        y = row['y']
        
        if len(x) < maxlen:
            x = np.pad(x, (0, maxlen - len(x)), constant_values=np.nan)
        if len(y) < maxlen:
            y = np.pad(y, (0, maxlen - len(y)), constant_values=np.nan)
        
        data[col_name] = (x, y)
    
    # Compose rows: mom_x(i), then mom_y(i)
    rows = []
    for coord in ['x', 'y']:
        for i in range(maxlen):
            row_label = f"mom_{coord}({i})"
            row_vals = [row_label]
            for col in larva_cols:
                arr = data[col][0 if coord == 'x' else 1]
                row_vals.append(arr[i])
            rows.append(row_vals)
    
    header = ['Index'] + larva_cols
    return pd.DataFrame(rows, columns=header)

# Generic loader supporting pickle, CSV and Excel plotting formats
def load_input_dataframe(input_file: Path) -> pd.DataFrame:
    n = str(input_file.name).lower()
    
    if n.endswith('.pkl') or n.endswith('.pickle') or n.endswith('.pkl.xz') or n.endswith('.xz'):
        return pd.read_pickle(input_file)
    elif n.endswith('.csv'):
        df_table = pd.read_csv(input_file, dtype={0: str})
        return table_df_to_multiindex(df_table)
    elif n.endswith('.xlsx') or n.endswith('.xls'):
        # read_excel may require openpyxl; ensure it's installed in the environment
        df_table = pd.read_excel(input_file)
        return table_df_to_multiindex(df_table)
    else:
        raise ValueError(f"Unsupported input file type: {input_file}")

class TrajectoryJoiner:

    def __init__(self, 
                 proximity_threshold: float = 5.0,
                 time_window: int = 200,
                 min_overlap: int = 10,
                 stop_threshold: float = 0.5,
                 collision_distance: float = 10.0,  # Maximum distance for collision detection
                 collision_time_gap: int = 10,      # Max frame gap for a join to count as a "collision"
                 config_file: Optional[Path] = None,
                 # Boundary / stop handling
                 max_stop_gap: int = 100,
                 boundary_margin: float = 5.0,
                 boundary_time_gap: int = 200,
                 boundary_distance: float = 150.0,
                 # Global optimization
                 w_dist: float = 1.0,
                 w_time: float = 1.0,
                 w_dtw: float = 1.0,
                 cost_threshold: float = 1e6,
                 dtw_missing_penalty: float = 3.0,  # cost stand-in when DTW can't be computed
                 # Termination detection (stopped / left-arena reporting)
                 termination_end_margin: int = 100,  # frames-before-end still counted as "tracked to the end"
                 stop_window: int = 10               # frames before a track ends used to test for stillness
                ):
        # Set all defaults first. A config file (if given) then overrides only
        # the keys it actually specifies. Previously the weights/cost_threshold
        # were assigned *after* load_config, so values coming from a config file
        # (e.g. the GUI's temp config) were clobbered back to defaults.
        self.proximity_threshold = proximity_threshold
        self.time_window = time_window
        self.min_overlap = min_overlap
        self.stop_threshold = stop_threshold
        self.collision_distance = collision_distance
        self.collision_time_gap = collision_time_gap
        self.max_stop_gap = max_stop_gap
        self.boundary_margin = boundary_margin
        self.boundary_time_gap = boundary_time_gap
        self.boundary_distance = boundary_distance
        self.w_dist = w_dist
        self.w_time = w_time
        self.w_dtw = w_dtw
        self.cost_threshold = cost_threshold
        self.dtw_missing_penalty = dtw_missing_penalty
        self.termination_end_margin = termination_end_margin
        self.stop_window = stop_window
        
        if config_file is not None:
            self.load_config(config_file)
        
        self.join_history = {}
        self.termination_analysis = {}  # per-experiment stopped / left-arena larvae
        self.arena_boundaries = {}  # Will store boundaries per experiment    

    def load_config(self, config_file: Path) -> None:
        """
        Load parameters from configuration file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Only override keys that are present; everything else keeps the
            # current (default) value. This lets a partial config file work.
            self.proximity_threshold = config.get('proximity_threshold', self.proximity_threshold)
            self.time_window = config.get('time_window', self.time_window)
            self.min_overlap = config.get('min_overlap', self.min_overlap)
            self.stop_threshold = config.get('stop_threshold', self.stop_threshold)
            self.collision_distance = config.get('collision_distance', self.collision_distance)
            self.collision_time_gap = config.get('collision_time_gap', self.collision_time_gap)
            self.max_stop_gap = config.get('max_stop_gap', self.max_stop_gap)
            self.boundary_margin = config.get('boundary_margin', self.boundary_margin)
            self.boundary_time_gap = config.get('boundary_time_gap', self.boundary_time_gap)
            self.boundary_distance = config.get('boundary_distance', self.boundary_distance)
            # Cost weights / thresholds were previously NOT read here, so values
            # set in the GUI or run script were silently ignored at run time.
            self.w_dist = config.get('w_dist', self.w_dist)
            self.w_time = config.get('w_time', self.w_time)
            self.w_dtw = config.get('w_dtw', self.w_dtw)
            self.cost_threshold = config.get('cost_threshold', self.cost_threshold)
            self.dtw_missing_penalty = config.get('dtw_missing_penalty', self.dtw_missing_penalty)
            self.termination_end_margin = config.get('termination_end_margin', self.termination_end_margin)
            self.stop_window = config.get('stop_window', self.stop_window)
            
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def save_config(self, config_file: Path) -> None:
        """Save current parameters to configuration file."""
        config = {
            'proximity_threshold': self.proximity_threshold,
            'time_window': self.time_window,
            'min_overlap': self.min_overlap,
            'stop_threshold': self.stop_threshold,
            'collision_distance': self.collision_distance,
            'collision_time_gap': self.collision_time_gap,
            'max_stop_gap': self.max_stop_gap,
            'boundary_margin': self.boundary_margin,
            'boundary_time_gap': self.boundary_time_gap,
            'boundary_distance': self.boundary_distance,
            'w_dist': self.w_dist,
            'w_time': self.w_time,
            'w_dtw': self.w_dtw,
            'cost_threshold': self.cost_threshold,
            'dtw_missing_penalty': self.dtw_missing_penalty,
            'termination_end_margin': self.termination_end_margin,
            'stop_window': self.stop_window,
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved configuration to {config_file}")

    def detect_arena_boundaries(self, df: pd.DataFrame, exp_name: str) -> None:
        """
        Detect arena boundaries for a specific experiment.
        """
        exp_mask = (df.index.get_level_values(3) == exp_name)
        exp_data = df[exp_mask]
        
        # Collect all valid x and y coordinates
        all_x = []
        all_y = []
        for _, row in exp_data.iterrows():
            valid_mask = ~np.isnan(row['x']) & ~np.isnan(row['y'])
            all_x.extend(row['x'][valid_mask])
            all_y.extend(row['y'][valid_mask])
        
        if len(all_x) == 0:
            logger.warning(f"No valid coordinates to detect boundaries for {exp_name}")
            return
        
        # Arena extent = trimmed data extent. The 0.5/99.5 percentiles discard
        # tracking outliers; we deliberately do NOT pad outward here, because
        # is_at_boundary tests proximity to these edges within `boundary_margin`.
        # (The old code padded by +10, which pushed the edge beyond every real
        # data point, so nothing could ever register as "at boundary".)
        x_min = float(np.percentile(all_x, 0.5))
        x_max = float(np.percentile(all_x, 99.5))
        y_min = float(np.percentile(all_y, 0.5))
        y_max = float(np.percentile(all_y, 99.5))
        
        self.arena_boundaries[exp_name] = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max
        }

    def is_at_boundary(self, x: float, y: float, exp_name: str) -> bool:
        """Check if a point is at the arena boundary."""
        if exp_name not in self.arena_boundaries:
            return False
            
        bounds = self.arena_boundaries[exp_name]
        margin = self.boundary_margin  # configurable (was hardcoded 5.0)
        
        return (abs(x - bounds['x_min']) < margin or
                abs(x - bounds['x_max']) < margin or
                abs(y - bounds['y_min']) < margin or
                abs(y - bounds['y_max']) < margin)

    def get_boundary_vector(self, x, y, idx, window=5, forward=True):
        """Calculate movement vector near boundary."""
        if forward:
            end_idx = min(idx + window, len(x))
            points = np.column_stack([x[idx:end_idx], y[idx:end_idx]])
        else:
            start_idx = max(0, idx - window)
            points = np.column_stack([x[start_idx:idx], y[start_idx:idx]])
        
        if len(points) >= 2:
            return points[-1] - points[0]
        return None
    
    def are_vectors_compatible(self, v1, v2, angle_threshold=np.pi/2):
        """Check if movement vectors are compatible."""
        if v1 is None or v2 is None:
            return False
        angle = np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0))
        return angle < angle_threshold

    def analyze_boundary_direction(self, x: np.ndarray, y: np.ndarray, 
                                 idx: int, window: int = 10) -> Dict[str, Any]:
        """Analyze movement direction near boundaries."""
        try:
            # Get valid points before/after index
            pre_idx = max(0, idx - window)
            post_idx = min(len(x), idx + window)
            
            pre_points = np.column_stack([x[pre_idx:idx], y[pre_idx:idx]])
            post_points = np.column_stack([x[idx:post_idx], y[idx:post_idx]])
            
            # Calculate movement vectors
            if len(pre_points) > 1:
                pre_vector = pre_points[-1] - pre_points[0]
                pre_angle = np.arctan2(pre_vector[1], pre_vector[0])
            else:
                pre_angle = None
                
            if len(post_points) > 1:
                post_vector = post_points[-1] - post_points[0]
                post_angle = np.arctan2(post_vector[1], post_vector[0])
            else:
                post_angle = None
                
            # Determine if movement is toward/away from boundary
            if pre_angle is not None and post_angle is not None:
                angle_diff = np.abs(post_angle - pre_angle)
                is_reversal = angle_diff > np.pi
                
                return {
                    'pre_angle': pre_angle,
                    'post_angle': post_angle,
                    'angle_diff': angle_diff,
                    'is_reversal': is_reversal,
                    'confidence': 1.0 - (angle_diff / (2 * np.pi))
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Direction analysis failed: {e}")
            return None
    
    def validate_boundary_match(self, df: pd.DataFrame, match: Dict, exp_name: str) -> float:
        """Validate potential boundary match using directional analysis."""
        larva1, larva2 = match['larva1'], match['larva2']
        
        # Get trajectory data
        base_mask = (df.index.get_level_values(3) == exp_name)
        mask1 = base_mask & (df.index.get_level_values(4) == larva1)
        mask2 = base_mask & (df.index.get_level_values(4) == larva2)
        
        traj1 = df[mask1].iloc[0]
        traj2 = df[mask2].iloc[0]
        
        # Analyze directions
        exit_analysis = self.analyze_boundary_direction(
            traj1['x'], traj1['y'], match['idx1']
        )
        entry_analysis = self.analyze_boundary_direction(
            traj2['x'], traj2['y'], match['idx2']
        )
        
        if exit_analysis and entry_analysis:
            angle_compatibility = np.cos(exit_analysis['pre_angle'] - 
                                       entry_analysis['post_angle'])
            
            confidence = (
                angle_compatibility * 0.4 +
                exit_analysis['confidence'] * 0.3 +
                entry_analysis['confidence'] * 0.3
            )
            
            return confidence
        
        return 0.0

    def detect_larva_resumption(self,
                             traj1_end: np.ndarray,
                             traj2_start: np.ndarray,
                             end_frame: int,
                             start_frame: int,
                             max_stop_gap: int = 100) -> Dict[str, Any]:
        """
        Detect if a trajectory is likely a resumed movement of a stopped larva.
        
        Args:
            traj1_end: End point of stopped trajectory [x, y]
            traj2_start: Start point of potential resumption [x, y]
            end_frame: Frame where first trajectory ends
            start_frame: Frame where second trajectory starts
            max_stop_gap: Maximum allowed frame gap for stopped larva
        
        Returns:
            Dictionary with detection information
        """
        spatial_distance = np.sqrt(np.sum((traj1_end - traj2_start) ** 2))
        temporal_gap = start_frame - end_frame
        
        # Very strict spatial threshold for stopped larvae
        stop_distance_threshold = self.collision_distance / 2
        
        is_resumption = (spatial_distance < stop_distance_threshold and 
                        temporal_gap <= max_stop_gap)
        
        resumption_info = {
            'is_resumption': is_resumption,
            'spatial_distance': spatial_distance,
            'temporal_gap': temporal_gap,
            'end_point': traj1_end,
            'start_point': traj2_start
        }
        
        if is_resumption:
            logger.info(f"\nPotential Movement Resumption Detected:")
            logger.info(f"  Spatial distance: {spatial_distance:.2f} units")
            logger.info(f"  Temporal gap: {temporal_gap} frames")
            logger.info(f"  Stop position: ({traj1_end[0]:.2f}, {traj1_end[1]:.2f})")
            logger.info(f"  Resume position: ({traj2_start[0]:.2f}, {traj2_start[1]:.2f})")
        
        return resumption_info

    def detect_collision(self, 
                        traj1_end: np.ndarray, 
                        traj2_start: np.ndarray,
                        end_frame: int,
                        start_frame: int,
                        max_frame_gap: int = 10) -> Dict[str, Any]:
        """
        Detect collisions considering both spatial and temporal proximity.
        
        Args:
            traj1_end: End point of first trajectory [x, y]
            traj2_start: Start point of second trajectory [x, y]
            end_frame: Frame where trajectory 1 ends
            start_frame: Frame where trajectory 2 starts
            max_frame_gap: Maximum allowed frame gap for collision
            
        Returns:
            Dictionary with collision information
        """
        spatial_distance = np.sqrt(np.sum((traj1_end - traj2_start) ** 2))
        temporal_gap = start_frame - end_frame
        
        is_collision = (spatial_distance < self.collision_distance and 
                       temporal_gap <= max_frame_gap)
        
        collision_info = {
            'is_collision': is_collision,
            'spatial_distance': spatial_distance,
            'temporal_gap': temporal_gap,
            'end_point': traj1_end,
            'start_point': traj2_start
        }
        
        if is_collision:
            logger.info(f"\nCollision Detection Details:")
            logger.info(f"  Spatial distance: {spatial_distance:.2f} units")
            logger.info(f"  Temporal gap: {temporal_gap} frames")
            logger.info(f"  End point: ({traj1_end[0]:.2f}, {traj1_end[1]:.2f})")
            logger.info(f"  Start point: ({traj2_start[0]:.2f}, {traj2_start[1]:.2f})")
        
        return collision_info


    def find_potential_matches(self, df: pd.DataFrame, exp_name: str, detailed_logging: bool = True) -> List[Dict]:
        """
        Use global optimization (Hungarian algorithm) to find optimal set of joins.
        Cost = weighted sum of spatial distance, temporal gap, and DTW similarity.
        Uses KDTree for fast spatial search and joblib.Parallel for parallel DTW scoring.
        """
        logger.info(f"[DEBUG] DataFrame type: {type(df)}, shape: {df.shape}")
        logger.info(f"[DEBUG] Parameters: proximity_threshold={self.proximity_threshold}, collision_distance={self.collision_distance}, time_window={self.time_window}, min_overlap={self.min_overlap}, w_dist={self.w_dist}, w_time={self.w_time}, w_dtw={self.w_dtw}, cost_threshold={self.cost_threshold}")
        
        matches = []
        
        if exp_name not in self.arena_boundaries:
            self.detect_arena_boundaries(df, exp_name)
            
        # Universal: select by experiment only, so any condition/substrate
        # (any arena type) is processed. Previously this hardcoded
        # condition=='Homogeneous' & substrate=='Agar', which silently
        # produced zero matches for every other arena type.
        exp_mask = (df.index.get_level_values(3) == exp_name)
        exp_data = df[exp_mask]
        larvae_ids = exp_data.index.get_level_values(4).unique()
        larvae_ids = sorted(larvae_ids, key=lambda x: int(x.replace('L', '')))
        n = len(larvae_ids)
        
        # Build KDTree for all start and end points
        end_points = []
        start_points = []
        last_valid_idxs = []
        first_valid_idxs = []
        
        for larva in larvae_ids:
            traj = exp_data.loc[exp_data.index.get_level_values(4) == larva].iloc[0]
            x = traj['x']
            y = traj['y']
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            
            if not np.any(valid_mask):
                end_points.append([np.nan, np.nan])
                start_points.append([np.nan, np.nan])
                last_valid_idxs.append(-1)
                first_valid_idxs.append(-1)
                continue
                
            last_idx = np.where(valid_mask)[0][-1]
            first_idx = np.where(valid_mask)[0][0]
            
            end_points.append([x[last_idx], y[last_idx]])
            start_points.append([x[first_idx], y[first_idx]])
            last_valid_idxs.append(last_idx)
            first_valid_idxs.append(first_idx)
        
        end_points = np.array(end_points)
        start_points = np.array(start_points)
        
        # Build KDTree for start points
        valid_start = ~np.isnan(start_points).any(axis=1)
        valid_end = ~np.isnan(end_points).any(axis=1)
        
        if not np.any(valid_start):
            logger.warning(f"No valid start points found for experiment {exp_name}")
            return []
            
        start_tree = KDTree(start_points[valid_start])
        
        # Candidate pairs: for each end point, find start points within threshold
        candidate_pairs = []
        
        for i, (larva1, end_pt, last_idx) in enumerate(zip(larvae_ids, end_points, last_valid_idxs)):
            if not valid_end[i] or last_idx == -1:
                continue
                
            # Query for all start points within max(proximity, collision) threshold
            dmax = max(self.proximity_threshold, self.collision_distance)
            idxs = start_tree.query_ball_point(end_pt, dmax)
            
            for j0 in idxs:
                # Map back to original index
                j = np.where(valid_start)[0][j0]
                larva2 = larvae_ids[j]
                
                if i == j:
                    continue
                    
                first_idx = first_valid_idxs[j]
                temporal_gap = first_idx - last_idx
                
                if temporal_gap <= 0 or temporal_gap > self.time_window:
                    continue
                    
                candidate_pairs.append((i, j, larva1, larva2, last_idx, first_idx))
        
        # Phase 1 (parallel): compute the raw features for each candidate pair.
        # Cost is intentionally NOT computed here, because the DTW term needs a
        # population-level scale (see Phase 2) to be meaningful.
        def score_pair(i, j, larva1, larva2, last_idx, first_idx):
            traj1 = exp_data.loc[exp_data.index.get_level_values(4) == larva1].iloc[0]
            traj2 = exp_data.loc[exp_data.index.get_level_values(4) == larva2].iloc[0]
            
            traj1_x = traj1['x']
            traj1_y = traj1['y']
            traj2_x = traj2['x']
            traj2_y = traj2['y']
            
            end_point = np.array([traj1_x[last_idx], traj1_y[last_idx]])
            start_point = np.array([traj2_x[first_idx], traj2_y[first_idx]])
            
            spatial_distance = np.linalg.norm(end_point - start_point)
            
            traj1_window = self.get_trajectory_window(traj1_x, traj1_y, last_idx, forward=False)
            traj2_window = self.get_trajectory_window(traj2_x, traj2_y, first_idx, forward=True)
            
            if len(traj1_window) < self.min_overlap or len(traj2_window) < self.min_overlap:
                return None
                
            similarity = self.calculate_dtw_similarity(traj1_window, traj2_window)
            
            return (i, j, spatial_distance, first_idx - last_idx, similarity, end_point, start_point)
        
        raw_results = Parallel(n_jobs=-1, prefer="threads")(delayed(score_pair)(*args) for args in candidate_pairs)
        
        # Derive a scale for the DTW term from the observed similarities so that
        # shape similarity actually influences the match. The old formulation
        # norm_dtw = similarity / (similarity + 1e-6) evaluated to ~1.0 for every
        # pair, which silently removed all DTW shape information from the cost.
        # Using the median makes "better than typical shape" lower the cost and
        # "worse than typical" raise it.
        finite_sims = [r[4] for r in raw_results if r is not None and np.isfinite(r[4])]
        dtw_scale = max(float(np.median(finite_sims)), 1e-6) if finite_sims else 1.0
        logger.info(f"[DTW] scale (median similarity) = {dtw_scale:.4f} over {len(finite_sims)} finite candidate(s)")
        
        # Phase 2: build cost matrix and metadata.
        costs = np.full((n, n), np.inf)
        meta = [[None for _ in range(n)] for _ in range(n)]
        all_distances = []
        all_time_gaps = []
        all_similarities = []
        
        for res in raw_results:
            if res is None:
                continue
                
            i, j, spatial_distance, temporal_gap, similarity, end_point, start_point = res
            
            norm_dist = spatial_distance / (self.proximity_threshold + 1e-6)
            norm_time = temporal_gap / (self.time_window + 1e-6)
            if np.isfinite(similarity):
                norm_dtw = similarity / dtw_scale
            else:
                # DTW could not be computed for this pair; apply a fixed penalty
                # instead of pretending the shapes matched perfectly.
                norm_dtw = self.dtw_missing_penalty
            
            cost = self.w_dist * norm_dist + self.w_time * norm_time + self.w_dtw * norm_dtw
            
            if cost > self.cost_threshold:
                continue
            
            all_distances.append(spatial_distance)
            all_time_gaps.append(temporal_gap)
            all_similarities.append(similarity)
            
            is_collision = bool(spatial_distance < self.collision_distance and
                                temporal_gap <= self.collision_time_gap)
            
            costs[i, j] = cost
            meta[i][j] = {
                'larva1': larvae_ids[i],
                'larva2': larvae_ids[j],
                'distance': spatial_distance,
                'temporal_gap': temporal_gap,
                'similarity': similarity,
                'idx1': last_valid_idxs[i],
                'idx2': first_valid_idxs[j],
                'is_collision': is_collision,
                'is_resumption': False,
                'at_boundary': self.is_at_boundary(end_point[0], end_point[1], exp_name),
                'end_point': end_point,
                'start_point': start_point,
                'match_type': 'collision' if is_collision else 'proximity'
            }
        
        # Log feature statistics for all candidate pairs
        if all_distances:
            logger.info(f"[Feature stats] Spatial distance: min={np.min(all_distances):.2f}, max={np.max(all_distances):.2f}, mean={np.mean(all_distances):.2f}, median={np.median(all_distances):.2f}")
            logger.info(f"[Feature stats] Temporal gap: min={np.min(all_time_gaps)}, max={np.max(all_time_gaps)}, mean={np.mean(all_time_gaps):.2f}, median={np.median(all_time_gaps):.2f}")
            logger.info(f"[Feature stats] DTW similarity: min={np.min(all_similarities):.2f}, max={np.max(all_similarities):.2f}, mean={np.mean(all_similarities):.2f}, median={np.median(all_similarities):.2f}")
        else:
            logger.info("[Feature stats] No candidate pairs for feature statistics.")
        
        # Log cost matrix shape before filtering
        logger.info(f"[DEBUG] Cost matrix shape before filtering: {costs.shape}")
        
        # Check for any finite costs
        finite_costs = costs[np.isfinite(costs)]
        if finite_costs.size == 0:
            logger.warning(f"No valid matches found for experiment {exp_name} (cost matrix is infeasible).")
            logger.info(f"Cost matrix shape: {costs.shape}")
            logger.info(f"Parameter summary: proximity_threshold={self.proximity_threshold}, collision_distance={self.collision_distance}, time_window={self.time_window}, min_overlap={self.min_overlap}, w_dist={self.w_dist}, w_time={self.w_time}, w_dtw={self.w_dtw}, cost_threshold={self.cost_threshold}")
            logger.info("No finite costs in cost matrix. Try relaxing thresholds or weights.")
            return []
        
        logger.info(f"Cost matrix stats for experiment {exp_name}: min={finite_costs.min():.4f}, max={finite_costs.max():.4f}, mean={finite_costs.mean():.4f}, median={np.median(finite_costs):.4f}, count={finite_costs.size}")
        
        # Remove all-infinite rows and columns before assignment
        valid_rows = np.any(np.isfinite(costs), axis=1)
        valid_cols = np.any(np.isfinite(costs), axis=0)
        
        logger.info(f"[DEBUG] Number of valid rows: {np.sum(valid_rows)}, indices: {np.where(valid_rows)[0].tolist()}")
        logger.info(f"[DEBUG] Number of valid cols: {np.sum(valid_cols)}, indices: {np.where(valid_cols)[0].tolist()}")
        
        if not np.any(valid_rows) or not np.any(valid_cols):
            logger.warning(f"No valid rows or columns for assignment in experiment {exp_name}.")
            return []
        
        reduced_costs = costs[np.ix_(valid_rows, valid_cols)]
        reduced_meta = np.array(meta, dtype=object)[np.ix_(valid_rows, valid_cols)]
        
        logger.info(f"[DEBUG] Reduced cost matrix shape: {reduced_costs.shape}")
        
        # Remove all-infinite rows/cols iteratively and log unjoinable larvae
        row_indices = np.where(valid_rows)[0]
        col_indices = np.where(valid_cols)[0]
        
        while True:
            row_all_inf = np.where(~np.any(np.isfinite(reduced_costs), axis=1))[0]
            col_all_inf = np.where(~np.any(np.isfinite(reduced_costs), axis=0))[0]
            
            if len(row_all_inf) == 0 and len(col_all_inf) == 0:
                break
                
            if len(row_all_inf) > 0:
                unjoinable_rows = [larvae_ids[row_indices[i]] for i in row_all_inf]
                logger.info(f"[Unjoinable] Base larvae with no valid partners: {unjoinable_rows}")
                keep = [i for i in range(reduced_costs.shape[0]) if i not in row_all_inf]
                reduced_costs = reduced_costs[keep, :]
                reduced_meta = reduced_meta[keep, :]
                row_indices = row_indices[keep]
                
            if len(col_all_inf) > 0:
                unjoinable_cols = [larvae_ids[col_indices[j]] for j in col_all_inf]
                logger.info(f"[Unjoinable] Join target larvae with no valid partners: {unjoinable_cols}")
                keep = [j for j in range(reduced_costs.shape[1]) if j not in col_all_inf]
                reduced_costs = reduced_costs[:, keep]
                reduced_meta = reduced_meta[:, keep]
                col_indices = col_indices[keep]
                
            if reduced_costs.size == 0:
                logger.warning(f"Reduced cost matrix became empty after removing unjoinable larvae in experiment {exp_name}.")
                return []
        
        # Now reduced_costs has no all-infinite rows/cols
        try:
            row_ind, col_ind = linear_sum_assignment(reduced_costs)
            
            if detailed_logging:
                logger.info(f"Hungarian assignment: {list(zip(row_ind, col_ind))}")
                print(f"Hungarian assignment: {list(zip(row_ind, col_ind))}")
                
            # Map back to original indices with proper bounds checking
            for i, j in zip(row_ind, col_ind):
                # Ensure indices are within bounds for reduced arrays
                if i < len(row_indices) and j < len(col_indices):
                    ri = row_indices[i]  # Original row index
                    cj = col_indices[j]  # Original col index
                    
                    # Ensure original indices are within bounds
                    if ri < costs.shape[0] and cj < costs.shape[1]:
                        if detailed_logging:
                            logger.info(f"Assignment: i={ri}, j={cj}, cost={costs[ri, cj]}, meta={meta[ri][cj]}")
                            print(f"Assignment: i={ri}, j={cj}, cost={costs[ri, cj]}, meta={meta[ri][cj]}")
                            
                        if np.isfinite(costs[ri, cj]) and meta[ri][cj] is not None:
                            matches.append(meta[ri][cj])
                            
            logger.info(f"Number of matches after assignment: {len(matches)}")
            print(f"Number of matches after assignment: {len(matches)}")
            
        except Exception as e:
            logger.error(f"Hungarian assignment failed: {e}")
            print(f"Hungarian assignment failed: {e}")
            
            # Fixed fallback: maximum bipartite matching on finite entries
            finite_mask = np.isfinite(reduced_costs)
            if np.any(finite_mask):
                graph = csr_matrix(finite_mask.astype(int))
                match_cols = maximum_bipartite_matching(graph, perm_type='column')
                # match_cols is length = number of columns; value is matched row or -1
                paired = [(r, c) for c, r in enumerate(match_cols) if r != -1]
                
                if paired:
                    logger.info(f"[Fallback] Maximum bipartite matching pairs: {paired}")
                    # Map back to original indices with proper bounds checking
                    for r, c in paired:
                        # Ensure indices are within bounds for reduced arrays
                        if r < len(row_indices) and c < len(col_indices):
                            ri = row_indices[r]  # Map back to original row index
                            cj = col_indices[c]  # Map back to original col index
                            
                            # Double check bounds for original matrix
                            if ri < costs.shape[0] and cj < costs.shape[1]:
                                if np.isfinite(costs[ri, cj]) and meta[ri][cj] is not None:
                                    matches.append(meta[ri][cj])
                        
                    logger.info(f"Number of matches after fallback matching: {len(matches)}")
                else:
                    logger.warning("[Fallback] No bipartite matches found among finite entries.")
            else:
                logger.warning("[Fallback] Reduced costs contain no finite entries for matching.")
        
        matches.sort(key=lambda x: (
            int(x['larva1'].replace('L', '')),
            not x['is_collision'],
            x['distance'],
            x['temporal_gap'],
            x['similarity']
        ))
        
        return matches


    def analyze_parameter_sensitivity(self, df: pd.DataFrame, exp_name: str) -> Dict[str, Any]:
        """
        Analyze parameter sensitivity to suggest optimal values for maximum joins.
        Returns comprehensive analysis with parameter recommendations.
        Fixed version with proper constraints and validation.
        """
        logger.info(f"[Parameter Analysis] Starting sensitivity analysis for {exp_name}")
        
        # Get experiment data
        # Universal: select by experiment only (any arena type).
        exp_mask = (df.index.get_level_values(3) == exp_name)
        exp_data = df[exp_mask]
        larvae_ids = exp_data.index.get_level_values(4).unique()
        larvae_ids = sorted(larvae_ids, key=lambda x: int(x.replace('L', '')))
        n_larvae = len(larvae_ids)
        
        # Collect all potential pair statistics
        pair_stats = []
        
        # Limit analysis to prevent performance issues
        max_larvae_to_analyze = min(100, n_larvae)  # Cap at 100 larvae for analysis
        larvae_sample = larvae_ids[:max_larvae_to_analyze]
        
        logger.info(f"[Parameter Analysis] Analyzing {len(larvae_sample)} larvae (sampled from {n_larvae} total)")
        
        for i, larva1 in enumerate(larvae_sample):
            for j, larva2 in enumerate(larvae_sample):
                if i >= j:
                    continue
                    
                try:
                    traj1 = exp_data.loc[exp_data.index.get_level_values(4) == larva1].iloc[0]
                    traj2 = exp_data.loc[exp_data.index.get_level_values(4) == larva2].iloc[0]
                    
                    x1, y1 = traj1['x'], traj1['y']
                    x2, y2 = traj2['x'], traj2['y']
                    
                    # Find valid segments
                    valid1 = ~np.isnan(x1) & ~np.isnan(y1)
                    valid2 = ~np.isnan(x2) & ~np.isnan(y2)
                    
                    if not np.any(valid1) or not np.any(valid2):
                        continue
                        
                    last_idx1 = np.where(valid1)[0][-1]
                    first_idx2 = np.where(valid2)[0][0]
                    
                    # Calculate metrics for all possible connections
                    end_point = np.array([x1[last_idx1], y1[last_idx1]])
                    start_point = np.array([x2[first_idx2], y2[first_idx2]])
                    spatial_dist = np.linalg.norm(end_point - start_point)
                    temporal_gap = first_idx2 - last_idx1
                    
                    # Skip unrealistic pairs
                    if spatial_dist > 100 or temporal_gap <= 0 or temporal_gap > 1000:
                        continue
                    
                    # Calculate DTW only for promising pairs to save time
                    dtw_similarity = float('inf')
                    if temporal_gap > 0 and spatial_dist < 50:
                        try:
                            traj1_window = self.get_trajectory_window(x1, y1, last_idx1, forward=False)
                            traj2_window = self.get_trajectory_window(x2, y2, first_idx2, forward=True)
                            if len(traj1_window) >= 5 and len(traj2_window) >= 5:
                                dtw_similarity = self.calculate_dtw_similarity(traj1_window, traj2_window)
                        except:
                            pass
                    
                    pair_stats.append({
                        'larva1': larva1,
                        'larva2': larva2,
                        'spatial_distance': spatial_dist,
                        'temporal_gap': temporal_gap,
                        'dtw_similarity': dtw_similarity,
                        'is_valid_sequence': temporal_gap > 0,
                        'end_point': end_point,
                        'start_point': start_point
                    })
                    
                except Exception as e:
                    continue
        
        # Convert to DataFrame for analysis
        stats_df = pd.DataFrame(pair_stats)
        valid_pairs = stats_df[stats_df['is_valid_sequence'] & (stats_df['temporal_gap'] <= 500)]
        
        logger.info(f"[Parameter Analysis] Found {len(valid_pairs)} valid sequential pairs out of {len(pair_stats)} total pairs")
        
        if len(valid_pairs) == 0:
            return {
                'recommendations': {'error': 'No valid sequential pairs found'},
                'analysis': {'total_pairs': len(pair_stats), 'valid_pairs': 0}
            }
        
        # Analyze distributions with outlier removal
        spatial_distances = valid_pairs['spatial_distance']
        temporal_gaps = valid_pairs['temporal_gap']
        
        # Remove extreme outliers (beyond 95th percentile)
        spatial_95 = spatial_distances.quantile(0.95)
        temporal_95 = temporal_gaps.quantile(0.95)
        
        spatial_clean = spatial_distances[spatial_distances <= spatial_95]
        temporal_clean = temporal_gaps[temporal_gaps <= temporal_95]
        
        spatial_stats = {
            'min': spatial_clean.min(),
            'max': spatial_clean.max(),
            'mean': spatial_clean.mean(),
            'median': spatial_clean.median(),
            'p25': spatial_clean.quantile(0.25),
            'p75': spatial_clean.quantile(0.75),
            'p90': spatial_clean.quantile(0.90),
            'p95': spatial_clean.quantile(0.95)
        }
        
        temporal_stats = {
            'min': temporal_clean.min(),
            'max': temporal_clean.max(),
            'mean': temporal_clean.mean(),
            'median': temporal_clean.median(),
            'p25': temporal_clean.quantile(0.25),
            'p75': temporal_clean.quantile(0.75),
            'p90': temporal_clean.quantile(0.90),
            'p95': temporal_clean.quantile(0.95)
        }
        
        # DTW analysis (excluding infinite values)
        finite_dtw = valid_pairs[valid_pairs['dtw_similarity'] != float('inf')]['dtw_similarity']
        dtw_stats = {}
        if len(finite_dtw) > 0:
            dtw_stats = {
                'min': finite_dtw.min(),
                'max': finite_dtw.max(),
                'mean': finite_dtw.mean(),
                'median': finite_dtw.median(),
                'p75': finite_dtw.quantile(0.75),
                'p90': finite_dtw.quantile(0.90)
            }
        
        # Generate constrained parameter recommendations
        recommendations = {}
        
        # Proximity threshold: be conservative, capture 75-85% of pairs
        # Use geometric mean to avoid extreme values
        prox_candidates = [
            spatial_stats['p75'] * 1.2,
            spatial_stats['p90'],
            spatial_stats['median'] * 2.0
        ]
        recommended_proximity = min(prox_candidates)
        # Apply hard constraints
        recommendations['proximity_threshold'] = max(5.0, min(30.0, recommended_proximity))
        
        # Collision distance: capture close pairs but be realistic
        coll_candidates = [
            spatial_stats['p25'],
            spatial_stats['median'] * 0.8,
            spatial_stats['p75'] * 0.5
        ]
        recommended_collision = min(coll_candidates)
        # Apply hard constraints  
        recommendations['collision_distance'] = max(3.0, min(15.0, recommended_collision))
        
        # Time window: capture most gaps but be reasonable
        time_candidates = [
            temporal_stats['p75'] * 1.5,
            temporal_stats['p90'],
            temporal_stats['median'] * 3.0
        ]
        recommended_time_window = min(time_candidates)
        # Apply hard constraints
        recommendations['time_window'] = max(50, min(500, int(recommended_time_window)))
        
        # Min overlap: conservative based on data availability
        min_overlap_candidate = max(3, min(15, int(temporal_stats['median'] * 0.1)))
        recommendations['min_overlap'] = min_overlap_candidate
        
        # Weight recommendations based on data characteristics but constrained
        dtw_ratio = len(finite_dtw) / len(valid_pairs) if len(valid_pairs) > 0 else 0
        
        if dtw_ratio > 0.3:  # If DTW is computable for >30% of pairs
            recommendations['w_dtw'] = 1.0  # Moderate DTW weight
            recommendations['w_dist'] = 1.0
            recommendations['w_time'] = 0.8
        else:  # If DTW is sparse, rely more on spatial-temporal
            recommendations['w_dtw'] = 0.3  # Low DTW weight
            recommendations['w_dist'] = 1.2
            recommendations['w_time'] = 1.0
        
        # Cost threshold: be reasonable, allow most good pairs
        # Base cost estimate using recommended parameters
        norm_dist = spatial_stats['p75'] / recommendations['proximity_threshold']
        norm_time = temporal_stats['p75'] / recommendations['time_window']
        norm_dtw = 1.0  # Assume moderate DTW
        
        base_cost = (norm_dist * recommendations['w_dist'] + 
                    norm_time * recommendations['w_time'] + 
                    norm_dtw * recommendations['w_dtw'])
        
        # Set cost threshold to allow 70-80% of reasonable pairs
        recommendations['cost_threshold'] = max(2.0, min(10.0, base_cost * 1.8))
        
        # Estimate potential joins with recommended parameters
        potential_joins = valid_pairs[
            (valid_pairs['spatial_distance'] <= recommendations['proximity_threshold']) &
            (valid_pairs['temporal_gap'] <= recommendations['time_window'])
        ]
        
        # Scale estimate to full dataset
        scale_factor = n_larvae / len(larvae_sample) if len(larvae_sample) > 0 else 1
        scaled_potential_joins = int(len(potential_joins) * scale_factor)
        
        # Conservative estimate: can't join more than half the larvae
        join_estimate = min(scaled_potential_joins, n_larvae // 2)
        
        analysis_summary = {
            'total_larvae': n_larvae,
            'analyzed_larvae': len(larvae_sample),
            'total_pairs': len(pair_stats),
            'valid_sequential_pairs': len(valid_pairs),
            'potential_joins_with_recommended_params': scaled_potential_joins,
            'estimated_final_larvae_count': n_larvae - join_estimate,
            'join_efficiency': f"{join_estimate}/{n_larvae} ({100*join_estimate/n_larvae:.1f}%)",
            'spatial_stats': spatial_stats,
            'temporal_stats': temporal_stats,
            'dtw_stats': dtw_stats,
            'dtw_coverage': f"{len(finite_dtw)}/{len(valid_pairs)} ({100*dtw_ratio:.1f}%)"
        }
        
        # Validation check
        for param, value in recommendations.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                logger.warning(f"Invalid recommendation for {param}: {value}, using fallback")
                fallbacks = {
                    'proximity_threshold': 10.0,
                    'collision_distance': 5.0,
                    'time_window': 200,
                    'min_overlap': 5,
                    'w_dist': 1.0,
                    'w_time': 1.0,
                    'w_dtw': 0.5,
                    'cost_threshold': 3.0
                }
                recommendations[param] = fallbacks.get(param, 1.0)
        
        logger.info(f"[Parameter Analysis] Generated recommendations: {recommendations}")
        
        return {
            'recommendations': recommendations,
            'analysis': analysis_summary,
            'pair_data': valid_pairs.to_dict('records')[:20]  # Sample of pairs for inspection
        }

    def get_trajectory(self, larva_id: str) -> Dict:
        """Get trajectory data for a larva."""
        mask = (self.df.index.get_level_values(4) == larva_id)
        traj = self.df[mask].iloc[0]
        return {'x': traj['x'], 'y': traj['y']}    
    
    def extract_trajectory(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Convert x, y coordinates into a trajectory array.
        
        Args:
            x: Array of x coordinates
            y: Array of y coordinates
            
        Returns:
            Array of [x, y] coordinates with NaN values removed
        """
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        return np.column_stack([x[valid_mask], y[valid_mask]])

    def calculate_dtw_similarity(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Calculate DTW similarity between two trajectories with separate x, y handling.
        
        Args:
            traj1: First trajectory array
            traj2: Second trajectory array
            
        Returns:
            Combined DTW distance for x and y coordinates
        """
        if len(traj1) < 2 or len(traj2) < 2:
            return float('inf')
        
        try:
            # Ensure trajectories are properly shaped and handle NaN values
            valid_mask1 = ~np.isnan(traj1[:, 0]) & ~np.isnan(traj1[:, 1])
            valid_mask2 = ~np.isnan(traj2[:, 0]) & ~np.isnan(traj2[:, 1])
            
            traj1_clean = traj1[valid_mask1]
            traj2_clean = traj2[valid_mask2]
            
            if len(traj1_clean) < 2 or len(traj2_clean) < 2:
                return float('inf')
            
            # Normalize trajectories independently for x and y
            traj1_norm_x = (traj1_clean[:, 0] - np.mean(traj1_clean[:, 0])) / (np.std(traj1_clean[:, 0]) + 1e-10)
            traj1_norm_y = (traj1_clean[:, 1] - np.mean(traj1_clean[:, 1])) / (np.std(traj1_clean[:, 1]) + 1e-10)
            
            traj2_norm_x = (traj2_clean[:, 0] - np.mean(traj2_clean[:, 0])) / (np.std(traj2_clean[:, 0]) + 1e-10)
            traj2_norm_y = (traj2_clean[:, 1] - np.mean(traj2_clean[:, 1])) / (np.std(traj2_clean[:, 1]) + 1e-10)
            
            # Calculate DTW separately for x and y coordinates.
            # distance_fast uses the compiled C path (much faster) but needs
            # contiguous float64 arrays and the C extension; fall back to the
            # pure-Python implementation if it is unavailable.
            dtw_x = self._dtw_distance(traj1_norm_x, traj2_norm_x)
            dtw_y = self._dtw_distance(traj1_norm_y, traj2_norm_y)
            
            # Combine distances with equal weighting
            combined_distance = dtw_x + dtw_y
            
            return combined_distance
            
        except Exception as e:
            logger.warning(f"DTW calculation failed: {e}")
            return float('inf')

    @staticmethod
    def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1-D DTW distance using the fast C path when available."""
        a = np.ascontiguousarray(a, dtype=np.double)
        b = np.ascontiguousarray(b, dtype=np.double)
        try:
            return dtw.distance_fast(a, b)
        except Exception:
            return dtw.distance(a, b)

    def get_trajectory_window(self, 
                            x: np.ndarray, 
                            y: np.ndarray, 
                            frame_idx: int, 
                            forward: bool = True) -> np.ndarray:
        """
        Extract trajectory window before or after given frame.
        
        Args:
            x: Array of x coordinates
            y: Array of y coordinates
            frame_idx: Reference frame index
            forward: If True, get window after frame_idx, else before
            
        Returns:
            Trajectory array for the specified window
        """
        if forward:
            end_idx = min(frame_idx + self.time_window, len(x))
            window_x = x[frame_idx:end_idx]
            window_y = y[frame_idx:end_idx]
        else:
            start_idx = max(0, frame_idx - self.time_window)
            window_x = x[start_idx:frame_idx]
            window_y = y[start_idx:frame_idx]
        
        return self.extract_trajectory(window_x, window_y)

    def check_stopped_larva(self, x: np.ndarray, y: np.ndarray, start_idx: int, window_size: int = 10) -> bool:
        """
        Check if larva has stopped moving.
        
        Args:
            x: Array of x coordinates
            y: Array of y coordinates
            start_idx: Starting frame index
            window_size: Number of frames to check for movement
            
        Returns:
            True if larva is considered stopped
        """
        end_idx = min(start_idx + window_size, len(x))
        if end_idx - start_idx < window_size:
            return False
            
        window_x = x[start_idx:end_idx]
        window_y = y[start_idx:end_idx]
        
        if np.any(np.isnan(window_x)) or np.any(np.isnan(window_y)):
            return False
            
        total_movement = np.sqrt(np.diff(window_x)**2 + np.diff(window_y)**2).sum()
        return total_movement < self.stop_threshold * window_size
  
    def check_boundary_reentry(self, df: pd.DataFrame, larva_id: str, exp_name: str, 
                             exit_frame: int, exit_pos: np.ndarray) -> List[Dict]:
        """Find potential reentries after boundary exit."""
        possible_matches = []
        max_time_gap = 200  # Maximum frames to look ahead
        spatial_threshold = 150.0  # Maximum distance for reentry
    
        exp_mask = (df.index.get_level_values(3) == exp_name)
        other_larvae = df[exp_mask].index.get_level_values(4)
        
        for other_id in other_larvae:
            if other_id == larva_id:
                continue
                
            other_mask = exp_mask & (df.index.get_level_values(4) == other_id)
            other_traj = df[other_mask].iloc[0]
            
            # Check if trajectory starts after exit
            valid_mask = ~np.isnan(other_traj['x']) & ~np.isnan(other_traj['y'])
            if not np.any(valid_mask):
                continue
                
            start_idx = np.where(valid_mask)[0][0]
            if start_idx < exit_frame or start_idx > exit_frame + max_time_gap:
                continue
                
            entry_pos = np.array([other_traj['x'][start_idx], other_traj['y'][start_idx]])
            distance = np.linalg.norm(entry_pos - exit_pos)
            
            if distance < spatial_threshold:
                possible_matches.append({
                    'larva_id': other_id,
                    'entry_frame': start_idx,
                    'entry_pos': entry_pos,
                    'distance': distance,
                    'time_gap': start_idx - exit_frame
                })
        
        return possible_matches
  
    def join_trajectories(self, df: pd.DataFrame, exp_name: str, match: Dict) -> pd.DataFrame:
        """
        Join two trajectories while preserving movement patterns and frame alignment using the actual frame indices from the CSV.
        For each frame, use base larva's data if it exists, else joined larva's data, else NaN.
        Handles different lengths and gaps robustly.
        """
        larva1, larva2 = match['larva1'], match['larva2']
        try:
            # Universal: select by experiment only (any arena type).
            base_mask = (df.index.get_level_values(3) == exp_name)
            mask1 = base_mask & (df.index.get_level_values(4) == larva1)
            mask2 = base_mask & (df.index.get_level_values(4) == larva2)
            
            traj1_idx = df[mask1].index[0]
            traj2_idx = df[mask2].index[0]
            traj1_x = df.at[traj1_idx, 'x']
            traj1_y = df.at[traj1_idx, 'y']
            traj2_x = df.at[traj2_idx, 'x']
            traj2_y = df.at[traj2_idx, 'y']
            
            # Get the full frame index from the longer of the two
            full_len = max(len(traj1_x), len(traj2_x))
            
            # If the DataFrame has an attribute for the full frame index, use it
            frame_index = None
            if hasattr(df, 'frame_index'):
                frame_index = df.frame_index
            else:
                # Try to infer from the length
                frame_index = list(range(full_len))
            
            # Build Series with frame index for both larvae
            s1_x = pd.Series(traj1_x, index=frame_index[:len(traj1_x)])
            s1_y = pd.Series(traj1_y, index=frame_index[:len(traj1_y)])
            s2_x = pd.Series(traj2_x, index=frame_index[:len(traj2_x)])
            s2_y = pd.Series(traj2_y, index=frame_index[:len(traj2_y)])
            
            # Union of all frame indices
            all_idx = sorted(set(s1_x.index).union(s2_x.index))
            
            # Merge by index: prefer s1, else s2, else NaN
            merged_x = s1_x.reindex(all_idx)
            merged_y = s1_y.reindex(all_idx)
            fill_x = s2_x.reindex(all_idx)
            fill_y = s2_y.reindex(all_idx)
            
            merged_x = merged_x.combine_first(fill_x)
            merged_y = merged_y.combine_first(fill_y)
            
            # Overlap detection
            overlap_info = {}
            overlap_x_idx = [i for i in all_idx if pd.notna(s1_x.get(i, np.nan)) and pd.notna(s2_x.get(i, np.nan))]
            if overlap_x_idx:
                logger.info(f"[Overlap] {larva1} and {larva2} have {len(overlap_x_idx)} overlapping frames in x: {overlap_x_idx}")
                logger.info(f"  {larva1} x values: {[s1_x[i] for i in overlap_x_idx]}")
                logger.info(f"  {larva2} x values: {[s2_x[i] for i in overlap_x_idx]}")
                overlap_info['x'] = {
                    'indices': overlap_x_idx,
                    larva1: [float(s1_x[i]) for i in overlap_x_idx],
                    larva2: [float(s2_x[i]) for i in overlap_x_idx],
                    'count': int(len(overlap_x_idx))
                }
            
            overlap_y_idx = [i for i in all_idx if pd.notna(s1_y.get(i, np.nan)) and pd.notna(s2_y.get(i, np.nan))]
            if overlap_y_idx:
                logger.info(f"[Overlap] {larva1} and {larva2} have {len(overlap_y_idx)} overlapping frames in y: {overlap_y_idx}")
                logger.info(f"  {larva1} y values: {[s1_y[i] for i in overlap_y_idx]}")
                logger.info(f"  {larva2} y values: {[s2_y[i] for i in overlap_y_idx]}")
                overlap_info['y'] = {
                    'indices': overlap_y_idx,
                    larva1: [float(s1_y[i]) for i in overlap_y_idx],
                    larva2: [float(s2_y[i]) for i in overlap_y_idx],
                    'count': int(len(overlap_y_idx))
                }
            
            # Convert merged Series back to numpy arrays
            merged_x_arr = merged_x.values
            merged_y_arr = merged_y.values
            
            updated_row = df.loc[traj1_idx].copy()
            updated_row['x'] = merged_x_arr
            updated_row['y'] = merged_y_arr
            
            if 'simple_trajectory' in df.columns:
                valid_mask = ~np.isnan(merged_x_arr) & ~np.isnan(merged_y_arr)
                new_traj = np.column_stack([merged_x_arr[valid_mask], merged_y_arr[valid_mask]])
                updated_row['simple_trajectory'] = new_traj
            
            if 'idx_turn_points' in df.columns:
                valid_points = ~np.isnan(merged_x_arr) & ~np.isnan(merged_y_arr)
                points = np.column_stack([merged_x_arr[valid_points], merged_y_arr[valid_points]])
                vectors = np.diff(points, axis=0)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                angle_diff = np.diff(angles)
                turn_points = np.where(np.abs(angle_diff) > np.pi/4)[0] + 1
                updated_row['idx_turn_points'] = turn_points
            
            df.loc[traj1_idx] = updated_row
            df = df.drop(index=traj2_idx)
            
            if exp_name not in self.join_history:
                self.join_history[exp_name] = []
            
            # Classify the join honestly. Previously every non-collision join was
            # labelled 'boundary', inflating boundary counts in the summary.
            if match.get('is_collision'):
                join_type = 'collision'
            elif match.get('match_type') == 'manual':
                join_type = 'manual'
            elif match.get('at_boundary'):
                join_type = 'boundary'
            else:
                join_type = 'proximity'
            
            join_info = {
                'larva1': larva1,
                'larva2': larva2,
                'frame': int(match['idx1']),
                'distance': float(match['distance']),
                'temporal_gap': int(match.get('temporal_gap', (match.get('idx2', 0) - match.get('idx1', 0)))) if isinstance(match.get('temporal_gap', None), (int, np.integer)) or match.get('temporal_gap', None) is not None else None,
                'similarity': float(match.get('similarity', float('nan'))),
                'join_type': join_type,
                'overlap': overlap_info
            }
            self.join_history[exp_name].append(join_info)
            
            return df
            
        except Exception as e:
            logger.error(f"Error joining trajectories: {str(e)}")
            raise

    def process_experiment(self, df: pd.DataFrame, exp_name: str, progress_callback=None) -> pd.DataFrame:
        """Process single experiment with refined joining criteria."""
        logger.info(f"Processing experiment: {exp_name}")
        df_exp = df.copy()
        
        try:
            iteration = 0
            while True:
                matches = self.find_potential_matches(df_exp, exp_name)
                logger.info(f"[Join Loop] Iteration {iteration}: Matches found = {len(matches)}")
                
                if not matches:
                    break
                
                # Apply only non-conflicting matches: no larva participates more than once in any role
                joins_applied = 0
                used_larvae = set()
                current_larvae = set(df_exp[df_exp.index.get_level_values(3) == exp_name].index.get_level_values(4))
                
                for match in matches:
                    larva1 = match.get('larva1')
                    larva2 = match.get('larva2')
                    
                    if larva1 in used_larvae or larva2 in used_larvae:
                        continue
                    
                    if larva1 not in current_larvae or larva2 not in current_larvae:
                        continue
                    
                    if match.get('is_collision'):
                        logger.info(f"Joining trajectories due to collision in experiment {exp_name}")
                    elif match.get('at_boundary'):
                        logger.info(f"Joining trajectories due to boundary exit/return in experiment {exp_name}")
                    
                    try:
                        df_exp = self.join_trajectories(df_exp, exp_name, match)
                        used_larvae.add(larva1)
                        used_larvae.add(larva2)
                        joins_applied += 1
                    except Exception as join_err:
                        logger.error(f"[Join Loop] Skipped join {larva1}->{larva2} due to error: {join_err}")
                        continue
                
                logger.info(f"[Join Loop] Iteration {iteration}: Applied {joins_applied} joins")
                iteration += 1
                
                # Optionally update progress here if progress_callback is provided
            
            return df_exp
            
        except Exception as e:
            logger.error(f"Error processing experiment {exp_name}: {str(e)}")
            raise

    def _was_stationary_before(self, x: np.ndarray, y: np.ndarray,
                               end_idx: int, window_size: int = None) -> bool:
        """Was the larva nearly stationary in the frames leading UP TO end_idx?

        This looks *backward* from the last tracked frame. (The old
        check_stopped_larva looked forward from the last valid frame, into
        frames that are NaN by definition, so it almost never fired.)
        """
        if window_size is None:
            window_size = self.stop_window
        start_idx = max(0, end_idx - window_size + 1)
        wx = x[start_idx:end_idx + 1]
        wy = y[start_idx:end_idx + 1]
        if len(wx) < 2 or np.any(np.isnan(wx)) or np.any(np.isnan(wy)):
            return False
        total_movement = np.sqrt(np.diff(wx) ** 2 + np.diff(wy) ** 2).sum()
        return total_movement < self.stop_threshold * len(wx)

    def _analyze_stop(self, x: np.ndarray, y: np.ndarray) -> Optional[Dict]:
        """Characterise the FIRST sustained stop in a trajectory.

        Returns a dict describing how long the larva moved before stopping, how
        long it then stayed still, and whether it moved again afterwards:

            stop_onset_frame          frame at which the larva first stopped
            stop_position             (x, y) at that frame
            moving_frames_before_stop frames from the first tracked frame to the stop
            stopped_frames            frames it stayed still (until it resumed,
                                      or until the last tracked frame if it never did)
            resumed                   True if sustained movement occurred after the stop
            last_tracked_frame        last frame the larva was tracked

        Stillness is judged on a *smoothed* (centred rolling-mean) speed rather
        than each raw frame step. This matches the averaging used to flag a larva
        as stopped, so a few jittery frames at the resting spot no longer break
        the stop and leave the durations blank. If no still region is found at
        all (very noisy track), a trailing fallback is used so the method always
        returns numbers when there are at least two tracked points.
        """
        valid = np.where(~np.isnan(x) & ~np.isnan(y))[0]
        if len(valid) < 2:
            return None
        xv = x[valid].astype(float)
        yv = y[valid].astype(float)
        # per-step speed = distance between consecutive valid points / frame gap
        speed = np.sqrt(np.diff(xv) ** 2 + np.diff(yv) ** 2) / np.maximum(np.diff(valid), 1)
        n = len(speed)

        # Centred rolling-mean of speed (jitter-tolerant stillness).
        w = max(1, int(self.stop_window) - 1)
        half = w // 2
        csum = np.cumsum(np.insert(speed, 0, 0.0))
        i = np.arange(n)
        lo = np.maximum(0, i - half)
        hi = np.minimum(n, i + half + 1)
        smooth = (csum[hi] - csum[lo]) / (hi - lo)
        # Match the flag's criterion exactly: it compares total window movement
        # to stop_threshold * (window POINTS), i.e. a per-step mean of
        # stop_threshold * window/(window-1). Using the same threshold here keeps
        # the stop breakdown consistent with whether the larva was flagged.
        thr = self.stop_threshold * (self.stop_window / max(1, self.stop_window - 1))
        still = smooth < thr
        # A stop only needs a short still run to be located, but a genuine
        # RESUMPTION must be sustained motion (a real crawl-away), not a couple
        # of jittery frames — otherwise noise at the resting spot reads as
        # "moved again". So require a longer run to call it a resumption.
        still_run_min = max(2, w // 2)
        move_run_min = max(3, int(self.stop_window))

        def runs(flags, want):
            """List of (start, length) for maximal runs equal to `want`."""
            out = []
            start = None
            for k, v in enumerate(flags):
                if bool(v) == want:
                    if start is None:
                        start = k
                elif start is not None:
                    out.append((start, k - start))
                    start = None
            if start is not None:
                out.append((start, len(flags) - start))
            return out

        still_runs = runs(still, True)
        if not still_runs:
            # No still region at all: treat the very end as the stop.
            last = int(valid[-1])
            return {
                'stop_onset_frame': last,
                'stop_position': (float(x[last]), float(y[last])),
                'moving_frames_before_stop': int(last - valid[0]),
                'stopped_frames': 1,
                'resumed': False,
                'last_tracked_frame': last,
            }

        # First run long enough to be a real stop; else the longest still run.
        chosen = next((r for r in still_runs if r[1] >= still_run_min),
                      max(still_runs, key=lambda r: r[1]))
        onset_step = chosen[0]
        onset_frame = int(valid[onset_step])
        moving_before = int(onset_frame - valid[0])

        # Sustained movement again after the stop?
        moving_after = [r for r in runs(still[onset_step:], False) if r[1] >= move_run_min]
        if moving_after:
            resume_frame = int(valid[onset_step + moving_after[0][0]])
            resumed = True
            stopped_frames = int(resume_frame - onset_frame)
        else:
            resumed = False
            stopped_frames = int(valid[-1] - onset_frame + 1)

        return {
            'stop_onset_frame': onset_frame,
            'stop_position': (float(x[onset_frame]), float(y[onset_frame])),
            'moving_frames_before_stop': moving_before,
            'stopped_frames': stopped_frames,
            'resumed': resumed,
            'last_tracked_frame': int(valid[-1]),
        }

    def analyze_terminations(self, exp_data: pd.DataFrame, exp_name: str) -> Dict[str, Dict]:
        """Classify why each larva's track ends.

        Returns {'stopped': {...}, 'left_arena': {...}}.

        A track that ends within `termination_end_margin` frames of the end of
        the recording is treated as "tracked to the end" and ignored. Of the
        tracks that end earlier:
          - if the last position is at the arena boundary  -> left_arena
          - else if it was nearly stationary just before    -> stopped
          - otherwise it is left unclassified (e.g. a brief loss / occlusion).

        Intended to be run on the *joined* data, so that an early termination
        that was merely an occlusion (and got reconnected) is not mislabelled.
        """
        if exp_name not in self.arena_boundaries:
            self.detect_arena_boundaries(exp_data, exp_name)

        stopped = {}
        left_arena = {}

        for idx in exp_data.index:
            larva_id = idx[-1]
            x = exp_data.loc[idx, 'x']
            y = exp_data.loc[idx, 'y']
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            if not np.any(valid_mask):
                continue

            last_valid_idx = int(np.where(valid_mask)[0][-1])
            # Tracked (essentially) to the end -> not a stop / exit event.
            if last_valid_idx >= len(x) - self.termination_end_margin:
                continue

            end_point = (float(x[last_valid_idx]), float(y[last_valid_idx]))

            if self.is_at_boundary(end_point[0], end_point[1], exp_name):
                left_arena[larva_id] = {
                    'exit_frame': last_valid_idx,
                    'exit_position': end_point,
                }
            elif self._was_stationary_before(x, y, last_valid_idx):
                stats = self._analyze_stop(x, y)
                if not stats:
                    # Final safety net: derive durations from the trailing window
                    # so a flagged larva never shows blank ('?') values.
                    first_valid = int(np.where(valid_mask)[0][0])
                    stats = {
                        'stop_onset_frame': last_valid_idx,
                        'stop_position': end_point,
                        'moving_frames_before_stop': int(last_valid_idx - first_valid),
                        'stopped_frames': int(self.stop_window),
                        'resumed': False,
                    }
                stopped[larva_id] = {
                    'stop_frame': stats.get('stop_onset_frame', last_valid_idx),
                    'stop_position': stats.get('stop_position', end_point),
                    'moving_frames_before_stop': stats.get('moving_frames_before_stop'),
                    'stopped_frames': stats.get('stopped_frames'),
                    'resumed': stats.get('resumed', False),
                    'last_tracked_frame': last_valid_idx,
                }

        return {'stopped': stopped, 'left_arena': left_arena}

    def analyze_stopped_larvae(self, exp_data: pd.DataFrame, exp_name: str) -> Dict:
        """Backward-compatible wrapper: returns only the 'stopped' larvae."""
        return self.analyze_terminations(exp_data, exp_name)['stopped']

    def remove_larvae(self, df: pd.DataFrame, to_remove) -> Tuple[pd.DataFrame, List]:
        """Drop the given larvae from df and return (clean_df, removed_pairs).

        `to_remove` may contain plain larva ids (e.g. 'L7', applied across all
        experiments) and/or (experiment, larva) tuples for precise removal.
        """
        plain = set()
        pairs = set()
        for item in to_remove:
            if isinstance(item, tuple):
                pairs.add(item)
            else:
                plain.add(item)

        lvl_exp = df.index.get_level_values(3)
        lvl_larva = df.index.get_level_values(4)
        mask = np.array([
            (lar in plain) or ((exp, lar) in pairs)
            for exp, lar in zip(lvl_exp, lvl_larva)
        ])
        removed = list(zip(lvl_exp[mask].tolist(), lvl_larva[mask].tolist()))
        logger.info(f"Removed {len(removed)} larvae: {removed}")
        return df[~mask], removed
    
    def log_experiment_summary(self, exp_name: str, summary: Dict, output_path: str = None, base_name: str = None):
        """
        Log detailed experiment summary and optionally export to a .txt file in the same directory as the output CSV.
        """
        lines = []
        lines.append(f"\nDetailed Summary for Experiment: {exp_name}")
        lines.append("-" * 50)
        lines.append(f"Initial number of larvae: {summary['initial_larvae']}")
        lines.append(f"Final number of larvae: {summary['final_larvae']}")
        lines.append(f"Total trajectory joins: {summary['total_joins']}")
        
        if summary['stopped_larvae']:
            lines.append("\nStopped Larvae (stopped moving inside the arena and were not reconnected):")
            for larva_id, info in summary['stopped_larvae'].items():
                moved = info.get('moving_frames_before_stop')
                still = info.get('stopped_frames')
                resumed = info.get('resumed', False)
                moved_s = f"{moved} frames" if moved is not None else "?"
                still_s = f"{still} frames" if still is not None else "?"
                resumed_s = "yes" if resumed else "no"
                lines.append(
                    f"  - Larva {larva_id}: stopped at frame {info['stop_frame']}, "
                    f"position {info['stop_position']}")
                lines.append(
                    f"        moved for {moved_s} before stopping; "
                    f"stayed still for {still_s}; moved again afterwards: {resumed_s}")
        
        if summary.get('boundary_exit_larvae'):
            lines.append("\nLeft-Arena Larvae (track ended at the boundary and did not return):")
            for larva_id, info in summary['boundary_exit_larvae'].items():
                lines.append(f"  - Larva {larva_id}: Exited at frame {info['exit_frame']}, position {info['exit_position']}")
        
        if summary['joins']:
            lines.append("\nTrajectory Joins:")
            for join in summary['joins']:
                join_type = join.get('join_type', 'standard')
                lines.append(f"  - Joined {join['larva1']} to {join['larva2']}")
                lines.append(f"    Frame: {join['frame']}, Type: {join_type}")
                lines.append(f"    Distance: {join['distance']:.2f}, Similarity: {join['similarity']:.2f}")
        
        # Generate data-driven suggestions
        potential_improvements = self.suggest_improvements(summary)
        if potential_improvements:
            lines.append("\nPotential Improvements:")
            for suggestion in potential_improvements:
                lines.append(f"  - {suggestion}")
        
        # Parameters used
        params = summary.get('parameters', {})
        if params:
            lines.append("\nParameters Used:")
            for k, v in params.items():
                lines.append(f"  - {k}: {v}")
        
        for line in lines:
            logger.info(line)
        
        if output_path:
            from pathlib import Path
            out_path = Path(output_path)
            # If output_path is a file, use its parent directory
            if out_path.is_file():
                out_dir = out_path.parent
            else:
                out_dir = out_path
            
            safe_base = base_name if base_name else f"join_summary_{exp_name}"
            summary_file = out_dir / f"{safe_base}_summary.txt"
            
            with open(summary_file, 'w') as f:
                for line in lines:
                    f.write(line + '\n')
            
            logger.info(f"Summary exported to {summary_file}")
            
            # Also export a concise CSV of joins for this experiment
            try:
                import csv
                pairs_file = out_dir / f"{safe_base}_join_pairs.csv"
                with open(pairs_file, 'w', newline='') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["larva1","larva2","frame","distance","temporal_gap","similarity","join_type"]) 
                    for j in summary['joins']:
                        writer.writerow([
                            j.get('larva1'), j.get('larva2'), j.get('frame'),
                            j.get('distance'), j.get('temporal_gap'), j.get('similarity'), j.get('join_type')
                        ])
                logger.info(f"Join pairs exported to {pairs_file}")
            except Exception as _:
                pass

    def suggest_improvements(self, summary: Dict) -> List[str]:
        """
        Generate data-driven suggestions for improving trajectory joining.
        Uses join history statistics to tailor recommendations.
        """
        suggestions: List[str] = []
        total = summary['initial_larvae']
        final_n = summary['final_larvae']
        joins = summary.get('joins', [])
        join_ratio = (len(joins) / total) if total else 0.0
        
        # Suggest relaxing thresholds if join ratio is low
        if join_ratio < 0.25:
            suggestions.append("Increase proximity_threshold and time_window; lower w_dtw and min_overlap to expand candidates")
        
        # Suggest tightening DTW if many joins but noisy
        if join_ratio > 0.6:
            suggestions.append("Increase w_dtw or min_overlap to prioritize shape consistency for remaining ambiguous pairs")
        
        # Analyze join types
        collision_joins = sum(1 for j in joins if j.get('join_type') == 'collision')
        boundary_joins = sum(1 for j in joins if j.get('join_type') == 'boundary')
        
        if collision_joins < boundary_joins:
            suggestions.append("Increase collision_distance slightly to capture short separations after collisions")
        
        # Check stopped larvae presence
        if len(summary.get('stopped_larvae', {})) > 0:
            suggestions.append("Several larvae stopped inside arena; consider increasing time_window and reducing stop_threshold")
        
        # Remaining large count
        if final_n > max(10, int(0.7 * total)):
            suggestions.append("Many trajectories remain; try raising cost_threshold and reducing w_dtw to allow more joins")
        
        return suggestions

    def process_all_data(self, df: pd.DataFrame, progress_callback=None, output_path=None, filename_for_summary=None) -> pd.DataFrame:
        """
        Process all experiments in the dataset with detailed tracking and analysis.
        Accepts an optional progress_callback(percent) for GUI progress updates.
        """
        df_processed = df.copy()
        analysis_summary = {}
        experiments = df.index.get_level_values(3).unique()
        
        logger.info(f"Found {len(experiments)} experiments: {experiments.tolist()}")
        
        total = len(experiments)
        for idx, exp_name in enumerate(experiments):
            try:
                exp_mask = df.index.get_level_values(3) == exp_name
                condition = df[exp_mask].index.get_level_values(2).unique()[0]
                initial_larvae = len(df[exp_mask])
                
                logger.info(f"\nProcessing experiment {exp_name} ({condition})")
                logger.info(f"Initial number of larvae: {initial_larvae}")
                
                df_processed = self.process_experiment(df_processed, exp_name)
                
                final_larvae = len(df_processed[df_processed.index.get_level_values(3) == exp_name])
                joins = self.join_history.get(exp_name, [])
                
                # Detect stopped / left-arena larvae on the JOINED data, so that
                # an early termination that was merely an occlusion (now
                # reconnected) is not mislabelled.
                exp_joined = df_processed[df_processed.index.get_level_values(3) == exp_name]
                terminations = self.analyze_terminations(exp_joined, exp_name)
                stopped_larvae = terminations['stopped']
                boundary_exit_larvae = terminations['left_arena']
                self.termination_analysis[exp_name] = terminations
                
                exp_summary = {
                    'initial_larvae': initial_larvae,
                    'final_larvae': final_larvae,
                    'total_joins': len(joins),
                    'stopped_larvae': stopped_larvae,
                    'boundary_exit_larvae': boundary_exit_larvae,
                    'joins': joins,
                    'parameters': {
                        'proximity_threshold': self.proximity_threshold,
                        'collision_distance': self.collision_distance,
                        'time_window': self.time_window,
                        'min_overlap': self.min_overlap,
                        'w_dist': self.w_dist,
                        'w_time': self.w_time,
                        'w_dtw': self.w_dtw,
                        'cost_threshold': self.cost_threshold,
                        'stop_threshold': self.stop_threshold
                    }
                }
                
                analysis_summary[exp_name] = exp_summary
                
                # Build a base name from original file stem where possible
                try:
                    if filename_for_summary:
                        input_stem = Path(filename_for_summary).stem
                    else:
                        input_stem = Path(original_filename).stem
                except Exception:
                    input_stem = f"join_summary_{exp_name}"
                
                self.log_experiment_summary(exp_name, exp_summary, output_path=output_path, base_name=input_stem)
                
            except Exception as e:
                logger.error(f"Error processing experiment {exp_name}: {str(e)}")
                continue
            
            if progress_callback:
                percent = 100 * (idx + 1) / total
                progress_callback(percent)
        
        self.analysis_summary = analysis_summary
        return df_processed

    def save_results(self, 
                    df: pd.DataFrame, 
                    output_path: Path, 
                    original_filename: str,
                    save_format: str = 'csv') -> Tuple[Path, Path, str]:
        """
        Save processed data and joining history with proper type conversion.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = f"{Path(original_filename).stem}_joined"
        
        # Save processed DataFrame
        if save_format == 'pkl.xz':
            output_filename = output_path / f"{base_name}.pkl.xz"
            df.to_pickle(output_filename)
        elif save_format == 'csv':
            output_filename = output_path / f"{base_name}.csv"
            # Write plotting-format CSV with only remaining joined larvae as columns
            multiindex_df_to_plotting_csv(df, output_filename)
            try:
                num_larvae_out = len(df.index)
                logger.info(f"Exported {num_larvae_out} larvae columns to {output_filename}")
            except Exception:
                pass
        elif save_format == 'xlsx' or save_format == 'excel':
            output_filename = output_path / f"{base_name}.xlsx"
            table = multiindex_df_to_plotting_table(df)
            try:
                with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                    table.to_excel(writer, sheet_name='joined_trajectories', index=False)
            except Exception:
                # Fallback to default engine
                table.to_excel(output_filename, sheet_name='joined_trajectories', index=False)
            try:
                num_larvae_out = table.shape[1] - 1  # minus the index column
                logger.info(f"Exported {num_larvae_out} larvae columns to {output_filename}")
            except Exception:
                pass
        elif save_format == 'parquet':
            output_filename = output_path / f"{base_name}.parquet"
            df.to_parquet(output_filename)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        # Convert numpy types to Python native types in join history
        converted_history = {}
        for exp_name, joins in self.join_history.items():
            converted_joins = []
            for join in joins:
                converted_join = {
                    'larva1': str(join['larva1']),  # Convert to string
                    'larva2': str(join['larva2']),  # Convert to string
                    'frame': int(join['frame']),    # Convert numpy.int64 to int
                    'distance': float(join['distance']), # Convert numpy.float64 to float
                    'similarity': float(join['similarity']) # Convert numpy.float64 to float
                }
                converted_joins.append(converted_join)
            converted_history[exp_name] = converted_joins
        
        # Save join history with converted types
        history_filename = output_path / f"join_history_{base_name}.json"
        with open(history_filename, 'w') as f:
            json.dump(converted_history, f, indent=2)
        
        logger.info(f"Results saved to {output_filename}")
        logger.info(f"Join history saved to {history_filename}")
        
        return output_filename, history_filename, base_name

def process_file(input_file: Path,
                output_path: Path,
                config_file: Optional[Path] = None,
                save_format: str = 'pkl.xz',
                filename_for_summary: str = None) -> Tuple[pd.DataFrame, Dict]:
    """Process a single trajectory file."""
    logger.info(f"Processing file: {input_file}")
    
    try:
        # Load data (pickle, CSV, or Excel)
        df = load_input_dataframe(input_file)
        
        # Log DataFrame structure
        logger.info(f"DataFrame index levels: {df.index.names}")
        logger.info(f"DataFrame columns: {df.columns}")
        logger.info(f"Number of rows: {len(df)}")
        
        joiner = TrajectoryJoiner(config_file=config_file)
        df_processed = joiner.process_all_data(df, output_path=output_path, filename_for_summary=filename_for_summary)
        
        joiner.save_results(df_processed, output_path, input_file.name, save_format)
        
        return df_processed, joiner.join_history
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

def process_directory(input_dir: Path,
                     output_path: Path,
                     config_file: Optional[Path] = None,
                     save_format: str = 'pkl.xz',
                     file_pattern: str = '**/data_all_rdp*.pkl.xz') -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Process all trajectory files in a directory.
    
    Args:
        input_dir: Directory containing input files
        output_path: Directory to save results
        config_file: Optional path to configuration file
        save_format: Format to save the processed data
        file_pattern: Pattern to match input files
    
    Returns:
        Dictionary mapping filenames to tuples of processed DataFrames and joining histories
    """
    logger.info(f"Processing directory: {input_dir}")
    results = {}
    
    for file_path in input_dir.glob(file_pattern):
        try:
            logger.info(f"Processing file: {file_path}")
            results[file_path.name] = process_file(file_path, output_path, config_file, save_format)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return results

def run_trajectory_processing(
    input_path: str,
    output_path: str,
    config_file: str = None,
    save_format: str = 'csv',
    file_pattern: str = '**/data_all_rdp*.pkl.xz',
    progress_callback=None,
    original_filename: str = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Interactive function to run trajectory processing from Spyder.
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)
        config_file = Path(config_file) if config_file else None
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Remove non-serializable items from kwargs before saving config
        serializable_kwargs = {k: v for k, v in kwargs.items() if not callable(v) and not hasattr(v, '__call__') and not isinstance(v, type(progress_callback))}
        
        # Create temporary config file if custom parameters are provided
        temp_config = None
        if serializable_kwargs:
            temp_config = output_path / 'temp_config.json'
            with open(temp_config, 'w') as f:
                json.dump(serializable_kwargs, f)
            config_file = temp_config
        
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            # Use original_filename if provided, otherwise use input_path name
            filename_for_summary = original_filename if original_filename else input_path.name
            result = process_file(input_path, output_path, config_file, save_format, filename_for_summary)
        elif input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
            result = process_directory(input_path, output_path, config_file, save_format, file_pattern)
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        if 'temp_config' in locals() and temp_config and temp_config.exists():
            temp_config.unlink()