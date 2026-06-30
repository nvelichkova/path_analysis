# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:18:13 2025

@author: bsmsa18b
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QSlider, QPushButton, QLabel, QComboBox, QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import pandas as pd
import numpy as np
from rdp import rdp
from tqdm import tqdm
import os

class TrajectoryViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scale = 8.0
        self.epsilon = 2.5
        self.data = None
        self.current_larva = 0
        self.global_bounds = None
        self.dot_size = 4
        self._zoom_selector = None
        self._zoom_active = False
        self._zoomed_in = False
        self._custom_limits = None
        self.loaded_files = []   # list of dicts: {name, path, data, bounds}
        self.setup_ui()
        self.csv_filename = None

    def load_data(self, csv_file):
        print("Reading CSV file...")
        data_aux = pd.read_csv(csv_file, dtype={0: str})
        data_aux.iloc[:, 1:] = data_aux.iloc[:, 1:].astype(float)
        
        processed_data = []
        n_larvae = data_aux.shape[1] - 1
        
        global_min_x = float('inf')
        global_max_x = float('-inf')
        global_min_y = float('inf')
        global_max_y = float('-inf')
        
        for i in tqdm(range(n_larvae)):
            x_coords = data_aux[data_aux.iloc[:, 0].str.contains('mom_x', case=False, na=False)]
            y_coords = data_aux[data_aux.iloc[:, 0].str.contains('mom_y', case=False, na=False)]
            
            x_data = x_coords.iloc[:, i + 1].values / self.scale
            y_data = y_coords.iloc[:, i + 1].values / self.scale
            
            valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            coords = np.column_stack([x_clean, y_clean])
            
            if len(x_clean) > 0:
                global_min_x = min(global_min_x, np.min(x_clean))
                global_max_x = max(global_max_x, np.max(x_clean))
                global_min_y = min(global_min_y, np.min(y_clean))
                global_max_y = max(global_max_y, np.max(y_clean))
            
            distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)) if len(coords) > 1 else np.array([])
            
            processed_data.append({
                'coords': coords,
                'distances': distances,
                'total_points': len(x_clean),
                'total_distance': np.sum(distances)
            })
        
        overall_min = min(global_min_x, global_min_y)
        overall_max = max(global_max_x, global_max_y)
        
        total_range = overall_max - overall_min
        padding = total_range * 0.1
        
        self.global_bounds = {
            'x_min': overall_min - padding,
            'x_max': overall_max + padding,
            'y_min': overall_min - padding,
            'y_max': overall_max + padding
        }
        
        self.xmin_spin.setValue(self.global_bounds['x_min'])
        self.xmax_spin.setValue(self.global_bounds['x_max'])
        self.ymin_spin.setValue(self.global_bounds['y_min'])
        self.ymax_spin.setValue(self.global_bounds['y_max'])
        
        return processed_data

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open CSV', '', 'CSV Files (*.csv)')
        if filename:
            try:
                entry = self._load_entry(filename)
            except Exception as e:
                QMessageBox.warning(self, 'Load failed', f'{os.path.basename(filename)}: {e}')
                return
            self._set_loaded_files([entry])

    def _make_progress(self, title, maximum):
        """Create a modal progress dialog with a Cancel button."""
        dlg = QProgressDialog(title, 'Cancel', 0, maximum, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)   # show immediately rather than after a delay
        dlg.setAutoClose(True)
        dlg.setValue(0)
        return dlg

    def load_folder(self):
        """Load every CSV in a folder into memory so you can browse between
        files/larvae, pick an epsilon visually, then process them all."""
        in_dir = QFileDialog.getExistingDirectory(
            self, 'Select folder containing CSV files')
        if not in_dir:
            return

        csv_files = [f for f in sorted(os.listdir(in_dir))
                     if f.lower().endswith('.csv')]
        if not csv_files:
            QMessageBox.warning(self, 'No CSV files',
                                'No .csv files were found in that folder.')
            return

        progress = self._make_progress('Loading files', len(csv_files))
        entries, errors, cancelled = [], [], False
        for i, fname in enumerate(csv_files):
            if progress.wasCanceled():
                cancelled = True
                break
            progress.setLabelText(f'Loading {fname}  ({i + 1}/{len(csv_files)})')
            progress.setValue(i)
            QApplication.processEvents()
            path = os.path.join(in_dir, fname)
            try:
                entries.append(self._load_entry(path))
            except Exception as e:
                errors.append(f'{fname}: {e}')
        progress.setValue(len(csv_files))   # closes the dialog

        if not entries:
            QMessageBox.warning(self, 'Nothing loaded',
                                'None of the CSV files could be loaded.\n\n'
                                + '\n'.join(errors))
            return

        self._set_loaded_files(entries)

        note = f'Loaded {len(entries)} of {len(csv_files)} file(s).'
        if cancelled:
            note += ' (Loading cancelled — only the files above are available.)'
        if errors:
            QMessageBox.warning(self, 'Some files failed to load',
                                note + f'\n\n{len(errors)} failed:\n'
                                + '\n'.join(errors))
        else:
            self.statusBar.showMessage(
                note + ' Browse, set epsilon, then "Process Loaded Files".')

    def _load_entry(self, path):
        """Load one CSV into an entry dict. load_data sets self.global_bounds
        (a fresh dict each call) as a side effect, which we capture per file."""
        data = self.load_data(path)
        return {
            'name': os.path.splitext(os.path.basename(path))[0],
            'path': path,
            'data': data,
            'bounds': self.global_bounds
        }

    def _set_loaded_files(self, entries):
        """Populate the file selector from a list of entries and show the first."""
        self.loaded_files = entries
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItems([e['name'] for e in entries])
        self.file_combo.setCurrentIndex(0)
        self.file_combo.blockSignals(False)
        self.select_file(0)

    def select_file(self, index):
        """Switch the active file to the one at `index` and refresh the view."""
        if index < 0 or index >= len(self.loaded_files):
            return
        entry = self.loaded_files[index]
        self.data = entry['data']
        self.csv_filename = entry['name']
        self.global_bounds = entry['bounds']

        # reflect this file's bounds in the axis spin boxes
        self.xmin_spin.setValue(self.global_bounds['x_min'])
        self.xmax_spin.setValue(self.global_bounds['x_max'])
        self.ymin_spin.setValue(self.global_bounds['y_min'])
        self.ymax_spin.setValue(self.global_bounds['y_max'])

        # reset view state for the new file
        self._zoomed_in = False
        self._custom_limits = None
        self.current_larva = 0

        # repopulate the larva selector without firing its signal mid-update
        self.larva_combo.blockSignals(True)
        self.larva_combo.clear()
        self.larva_combo.addItems([f'Larva {i}' for i in range(len(self.data))])
        self.larva_combo.setCurrentIndex(0)
        self.larva_combo.blockSignals(False)

        self.statusBar.showMessage(
            f'File: {entry["name"]}  ({index + 1}/{len(self.loaded_files)})')
        self.update_plot()

    def setup_ui(self):
        self.setWindowTitle('Trajectory Analyzer')
        self.setGeometry(100, 100, 1200, 800)
    
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_action = file_menu.addAction('Open CSV')
        open_action.triggered.connect(self.open_file)
        load_folder_action = file_menu.addAction('Load Folder...')
        load_folder_action.triggered.connect(self.load_folder)
        process_action = file_menu.addAction('Process Loaded Files...')
        process_action.triggered.connect(self.process_loaded_files)

        self.statusBar = self.statusBar()
        self.statusBar.showMessage('No file loaded')
    
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
    
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)
    
        controls = QHBoxLayout()
    
        # Epsilon slider
        slider_layout = QVBoxLayout()
        self.epsilon_label = QLabel(f'Epsilon: {self.epsilon}')
        self.epsilon_slider = QSlider()
        self.epsilon_slider.setOrientation(1)
        self.epsilon_slider.setMinimum(1)
        self.epsilon_slider.setMaximum(100)
        self.epsilon_slider.setValue(int(self.epsilon * 10))
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        slider_layout.addWidget(self.epsilon_label)
        slider_layout.addWidget(self.epsilon_slider)
        controls.addLayout(slider_layout)

        # Dot size control
        dot_size_layout = QVBoxLayout()
        self.dot_size_label = QLabel(f'Turn Dot Size: {self.dot_size}')
        self.dot_size_spin = QSpinBox()
        self.dot_size_spin.setMinimum(1)
        self.dot_size_spin.setMaximum(30)
        self.dot_size_spin.setValue(self.dot_size)
        self.dot_size_spin.valueChanged.connect(self.update_dot_size)
        dot_size_layout.addWidget(self.dot_size_label)
        dot_size_layout.addWidget(self.dot_size_spin)
        controls.addLayout(dot_size_layout)

        # Trace visibility checkboxes
        self.show_original_cb = QCheckBox('Show Original Trace')
        self.show_original_cb.setChecked(True)
        self.show_original_cb.stateChanged.connect(self.update_plot)
        controls.addWidget(self.show_original_cb)

        self.show_simplified_cb = QCheckBox('Show Simplified Trace')
        self.show_simplified_cb.setChecked(True)
        self.show_simplified_cb.stateChanged.connect(self.update_plot)
        controls.addWidget(self.show_simplified_cb)

        self.show_turn_points_cb = QCheckBox('Show Turn Points')
        self.show_turn_points_cb.setChecked(True)
        self.show_turn_points_cb.stateChanged.connect(self.update_plot)
        controls.addWidget(self.show_turn_points_cb)

        # Axis limit controls
        axis_layout = QVBoxLayout()
        axis_label = QLabel('Set Axis Limits:')
        axis_layout.addWidget(axis_label)
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setDecimals(2)
        self.xmin_spin.setRange(-1e6, 1e6)
        self.xmin_spin.setPrefix('X min: ')
        axis_layout.addWidget(self.xmin_spin)
        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setDecimals(2)
        self.xmax_spin.setRange(-1e6, 1e6)
        self.xmax_spin.setPrefix('X max: ')
        axis_layout.addWidget(self.xmax_spin)
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setDecimals(2)
        self.ymin_spin.setRange(-1e6, 1e6)
        self.ymin_spin.setPrefix('Y min: ')
        axis_layout.addWidget(self.ymin_spin)
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setDecimals(2)
        self.ymax_spin.setRange(-1e6, 1e6)
        self.ymax_spin.setPrefix('Y max: ')
        axis_layout.addWidget(self.ymax_spin)
        set_axis_btn = QPushButton('Set Axis Limits')
        set_axis_btn.clicked.connect(self.set_axis_limits)
        axis_layout.addWidget(set_axis_btn)
        controls.addLayout(axis_layout)

        # Zoom and reset buttons
        zoom_btn = QPushButton('Zoom')
        zoom_btn.setCheckable(True)
        zoom_btn.clicked.connect(self.toggle_zoom)
        controls.addWidget(zoom_btn)

        reset_zoom_btn = QPushButton('Reset Zoom')
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        controls.addWidget(reset_zoom_btn)
    
        # File selector (populated when a file or folder is loaded)
        self.file_combo = QComboBox()
        controls.addWidget(self.file_combo)
        self.file_combo.currentIndexChanged.connect(self.select_file)

        # Larva selector
        self.larva_combo = QComboBox()
        controls.addWidget(self.larva_combo)
        self.larva_combo.currentIndexChanged.connect(self.update_larva)
    
        # Export buttons
        export_layout = QVBoxLayout()
        export_plot_btn = QPushButton('Export Plot as PDF')
        export_plot_btn.clicked.connect(self.export_plot)
        export_stats_btn = QPushButton('Export All Stats')
        export_stats_btn.clicked.connect(self.export_all_stats)
        export_layout.addWidget(export_plot_btn)
        export_layout.addWidget(export_stats_btn)
        process_btn = QPushButton('Process Loaded Files (current ε)')
        process_btn.clicked.connect(self.process_loaded_files)
        export_layout.addWidget(process_btn)
        controls.addLayout(export_layout)
    
        # Stats label
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("QLabel { background-color : white; padding: 10px; }")
        controls.addWidget(self.stats_label)
    
        layout.addLayout(controls)

    def export_plot(self):
        if self._zoom_selector is not None:
            self._zoom_selector.set_active(False)
            self._zoom_selector = None
        for patch in list(self.ax.patches):
            patch.remove()
        for artist in list(self.ax.artists):
            artist.remove()
        default_filename = f'{self.csv_filename}_larva{self.current_larva}_eps{self.epsilon}.pdf'
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Plot', default_filename, 'PDF Files (*.pdf)')
        if filename:
            self.fig.savefig(filename, bbox_inches='tight')
            print(f"Plot saved as {filename}")

    def export_all_stats(self):
        if self.data is None:
            QMessageBox.warning(self, 'No data', 'Please open a CSV file first.')
            return
        default_filename = f'{self.csv_filename}_analysis.xlsx'
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Stats', default_filename, 'Excel Files (*.xlsx)')
        if filename:
            self.write_analysis_workbook(filename)
            self.update_plot()
            print(f"Stats and turning angles exported to {filename}")

    def write_analysis_workbook(self, filename):
        """Write the per-larva stats + turning-angle workbook for the data
        currently held in self.data. Shared by the single-file export and the
        batch processor so both produce identical output. Returns the
        Summary_Stats DataFrame (or None if there is no data)."""
        original_larva = self.current_larva  # save so we can restore at the end
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            all_data = []
            for i in range(len(self.data)):
                self.current_larva = i
                coords = self.data[i]['coords']
                if len(coords) < 2:  # skip empty/trivial larvae
                    print(f"Skipping larva {i}: insufficient data")
                    continue
                stats, simple_traj = self.calculate_stats()
                stats['larva_idx'] = i
                all_data.append(stats)
            
            df_stats = pd.DataFrame(all_data)
            df_stats.to_excel(writer, sheet_name='Summary_Stats', index=False)
            
            for i in range(len(self.data)):
                self.current_larva = i
                coords = self.data[i]['coords']
                if len(coords) < 2:  # skip empty/trivial larvae
                    continue
                simple_traj = rdp(coords, epsilon=self.epsilon)
                turning_angles = self.calculate_turning_angles(simple_traj)
                
                turning_angles_deg = turning_angles * 180 / np.pi
                
                cw_turns = []
                ccw_turns = []
                
                for turn_num, angle in enumerate(turning_angles_deg, 1):
                    if angle < 0:
                        cw_turns.append({
                            'original_turn_number': turn_num,
                            'angle_degrees': angle,
                            'direction': 'CW'
                        })
                    else:
                        ccw_turns.append({
                            'original_turn_number': turn_num,
                            'angle_degrees': angle,
                            'direction': 'CCW'
                        })
                
                df_cw = pd.DataFrame(cw_turns)
                df_ccw = pd.DataFrame(ccw_turns)
                
                if not df_cw.empty:
                    df_cw['direction_turn_number'] = range(1, len(df_cw) + 1)
                if not df_ccw.empty:
                    df_ccw['direction_turn_number'] = range(1, len(df_ccw) + 1)
                
                df_angles = pd.concat([df_cw, df_ccw], ignore_index=True)
                
                # Only reorder columns if there is data (concat of two empty DFs has no columns)
                if not df_angles.empty:
                    df_angles = df_angles[[
                        'direction',
                        'direction_turn_number',
                        'original_turn_number',
                        'angle_degrees'
                    ]]
                
                sheet_name = f'Larva_{i}_Angles'
                df_angles.to_excel(writer, sheet_name=sheet_name, index=False)
            
        self.current_larva = original_larva  # restore original selection
        return df_stats

    def process_loaded_files(self):
        """Export the same stats + turning-angle workbook for every file already
        loaded in memory, using the CURRENT epsilon (the one you settled on while
        browsing). Writes one <name>_analysis.xlsx per file plus a combined
        batch_summary.xlsx."""
        if not self.loaded_files:
            QMessageBox.warning(
                self, 'No files loaded',
                'Load a file (Open CSV) or a folder (Load Folder) first.')
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, 'Select output folder for analysis files')
        if not out_dir:
            return

        # remember what the user was viewing so we can restore it afterwards
        current_index = self.file_combo.currentIndex()

        progress = self._make_progress('Processing files', len(self.loaded_files))
        saved, errors, combined, cancelled = [], [], [], False
        for i, entry in enumerate(self.loaded_files):
            if progress.wasCanceled():
                cancelled = True
                break
            progress.setLabelText(
                f'Processing {entry["name"]}  '
                f'({i + 1}/{len(self.loaded_files)})')
            progress.setValue(i)
            QApplication.processEvents()
            try:
                self.data = entry['data']
                self.csv_filename = entry['name']
                self.global_bounds = entry['bounds']
                self.current_larva = 0
                out_path = os.path.join(out_dir,
                                        f"{entry['name']}_analysis.xlsx")
                df_stats = self.write_analysis_workbook(out_path)
                if df_stats is not None and not df_stats.empty:
                    df_stats = df_stats.copy()
                    df_stats.insert(0, 'source_file', entry['name'])
                    combined.append(df_stats)
                saved.append(entry['name'])
            except Exception as e:
                errors.append(f"{entry['name']}: {e}")

        # Combined summary across all files (one row per larva, tagged by file)
        if combined:
            progress.setLabelText('Writing combined summary...')
            QApplication.processEvents()
            try:
                combined_df = pd.concat(combined, ignore_index=True)
                combined_df.to_excel(
                    os.path.join(out_dir, 'batch_summary.xlsx'), index=False)
            except Exception as e:
                errors.append(f'batch_summary.xlsx: {e}')
        progress.setValue(len(self.loaded_files))   # closes the dialog

        # restore the file the user was viewing
        if 0 <= current_index < len(self.loaded_files):
            self.select_file(current_index)

        msg = (f'Processed {len(saved)} of {len(self.loaded_files)} file(s) '
               f'with epsilon={self.epsilon}.')
        if cancelled:
            msg += ' (Cancelled — remaining files were not processed.)'
        if errors:
            msg += '\n\nProblems:\n' + '\n'.join(errors)
            QMessageBox.warning(self, 'Processing finished with errors', msg)
        else:
            QMessageBox.information(self, 'Processing complete', msg)
        self.statusBar.showMessage(msg.splitlines()[0])
                
    def update_epsilon(self):
        self.epsilon = self.epsilon_slider.value() / 10
        self.epsilon_label.setText(f'Epsilon: {self.epsilon}')
        self.update_plot()

    def update_dot_size(self):
        self.dot_size = self.dot_size_spin.value()
        self.dot_size_label.setText(f'Turn Dot Size: {self.dot_size}')
        self.update_plot()

    def update_larva(self, index):
        self.current_larva = index
        self.update_plot()

    def calculate_stats(self):
        coords = self.data[self.current_larva]['coords']
        distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)) * (1/self.scale)

        time = np.arange(len(coords)) * 0.5  # 2fps sampling
        
        simple_traj = rdp(coords, epsilon=self.epsilon)
        turning_angles = self.calculate_turning_angles(simple_traj)
        
        velocity = self.calculate_velocity(coords[:, 0], coords[:, 1], time)
        heading = self.calculate_heading(velocity)
        speed = np.linalg.norm(velocity, axis=1)
        turn_rate = self.calculate_turn_rate(heading, time[1:])
        
        total_distance = np.sum(distances)
        total_distance_cm = total_distance / 10
        
        stats = {
            'total_distance': total_distance,
            'total_distance_cm': total_distance_cm,
            'avg_step': np.mean(distances) if len(distances) > 0 else 0,
            'max_step': np.max(distances) if len(distances) > 0 else 0,
            'std_step': np.std(distances) if len(distances) > 0 else 0,
            'total_turns': len(simple_traj) - 2,
            'mean_turn_angle': np.mean(np.abs(turning_angles)) * 180 / np.pi if len(turning_angles) > 0 else 0,
            'average_speed': np.mean(speed) if len(speed) > 0 else 0,
            'average_turn_rate': np.mean(np.abs(turn_rate)) if len(turn_rate) > 0 else 0,
            'handedness_index': self.calculate_handedness_index(turning_angles),
            'turns_per_minute': self.calculate_turns_per_minute(simple_traj, len(coords)),
            'ccw_turns': np.sum(turning_angles > 0) if len(turning_angles) > 0 else 0,
            'cw_turns': np.sum(turning_angles < 0) if len(turning_angles) > 0 else 0
        }
        
        return stats, simple_traj

    def calculate_velocity(self, x, y, time):
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(time)
        vx = np.divide(dx, dt, out=np.zeros_like(dx), where=dt!=0)
        vy = np.divide(dy, dt, out=np.zeros_like(dy), where=dt!=0)
        return np.column_stack((vx, vy))

    def calculate_heading(self, velocity):
        speed = np.linalg.norm(velocity, axis=1)
        heading = np.divide(velocity, speed[:, np.newaxis], 
                          out=np.zeros_like(velocity), 
                          where=speed[:, np.newaxis]!=0)
        return heading

    def calculate_turn_rate(self, heading, time):
        if len(heading) < 2:
            return np.array([])
        dot_product = np.sum(heading[:-1] * heading[1:], axis=1)
        dot_product = np.clip(dot_product, -1, 1)
        delta_theta = np.arccos(dot_product)
        dt = np.diff(time)
        turn_rate = np.divide(delta_theta, dt, 
                            out=np.zeros_like(delta_theta), 
                            where=dt!=0)
        return np.abs(turn_rate)

    def calculate_turning_angles(self, trajectory):
        if len(trajectory) < 3:
            return np.array([])
        vectors = np.diff(trajectory, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        turning_angles = np.diff(angles)
        return (turning_angles + np.pi) % (2 * np.pi) - np.pi

    def calculate_handedness_index(self, turning_angles):
        if len(turning_angles) == 0:
            return 0.5
        ccw_turns = np.sum(turning_angles > 0)
        cw_turns = np.sum(turning_angles < 0)
        return ccw_turns / (ccw_turns + cw_turns) if (ccw_turns + cw_turns) > 0 else 0.5

    def calculate_turns_per_minute(self, trajectory, total_frames, fps=2):
        if len(trajectory) == 0 or total_frames == 0:
            return 0
        turn_vector = np.zeros(total_frames)
        turn_frames = np.clip(trajectory[:, 0].astype(int), 0, total_frames - 1)
        turn_vector[turn_frames] = 1
        window_size = 120  # 1 minute at 2fps
        turns_per_window = np.convolve(turn_vector, np.ones(window_size), mode='valid')
        return np.mean(turns_per_window)

    def update_plot(self):
        if self._zoomed_in:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        self.ax.clear()
        coords = self.data[self.current_larva]['coords']
        stats, simple_traj = self.calculate_stats()
        if self.show_original_cb.isChecked():
            self.ax.plot(coords[:, 0], coords[:, 1], 'gray', alpha=0.5, label='Original')
        if self.show_simplified_cb.isChecked():
            self.ax.plot(simple_traj[:, 0], simple_traj[:, 1], 'b-', label='Simplified')
        if self.show_turn_points_cb.isChecked():
            self.ax.plot(simple_traj[:, 0], simple_traj[:, 1], 'ro', label='Turn Points', markersize=self.dot_size)
        if self._custom_limits is not None:
            xmin, xmax, ymin, ymax = self._custom_limits
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
        elif self._zoomed_in:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            self.ax.set_xlim(self.global_bounds['x_min'], self.global_bounds['x_max'])
            self.ax.set_ylim(self.global_bounds['y_min'], self.global_bounds['y_max'])
        self.ax.set_aspect('equal')
        stats_text = (
            f"Movement Statistics:\n"
            f"Total points: {self.data[self.current_larva]['total_points']}\n"
            f"Total distance: {stats['total_distance']:.2f} mm\n"
            f"Avg step: {stats['avg_step']:.4f} mm\n"
            f"Max step: {stats['max_step']:.4f} mm\n"
            f"Step std: {stats['std_step']:.4f}\n"
            f"Average speed: {stats['average_speed']:.4f} mm/s\n"
            f"Average turn rate: {stats['average_turn_rate']:.4f}\n\n"
            f"Turn Analysis:\n"
            f"Total turns: {stats['total_turns']}\n"
            f"Mean turn angle: {stats['mean_turn_angle']:.2f}°\n"
            f"CCW turns: {stats['ccw_turns']}\n"
            f"CW turns: {stats['cw_turns']}\n"
            f"Handedness index: {stats['handedness_index']:.4f}\n"
            f"Turns per minute: {stats['turns_per_minute']:.2f}"
        )
        self.stats_label.setText(stats_text)
        self.ax.set_title(f'Larva {self.current_larva} Trajectory (ε={self.epsilon})')
        self.ax.set_xlabel('X position (mm)')
        self.ax.set_ylabel('Y position (mm)')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def toggle_zoom(self, checked):
        if checked:
            self._zoom_active = True
            if self._zoom_selector is not None:
                self._zoom_selector.set_active(False)
                self._zoom_selector = None
            for patch in list(self.ax.patches):
                patch.remove()
            for artist in list(self.ax.artists):
                artist.remove()
            self._zoom_selector = RectangleSelector(
                self.ax, self.on_select,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords='pixels', interactive=False
            )
            self.canvas.draw_idle()
        else:
            self._zoom_active = False
            if self._zoom_selector is not None:
                self._zoom_selector.set_active(False)
                self._zoom_selector = None
            for patch in list(self.ax.patches):
                patch.remove()
            for artist in list(self.ax.artists):
                artist.remove()
            self.canvas.draw_idle()

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        self.ax.set_xlim(min(x1, x2), max(x1, x2))
        self.ax.set_ylim(min(y1, y2), max(y1, y2))
        self.ax.figure.canvas.draw()
        self._zoomed_in = True

    def reset_zoom(self):
        if self.global_bounds:
            self.ax.set_xlim(self.global_bounds['x_min'], self.global_bounds['x_max'])
            self.ax.set_ylim(self.global_bounds['y_min'], self.global_bounds['y_max'])
            self.ax.figure.canvas.draw()
        if self._zoom_selector is not None:
            self._zoom_selector.set_active(False)
            self._zoom_selector = None
        self._zoom_active = False
        self._zoomed_in = False
        for patch in list(self.ax.patches):
            patch.remove()
        for artist in list(self.ax.artists):
            artist.remove()
        self._custom_limits = None

    def set_axis_limits(self):
        xmin = self.xmin_spin.value()
        xmax = self.xmax_spin.value()
        ymin = self.ymin_spin.value()
        ymax = self.ymax_spin.value()
        self._custom_limits = (xmin, xmax, ymin, ymax)
        self._zoomed_in = False
        self.update_plot()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = TrajectoryViewer()
    viewer.show()
    sys.exit(app.exec_())