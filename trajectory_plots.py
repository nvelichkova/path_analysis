# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:18:13 2025

@author: bsmsa18b
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QSlider, QPushButton, QLabel, QComboBox, QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox)
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
        self.global_bounds = None  # Add global bounds attribute
        self.dot_size = 4  # Default dot size for turn points
        self._zoom_selector = None
        self._zoom_active = False
        self._zoomed_in = False  # Track if user is zoomed in
        self._custom_limits = None  # Store custom axis limits if set
        self.setup_ui()
        self.csv_filename = None

    def load_data(self, csv_file):
        print("Reading CSV file...")
        data_aux = pd.read_csv(csv_file, dtype={0: str})
        data_aux.iloc[:, 1:] = data_aux.iloc[:, 1:].astype(float)
        
        processed_data = []
        n_larvae = data_aux.shape[1] - 1
        
        # Initialize global min/max values
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
            
            # Update global bounds
            global_min_x = min(global_min_x, np.min(x_clean))
            global_max_x = max(global_max_x, np.max(x_clean))
            global_min_y = min(global_min_y, np.min(y_clean))
            global_max_y = max(global_max_y, np.max(y_clean))
            
            distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            
            processed_data.append({
                'coords': coords,
                'distances': distances,
                'total_points': len(x_clean),
                'total_distance': np.sum(distances)
            })
        
        # Find the absolute min and max across both dimensions
        overall_min = min(global_min_x, global_min_y)
        overall_max = max(global_max_x, global_max_y)
        
        # Add padding (10%)
        total_range = overall_max - overall_min
        padding = total_range * 0.1
        
        # Use the same min and max for both axes
        self.global_bounds = {
            'x_min': overall_min - padding,
            'x_max': overall_max + padding,
            'y_min': overall_min - padding,
            'y_max': overall_max + padding
        }
        
        # Set axis spin boxes to global bounds
        self.xmin_spin.setValue(self.global_bounds['x_min'])
        self.xmax_spin.setValue(self.global_bounds['x_max'])
        self.ymin_spin.setValue(self.global_bounds['y_min'])
        self.ymax_spin.setValue(self.global_bounds['y_max'])
        
        return processed_data

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open CSV', '', 'CSV Files (*.csv)')
        if filename:
            self.csv_filename = os.path.splitext(os.path.basename(filename))[0]
            self.data = self.load_data(filename)
            self.current_larva = 0
            self.larva_combo.clear()
            self.larva_combo.addItems([f'Larva {i}' for i in range(len(self.data))])
            self.statusBar.showMessage(f'Current file: {filename}')
            self.update_plot()

    def setup_ui(self):
        self.setWindowTitle('Trajectory Analyzer')
        self.setGeometry(100, 100, 1200, 800)
    
        # Add menu bar first
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_action = file_menu.addAction('Open CSV')
        open_action.triggered.connect(self.open_file)

        # Add status bar after menu bar
        self.statusBar = self.statusBar()
        self.statusBar.showMessage('No file loaded')
    
        # Main widget setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
    
        # Figure setup
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)
    
        # Controls layout
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
        controls.addLayout(export_layout)
    
        # Stats label
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("QLabel { background-color : white; padding: 10px; }")
        controls.addWidget(self.stats_label)
    
        layout.addLayout(controls)

    def export_plot(self):
        # Deactivate and remove RectangleSelector before saving
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
        default_filename = f'{self.csv_filename}_analysis.xlsx'
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Stats', default_filename, 'Excel Files (*.xlsx)')
        if filename:
            # Create a Pandas Excel writer using openpyxl as the engine
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Export main statistics
                all_data = []
                for i in range(len(self.data)):
                    self.current_larva = i
                    stats, simple_traj = self.calculate_stats()
                    stats['larva_idx'] = i
                    all_data.append(stats)
                
                # Write main stats to the first sheet
                df_stats = pd.DataFrame(all_data)
                df_stats.to_excel(writer, sheet_name='Summary_Stats', index=False)
                
                # Export turning angles for each larva
                for i in range(len(self.data)):
                    self.current_larva = i
                    coords = self.data[i]['coords']
                    simple_traj = rdp(coords, epsilon=self.epsilon)
                    turning_angles = self.calculate_turning_angles(simple_traj)
                    
                    # Convert angles from radians to degrees
                    turning_angles_deg = turning_angles * 180 / np.pi
                    
                    # Create lists for CW and CCW turns
                    cw_turns = []
                    ccw_turns = []
                    
                    # Separate turns by direction while maintaining original order
                    for turn_num, angle in enumerate(turning_angles_deg, 1):
                        if angle < 0:  # CW turn
                            cw_turns.append({
                                'original_turn_number': turn_num,
                                'angle_degrees': angle,
                                'direction': 'CW'
                            })
                        else:  # CCW turn
                            ccw_turns.append({
                                'original_turn_number': turn_num,
                                'angle_degrees': angle,
                                'direction': 'CCW'
                            })
                    
                    # Create separate DataFrames for CW and CCW turns
                    df_cw = pd.DataFrame(cw_turns)
                    df_ccw = pd.DataFrame(ccw_turns)
                    
                    # Add sequential numbering within each direction
                    if not df_cw.empty:
                        df_cw['direction_turn_number'] = range(1, len(df_cw) + 1)
                    if not df_ccw.empty:
                        df_ccw['direction_turn_number'] = range(1, len(df_ccw) + 1)
                    
                    # Combine the DataFrames with CW first, then CCW
                    df_angles = pd.concat([df_cw, df_ccw], ignore_index=True)
                    
                    # Reorder columns for better readability
                    df_angles = df_angles[[
                        'direction',
                        'direction_turn_number',
                        'original_turn_number',
                        'angle_degrees'
                    ]]
                    
                    # Write to a separate sheet for each larva
                    sheet_name = f'Larva_{i}_Angles'
                    df_angles.to_excel(writer, sheet_name=sheet_name, index=False)
                
            print(f"Stats and turning angles exported to {filename}")
                
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
       #distances = self.data[self.current_larva]['distances']
       distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)) * (1/self.scale)  # distances in mm


       time = np.arange(len(coords)) * 0.5  # 2fps sampling
       
       # Basic RDP stats
       simple_traj = rdp(coords, epsilon=self.epsilon)
       turning_angles = self.calculate_turning_angles(simple_traj)
       
       # Calculate velocity and heading
       velocity = self.calculate_velocity(coords[:, 0], coords[:, 1], time)
       heading = self.calculate_heading(velocity)
       speed = np.linalg.norm(velocity, axis=1)
       turn_rate = self.calculate_turn_rate(heading, time[1:])
       
       # Calculate total distance in different units
       total_distance = np.sum(distances)  # in mm
       total_distance_cm = total_distance / 10  # Convert to cm
       
       stats = {
           'total_distance': total_distance,
           'total_distance_cm': total_distance_cm,
           'avg_step': np.mean(distances),
           'max_step': np.max(distances),
           'std_step': np.std(distances),
           'total_turns': len(simple_traj) - 2,
           'mean_turn_angle': np.mean(np.abs(turning_angles)) * 180 / np.pi if len(turning_angles) > 0 else 0,
           'average_speed': np.mean(speed),
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
        # Save current axis limits if zoomed in
        if self._zoomed_in:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        self.ax.clear()
        coords = self.data[self.current_larva]['coords']
        stats, simple_traj = self.calculate_stats()
        # Plot trajectories based on checkboxes
        if self.show_original_cb.isChecked():
            self.ax.plot(coords[:, 0], coords[:, 1], 'gray', alpha=0.5, label='Original')
        if self.show_simplified_cb.isChecked():
            self.ax.plot(simple_traj[:, 0], simple_traj[:, 1], 'b-', label='Simplified')
        if self.show_turn_points_cb.isChecked():
            self.ax.plot(simple_traj[:, 0], simple_traj[:, 1], 'ro', label='Turn Points', markersize=self.dot_size)
        # Set axis limits: custom > zoom > global
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
        # Update stats text
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
            # Remove previous selector if it exists
            if self._zoom_selector is not None:
                self._zoom_selector.set_active(False)
                self._zoom_selector = None
            # Clear any previous rectangle
            for patch in list(self.ax.patches):
                patch.remove()
            # Remove any leftover artists (handles, etc.)
            for artist in list(self.ax.artists):
                artist.remove()
            self._zoom_selector = RectangleSelector(
                self.ax, self.on_select,
                useblit=True,
                button=[1],  # left mouse button
                minspanx=5, minspany=5,
                spancoords='pixels', interactive=False
            )
            self.canvas.draw_idle()
        else:
            self._zoom_active = False
            if self._zoom_selector is not None:
                self._zoom_selector.set_active(False)
                self._zoom_selector = None
            # Also clear any leftover rectangle/handles when zoom is turned off
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
        # Deactivate zoom if active
        if self._zoom_selector is not None:
            self._zoom_selector.set_active(False)
            self._zoom_selector = None
        self._zoom_active = False
        self._zoomed_in = False
        # Also clear any leftover rectangle/handles when resetting zoom
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
        self._zoomed_in = False  # Custom limits override zoom
        self.update_plot()

if __name__ == '__main__':
   app = QApplication(sys.argv)
   viewer = TrajectoryViewer()
   viewer.show()
   sys.exit(app.exec_())