"""
Created on Thu Feb 13 15:54:48 2025

@author: N Velichkova
"""


import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QLabel, QFileDialog, QComboBox, 
                           QTreeView, QTextEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy import stats

class AnalysisViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Advanced Analysis')
        self.setGeometry(100, 100, 1400, 800)
        self.data = {}
        # Initialize with your actual variables, excluding larva_idx
        self.ylabels = {
            'total_distance': 'Total Distance',
            'total_distance_cm': 'Total Distance (cm)',
            'avg_step': 'Average Step Size (mm)',
            'max_step': 'Maximum Step Size (mm)',
            'std_step': 'Step Size Standard Deviation',
            'total_turns': 'Total Number of Turns',
            'mean_turn_angle': 'Mean Turn Angle (degrees)',
            'average_speed': 'Average Speed (mm/s)',
            'average_turn_rate': 'Average Turn Rate',
            'handedness_index': 'Handedness Index',
            'turns_per_minute': 'Turns per Minute',
            'ccw_turns': 'Counter-clockwise Turns',
            'cw_turns': 'Clockwise Turns'
        }
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Data loading
        load_btn = QPushButton('Load Data Directory')
        load_btn.clicked.connect(self.load_data)
        left_layout.addWidget(load_btn)

        # Variable selection
        self.var_combo = QComboBox()
        self.var_combo.addItems(list(self.ylabels.keys()))        
        
        left_layout.addWidget(QLabel('Select Variable:'))
        left_layout.addWidget(self.var_combo)

        # Analysis buttons
        analyze_btn = QPushButton('Run Analysis')
        analyze_btn.clicked.connect(self.run_analysis)
        export_btn = QPushButton('Export Results')
        export_btn.clicked.connect(self.export_results)
        left_layout.addWidget(analyze_btn)
        left_layout.addWidget(export_btn)

        # Status/log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(QLabel('Analysis Log:'))
        left_layout.addWidget(self.log_text)
        
        layout.addWidget(left_panel, stretch=1)

        # Right panel for plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)
        layout.addWidget(right_panel, stretch=2)

    def load_data(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.data = {}
            self.log_text.append("Loading data from: " + directory)
            
            try:
                # Now load all files
                for excel_file in Path(directory).rglob("*.xlsx"):
                    condition = excel_file.parent.name
                    df = pd.read_excel(excel_file)
                    if condition not in self.data:
                        self.data[condition] = []
                    self.data[condition].append(df)
                    self.log_text.append(f"Loaded {excel_file.name}")
                
                # Update combo box with available variables (excluding larva_idx)
                available_vars = [var for var in self.ylabels.keys() 
                                if any(var in df.columns for dfs in self.data.values() 
                                      for df in dfs)]
                self.var_combo.clear()
                self.var_combo.addItems(available_vars)
                
                self.log_text.append(f"\nAvailable variables for analysis:")
                for var in available_vars:
                    self.log_text.append(f"- {var}")
                self.log_text.append(f"\nSuccessfully loaded data from {len(self.data)} conditions")
                
            except Exception as e:
                self.log_text.append(f"Error loading data: {str(e)}")

    def run_analysis(self):
        if not self.data:
            self.log_text.append("No data loaded.")
            return
        
        variable = self.var_combo.currentText()
        self.log_text.append(f"\nAnalyzing {variable}")
        
        try:
            all_data = []
            for condition, dfs in self.data.items():
                for df in dfs:
                    df_copy = df.copy()
                    df_copy['condition'] = condition
                    if variable in df_copy.columns:
                        all_data.append(df_copy[[variable, 'condition']])
        
            if not all_data:
                raise ValueError(f"Variable {variable} not found in data")
                
            combined_df = pd.concat(all_data, ignore_index=True)
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            conditions = sorted(combined_df['condition'].unique())
            plot_data = [combined_df[combined_df['condition'] == cond][variable].values 
                        for cond in conditions]
            
            # Custom color palette - extend for more groups
            colors = ['#2E86C1', '#28B463', '#E74C3C', '#F39C12', '#8E44AD']  # Added more colors
            colors = colors[:len(conditions)]  # Use only as many colors as we have conditions
            
            # Set style
            sns.set_style("whitegrid")
            
            # Create box plot with custom colors
            sns.boxplot(x='condition', y=variable, data=combined_df, ax=ax,
                       palette=colors)
            
            # Add individual points
            sns.swarmplot(x='condition', y=variable, data=combined_df, 
                         color='black', size=4, alpha=0.6, ax=ax)
            
            # Statistical analysis
            normal_distribution = all(stats.shapiro(group)[1] > 0.05 
                                    for group in plot_data if len(group) > 3)
            
            # Perform statistical tests based on number of groups
            if len(plot_data) == 2:
                if normal_distribution:
                    t, p = stats.ttest_ind(plot_data[0], plot_data[1])
                    test_name = "T-test"
                else:
                    _, p = stats.mannwhitneyu(plot_data[0], plot_data[1])
                    test_name = "Mann-Whitney U"
                
                stats_text = f"{test_name}\np={p:.4f}"
                if p < 0.05:
                    stats_text = f"{self.get_significance_symbol(p)}\n{stats_text}"
                plt.figtext(0.1, 0.95, stats_text, fontsize=10)
                
                # Add significance bar for two groups
                if p < 0.05:
                    y_max = ax.get_ylim()[1]
                    y_inc = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                    ax.plot([0, 1], [y_max + y_inc, y_max + y_inc], 'k-')
                    ax.text(0.5, y_max + y_inc*1.2, self.get_significance_symbol(p), 
                           ha='center', va='bottom')
                    ax.set_ylim(ax.get_ylim()[0], y_max + y_inc*3)
            
            elif len(plot_data) > 2:
                # For more than two groups
                if normal_distribution:
                    f, p = stats.f_oneway(*plot_data)
                    test_name = "One-way ANOVA"
                else:
                    h, p = stats.kruskal(*plot_data)
                    test_name = "Kruskal-Wallis"
                
                stats_text = f"{test_name}\np={p:.4f}"
                if p < 0.05:
                    stats_text += "\n\nPost-hoc tests:"
                    
                    # Perform post-hoc tests
                    for i in range(len(conditions)):
                        for j in range(i+1, len(conditions)):
                            if normal_distribution:
                                _, p_posthoc = stats.ttest_ind(plot_data[i], plot_data[j])
                                test_type = "t-test"
                            else:
                                _, p_posthoc = stats.mannwhitneyu(plot_data[i], plot_data[j])
                                test_type = "Mann-Whitney U"
                            
                            if p_posthoc < 0.05:
                                stats_text += f"\n{conditions[i]} vs {conditions[j]}: "
                                stats_text += f"p={p_posthoc:.4f} {self.get_significance_symbol(p_posthoc)}"
                
                # Add text box with statistical results
                plt.figtext(0.1, 0.95, stats_text, fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title(self.ylabels[variable], fontsize=12, pad=20)
            plt.xlabel('Condition', fontsize=10)
            plt.ylabel(self.ylabels[variable], fontsize=10)
            
            # Customize the appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=9)
            
            plt.tight_layout()
            self.canvas.draw()
            
            # Add summary statistics to log
            self.log_text.append("\nSummary Statistics:")
            summary = combined_df.groupby('condition')[variable].describe()
            self.log_text.append(str(summary))
            
            # Add effect size calculation
            if len(plot_data) == 2:
                effect_size = abs(np.mean(plot_data[0]) - np.mean(plot_data[1])) / np.sqrt((np.var(plot_data[0]) + np.var(plot_data[1])) / 2)
                self.log_text.append(f"\nCohen's d effect size: {effect_size:.3f}")
            
        except Exception as e:
            self.log_text.append(f"Error: {str(e)}")

    def export_results(self):
        if not self.data:
            self.log_text.append("No data loaded.")
            return
            
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Export Results', 'analysis_results.pdf', 'PDF Files (*.pdf)')
            if filename:
                variables = [var for var in self.ylabels.keys() 
                            if any(var in df.columns for dfs in self.data.values() for df in dfs)]
                
                with PdfPages(filename) as pdf:
                    for var in variables:
                        fig = plt.figure(figsize=(10, 6))
                        
                        all_data = []
                        for condition, dfs in self.data.items():
                            for df in dfs:
                                if var in df.columns:
                                    df_copy = df.copy()
                                    df_copy['condition'] = condition
                                    all_data.append(df_copy[[var, 'condition']])
                        
                        if not all_data:
                            continue
                            
                        combined_df = pd.concat(all_data, ignore_index=True)
                        
                        conditions = sorted(combined_df['condition'].unique())
                        plot_data = [combined_df[combined_df['condition'] == cond][var].values 
                                   for cond in conditions]
                        
                        # Custom color palette
                        colors = ['#2E86C1', '#28B463', '#E74C3C', '#F39C12', '#8E44AD']  # Extended color palette
                        colors = colors[:len(conditions)]  # Use only as many colors as needed
                        
                        # Set style
                        sns.set_style("whitegrid")
                        
                        normal_distribution = all(stats.shapiro(group)[1] > 0.05 
                                               for group in plot_data if len(group) > 3)
                        
                        # Statistical analysis based on number of groups
                        if len(plot_data) == 2:
                            if normal_distribution:
                                t, p = stats.ttest_ind(plot_data[0], plot_data[1])
                                test_name = "T-test"
                            else:
                                _, p = stats.mannwhitneyu(plot_data[0], plot_data[1])
                                test_name = "Mann-Whitney U"
                            
                            stats_text = f"{test_name}\np={p:.4f}"
                            if p < 0.05:
                                stats_text = f"{self.get_significance_symbol(p)}\n{stats_text}"
                            plt.figtext(0.1, 0.88, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
                        
                        elif len(plot_data) > 2:
                            if normal_distribution:
                                f, p = stats.f_oneway(*plot_data)
                                test_name = "One-way ANOVA"
                            else:
                                h, p = stats.kruskal(*plot_data)
                                test_name = "Kruskal-Wallis"
                            
                            stats_text = f"{test_name}\np={p:.4f}"
                            if p < 0.05:
                                stats_text += "\n\nPost-hoc tests:"
                                for i in range(len(conditions)):
                                    for j in range(i+1, len(conditions)):
                                        if normal_distribution:
                                            _, p_posthoc = stats.ttest_ind(plot_data[i], plot_data[j])
                                        else:
                                            _, p_posthoc = stats.mannwhitneyu(plot_data[i], plot_data[j])
                                        if p_posthoc < 0.05:
                                            stats_text += f"\n{conditions[i]} vs {conditions[j]}: "
                                            stats_text += f"p={p_posthoc:.4f} {self.get_significance_symbol(p_posthoc)}"
                            
                            plt.figtext(0.1, 0.88, stats_text, fontsize=10,
                                      bbox=dict(facecolor='white', alpha=0.8, pad=5))
                        
                        ax = fig.add_subplot(111)
                        sns.boxplot(x='condition', y=var, data=combined_df, ax=ax,
                                  palette=colors)
                        sns.swarmplot(x='condition', y=var, data=combined_df, 
                                    color='black', size=4, alpha=0.6, ax=ax)
                        
                        if len(plot_data) == 2 and p < 0.05:
                            y_max = ax.get_ylim()[1]
                            y_inc = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                            ax.plot([0, 1], [y_max + y_inc, y_max + y_inc], 'k-')
                            ax.text(0.5, y_max + y_inc*1.2, self.get_significance_symbol(p), 
                                   ha='center', va='bottom')
                            ax.set_ylim(ax.get_ylim()[0], y_max + y_inc*3)
                        
                        plt.title(self.ylabels[var], fontsize=12, pad=20)
                        plt.xlabel('Condition', fontsize=10)
                        plt.ylabel(self.ylabels[var], fontsize=10)
                        
                        # Customize the appearance
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.tick_params(labelsize=9)
                        
                        # Adjust the layout to leave space for statistics
                        plt.subplots_adjust(top=0.85)  # Increase top margin for stats text
                        
                        pdf.savefig(fig)
                        plt.close()
                
                self.log_text.append(f"Exported to {filename}")
                
        except Exception as e:
            self.log_text.append(f"Error: {str(e)}")
   
    def add_stat_significance(self, ax, data, labels):
        y_max = ax.get_ylim()[1]
        y_inc = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
        
        # Check normality
        normal_distribution = all(stats.shapiro(group)[1] > 0.05 
                                for group in data if len(group) > 3)
        
        if len(data) == 2:
            if normal_distribution:
                t, p = stats.ttest_ind(data[0], data[1])
                test_name = "T-test"
            else:
                _, p = stats.mannwhitneyu(data[0], data[1])
                test_name = "Mann-Whitney U"
            
            self.log_text.append(f"\n{test_name}: p={p:.4f}")
            
        elif len(data) > 2:
            if normal_distribution:
                f, p = stats.f_oneway(*data)
                test_name = "ANOVA"
            else:
                h, p = stats.kruskal(*data)
                test_name = "Kruskal-Wallis"
            
            self.log_text.append(f"\n{test_name}: p={p:.4f}")
            
            if p < 0.05:
                self.log_text.append("\nPairwise comparisons:")
                for i in range(len(data)):
                    for j in range(i+1, len(data)):
                        if normal_distribution:
                            _, p_pair = stats.ttest_ind(data[i], data[j])
                        else:
                            _, p_pair = stats.mannwhitneyu(data[i], data[j])
                        self.log_text.append(f"{labels[i]} vs {labels[j]}: p={p_pair:.4f}")

    def get_significance_symbol(self, p):
        if p < 0.0001:
            return '****'
        elif p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = AnalysisViewer()
    viewer.show()
    sys.exit(app.exec_())
    
    
    
    
    
