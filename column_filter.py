import sys
import os
import traceback
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, 
                             QTableView, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class FileLoaderThread(QThread):
    """Thread to load one or more files without freezing the UI"""
    file_loaded = pyqtSignal(str, object)   # path, dataframe
    error_occurred = pyqtSignal(str, str)   # path, error message
    all_finished = pyqtSignal()

    def __init__(self, file_paths):
        super().__init__()
        # Accept either a single path (str) or a list of paths
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = file_paths

    def run(self):
        for file_path in self.file_paths:
            try:
                # Determine file type and use appropriate reading method
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == '.csv':
                    df = pd.read_csv(file_path, low_memory=False)
                elif file_ext in ['.xls', '.xlsx']:
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")

                self.file_loaded.emit(file_path, df)
            except Exception as e:
                error_msg = (f"Error loading {os.path.basename(file_path)}: "
                             f"{str(e)}\n{traceback.format_exc()}")
                self.error_occurred.emit(file_path, error_msg)
        self.all_finished.emit()

class ColumnFilterApp(QWidget):
    def __init__(self):
        super().__init__()
        # loaded_files holds one dict per file:
        #   {'path': str, 'original_df': DataFrame, 'filtered_df': DataFrame or None}
        self.loaded_files = []
        # references to the currently displayed file (kept for the table/display logic)
        self.original_df = None
        self.filtered_df = None
        self.load_errors = []
        self.initUI()
    
    def initUI(self):
        # Main layout
        layout = QVBoxLayout()
        
        # File selection section
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        select_file_btn = QPushButton("Select File(s)")
        select_file_btn.clicked.connect(self.select_files)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(select_file_btn)
        layout.addLayout(file_layout)
        
        # Minimum value section
        min_val_layout = QHBoxLayout()
        self.min_val_label = QLabel("Minimum non-null values:")
        self.min_val_input = QLineEdit()
        self.min_val_input.setPlaceholderText("Enter minimum number of non-null values")
        min_val_layout.addWidget(self.min_val_label)
        min_val_layout.addWidget(self.min_val_input)
        layout.addLayout(min_val_layout)
        
        # Filter and Save buttons
        btn_layout = QHBoxLayout()
        filter_btn = QPushButton("Filter Columns")
        filter_btn.clicked.connect(self.filter_columns)
        save_btn = QPushButton("Save Filtered File(s)")
        save_btn.clicked.connect(self.save_filtered_file)
        btn_layout.addWidget(filter_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)
        
        # Table view to show data (shows the first loaded file as a preview)
        self.table_view = QTableView()
        layout.addWidget(self.table_view)
        
        # Error log area
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setVisible(False)
        layout.addWidget(self.error_log)
        
        self.setLayout(layout)
        self.setWindowTitle('Column Filter')
        self.resize(1000, 700)
    
    def select_files(self):
        # Open file dialog to select one or more CSV / Excel files
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select CSV or Excel File(s)", 
            "", 
            "All Supported Files (*.csv *.xls *.xlsx);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx)"
        )
        
        if file_paths:
            # Reset state for a fresh batch
            self.loaded_files = []
            self.original_df = None
            self.filtered_df = None
            self.load_errors = []
            self.error_log.setVisible(False)

            if len(file_paths) == 1:
                self.file_path_label.setText(file_paths[0])
            else:
                self.file_path_label.setText(f"Loading {len(file_paths)} files...")

            # Use thread to load file(s)
            self.loader_thread = FileLoaderThread(file_paths)
            self.loader_thread.file_loaded.connect(self.on_file_loaded)
            self.loader_thread.error_occurred.connect(self.on_file_load_error)
            self.loader_thread.all_finished.connect(self.on_all_finished)
            self.loader_thread.start()
    
    def on_file_loaded(self, file_path, df):
        self.loaded_files.append({
            'path': file_path,
            'original_df': df,
            'filtered_df': None
        })
    
    def on_file_load_error(self, file_path, error_msg):
        self.load_errors.append(error_msg)
    
    def on_all_finished(self):
        # Surface any load errors
        if self.load_errors:
            combined = "\n\n".join(self.load_errors)
            QMessageBox.critical(self, "File Loading Error", combined)
            self.error_log.setText(combined)
            self.error_log.setVisible(True)

        if not self.loaded_files:
            self.file_path_label.setText("No file loaded")
            return

        # Display the first loaded file as a preview
        first = self.loaded_files[0]
        self.original_df = first['original_df']
        self.filtered_df = None
        self.display_dataframe(self.original_df)

        if len(self.loaded_files) == 1:
            self.file_path_label.setText(first['path'])
            df = first['original_df']
            info_msg = (f"File loaded successfully\n"
                        f"Total rows: {len(df)}\n"
                        f"Total columns: {len(df.columns)}")
        else:
            self.file_path_label.setText(
                f"{len(self.loaded_files)} files loaded "
                f"(previewing: {os.path.basename(first['path'])})"
            )
            lines = [f"{len(self.loaded_files)} files loaded successfully:"]
            for entry in self.loaded_files:
                d = entry['original_df']
                lines.append(f"  - {os.path.basename(entry['path'])}: "
                             f"{len(d)} rows, {len(d.columns)} columns")
            info_msg = "\n".join(lines)

        QMessageBox.information(self, "Files Loaded", info_msg)
    
    def filter_columns(self):
        # Validate input
        if not self.loaded_files:
            QMessageBox.warning(self, "Warning", "Please select a file first.")
            return
        
        try:
            min_val = int(self.min_val_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number.")
            return
        
        # Apply the same threshold to every loaded file (each filtered on its own columns)
        summary_lines = []
        for entry in self.loaded_files:
            df = entry['original_df']
            non_null_counts = df.count()
            kept_cols = non_null_counts[non_null_counts >= min_val].index
            entry['filtered_df'] = df[kept_cols]
            summary_lines.append(
                f"  - {os.path.basename(entry['path'])}: "
                f"{len(df.columns)} -> {len(kept_cols)} columns"
            )
        
        # Display the first file's filtered result as a preview
        first = self.loaded_files[0]
        self.filtered_df = first['filtered_df']
        self.display_dataframe(self.filtered_df)
        
        # Show info about filtering
        if len(self.loaded_files) == 1:
            msg = (f"Reduced from {len(first['original_df'].columns)} to "
                   f"{len(first['filtered_df'].columns)} columns")
        else:
            msg = "Filtering complete for all files:\n" + "\n".join(summary_lines)
        QMessageBox.information(self, "Filtering Complete", msg)
    
    def save_filtered_file(self):
        # Check we have at least one filtered result
        filtered_entries = [e for e in self.loaded_files if e['filtered_df'] is not None]
        if not filtered_entries:
            QMessageBox.warning(self, "Warning", "No filtered data to save.")
            return
        
        if len(filtered_entries) == 1:
            self._save_single(filtered_entries[0])
        else:
            self._save_batch(filtered_entries)
    
    def _default_save_name(self, source_path, ext='.csv'):
        """Build a default output name based on the source file name."""
        base = os.path.splitext(os.path.basename(source_path))[0]
        directory = os.path.dirname(source_path)
        return os.path.join(directory, f"{base}_filtered{ext}")
    
    def _save_single(self, entry):
        # Pre-fill the dialog with a name derived from the opened file
        default_path = self._default_save_name(entry['path'], '.csv')
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Filtered File", 
            default_path, 
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if save_path:
            try:
                if save_path.lower().endswith('.csv'):
                    entry['filtered_df'].to_csv(save_path, index=False)
                else:
                    entry['filtered_df'].to_excel(save_path, index=False)
                QMessageBox.information(self, "Success", f"File saved to {save_path}")
            except Exception as e:
                self._report_save_error(e)
    
    def _save_batch(self, entries):
        # For multiple files, ask for an output folder and derive each name
        directory = QFileDialog.getExistingDirectory(
            self, "Select folder to save filtered files"
        )
        if not directory:
            return
        
        saved, errors = [], []
        for entry in entries:
            base = os.path.splitext(os.path.basename(entry['path']))[0]
            src_ext = os.path.splitext(entry['path'])[1].lower()
            try:
                if src_ext == '.csv':
                    out_path = os.path.join(directory, f"{base}_filtered.csv")
                    entry['filtered_df'].to_csv(out_path, index=False)
                else:
                    # Excel inputs (.xls/.xlsx) are written back as .xlsx
                    out_path = os.path.join(directory, f"{base}_filtered.xlsx")
                    entry['filtered_df'].to_excel(out_path, index=False)
                saved.append(os.path.basename(out_path))
            except Exception as e:
                errors.append(f"{os.path.basename(entry['path'])}: {str(e)}")
        
        msg = f"Saved {len(saved)} file(s) to:\n{directory}"
        if saved:
            msg += "\n\n" + "\n".join(f"  - {n}" for n in saved)
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(f"  - {e}" for e in errors)
            QMessageBox.warning(self, "Saved with errors", msg)
        else:
            QMessageBox.information(self, "Success", msg)
    
    def _report_save_error(self, exc):
        error_msg = f"Could not save the file: {str(exc)}\n{traceback.format_exc()}"
        QMessageBox.critical(self, "Error", error_msg)
        self.error_log.setText(error_msg)
        self.error_log.setVisible(True)
    
    def display_dataframe(self, df):
        # Limit display to first 1000 rows to prevent UI freezing
        df_display = df.head(1000)
        
        # Convert DataFrame to QStandardItemModel for display
        model = QStandardItemModel(df_display.shape[0], df_display.shape[1])
        
        # Set column headers
        model.setHorizontalHeaderLabels([str(c) for c in df_display.columns])
        
        # Populate the model with data
        for row in range(df_display.shape[0]):
            for col in range(df_display.shape[1]):
                # Convert to string to handle various data types
                value = str(df_display.iloc[row, col]) if pd.notna(df_display.iloc[row, col]) else ''
                item = QStandardItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                model.setItem(row, col, item)
        
        self.table_view.setModel(model)

def main():
    app = QApplication(sys.argv)
    ex = ColumnFilterApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

# Installation instructions:
# 1. Make sure you have the following libraries installed:
#    pip install pandas PyQt5 openpyxl
# 2. Save this script and run it from Spyder or command line
# 3. Use the GUI to select one or more files, set minimum non-null values,
#    filter, then save. Single file -> save dialog with a derived name;
#    multiple files -> pick a folder and each is saved as <name>_filtered.<ext>
