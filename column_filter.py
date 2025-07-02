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
    """Thread to load large files without freezing the UI"""
    file_loaded = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # Determine file type and use appropriate reading method
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.csv':
                # For CSV, use chunksize to handle very large files
                df = pd.read_csv(self.file_path, low_memory=False)
            elif file_ext in ['.xls', '.xlsx']:
                # For Excel, try to read with specific parameters
                df = pd.read_excel(self.file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            self.file_loaded.emit(df)
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

class ColumnFilterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.original_df = None
        self.filtered_df = None
        self.initUI()
    
    def initUI(self):
        # Main layout
        layout = QVBoxLayout()
        
        # File selection section
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        select_file_btn = QPushButton("Select File")
        select_file_btn.clicked.connect(self.select_file)
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
        save_btn = QPushButton("Save Filtered File")
        save_btn.clicked.connect(self.save_filtered_file)
        btn_layout.addWidget(filter_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)
        
        # Table view to show data
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
    
    def select_file(self):
        # Open file dialog to select CSV or Excel file
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select CSV or Excel File", 
            "", 
            "All Supported Files (*.csv *.xls *.xlsx);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx)"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            
            # Use thread to load file
            self.loader_thread = FileLoaderThread(file_path)
            self.loader_thread.file_loaded.connect(self.on_file_loaded)
            self.loader_thread.error_occurred.connect(self.on_file_load_error)
            self.loader_thread.start()
    
    def on_file_loaded(self, df):
        self.original_df = df
        self.display_dataframe(self.original_df)
        
        # Show basic file info
        info_msg = (f"File loaded successfully\n"
                    f"Total rows: {len(df)}\n"
                    f"Total columns: {len(df.columns)}")
        QMessageBox.information(self, "File Loaded", info_msg)
    
    def on_file_load_error(self, error_msg):
        # Show error in a message box
        QMessageBox.critical(self, "File Loading Error", error_msg)
        
        # Also log detailed error
        self.error_log.setText(error_msg)
        self.error_log.setVisible(True)
    
    def filter_columns(self):
        # Validate input
        if self.original_df is None:
            QMessageBox.warning(self, "Warning", "Please select a file first.")
            return
        
        try:
            min_val = int(self.min_val_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number.")
            return
        
        # Filter columns with non-null values
        non_null_counts = self.original_df.count()
        self.filtered_df = self.original_df[non_null_counts[non_null_counts >= min_val].index]
        
        # Display filtered data
        self.display_dataframe(self.filtered_df)
        
        # Show info about filtering
        QMessageBox.information(
            self, 
            "Filtering Complete", 
            f"Reduced from {len(self.original_df.columns)} to {len(self.filtered_df.columns)} columns"
        )
    
    def save_filtered_file(self):
        if self.filtered_df is None:
            QMessageBox.warning(self, "Warning", "No filtered data to save.")
            return
        
        # Open save file dialog
        save_path, file_type = QFileDialog.getSaveFileName(
            self, 
            "Save Filtered File", 
            "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if save_path:
            try:
                if save_path.endswith('.csv'):
                    self.filtered_df.to_csv(save_path, index=False)
                else:
                    self.filtered_df.to_excel(save_path, index=False)
                
                QMessageBox.information(self, "Success", f"File saved to {save_path}")
            except Exception as e:
                error_msg = f"Could not save the file: {str(e)}\n{traceback.format_exc()}"
                QMessageBox.critical(self, "Error", error_msg)
                
                # Log detailed error
                self.error_log.setText(error_msg)
                self.error_log.setVisible(True)
    
    def display_dataframe(self, df):
        # Limit display to first 1000 rows to prevent UI freezing
        df_display = df.head(1000)
        
        # Convert DataFrame to QStandardItemModel for display
        model = QStandardItemModel(df_display.shape[0], df_display.shape[1])
        
        # Set column headers
        model.setHorizontalHeaderLabels(df_display.columns)
        
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
# 3. Use the GUI to select your file, set minimum non-null values, and filter columns