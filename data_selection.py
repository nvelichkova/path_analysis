
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:20:49 2025

@author: bsmsa18b
"""

import pandas as pd
import re
from tkinter import Tk, filedialog
import os

def select_file(multiple=False):
    """
    Open a file dialog to select one or multiple data files.
    
    Parameters:
    multiple (bool): If True, allows selecting multiple files
    
    Returns:
    str or list: Selected file path(s), or None if cancelled
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.update()
    
    if multiple:
        file_paths = filedialog.askopenfilenames(
            title='Select Data Files (hold Ctrl to select multiple)',
            filetypes=[
                ('CSV files', '*.csv'),
                ('Excel files', '*.xlsx *.xls'),
                ('All files', '*.*')
            ],
            initialdir=os.getcwd()
        )
        root.destroy()
        return list(file_paths) if file_paths else None
    else:
        file_path = filedialog.askopenfilename(
            title='Select Data File',
            filetypes=[
                ('CSV files', '*.csv'),
                ('Excel files', '*.xlsx *.xls'),
                ('All files', '*.*')
            ],
            initialdir=os.getcwd()
        )
        root.destroy()
        return file_path if file_path else None

def load_data(file_path):
    """
    Load data from CSV or Excel file based on file extension.
    
    Parameters:
    file_path (str): Path to the data file
    
    Returns:
    pandas.DataFrame: Loaded data
    """
    file_extension = file_path.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        print(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_unique_base_names(df):
    """
    Extract all unique base names from the first column, ignoring numbers in brackets.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    list: Sorted list of unique base names
    """
    first_column_values = df.iloc[:, 0].unique()
    pattern = r'(.+?)\(\d+\)'
    base_names = set()
    
    for value in first_column_values:
        match = re.match(pattern, str(value))
        if match:
            base_name = match.group(1)
            base_names.add(base_name)
    
    return sorted(list(base_names))

def select_and_order_base_names_data(df, selected_base_names):
    """
    Create a new DataFrame containing rows with selected base names,
    ordered by selection order and frame number.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    selected_base_names (list): List of base names in desired order
    
    Returns:
    pandas.DataFrame: New DataFrame with ordered selected base names data
    """
    def extract_frame_number(x):
        match = re.search(r'\((\d+)\)', str(x))
        return int(match.group(1)) if match else -1
    
    ordered_data_frames = []
    
    for base_name in selected_base_names:
        pattern = rf"^{re.escape(base_name)}\(\d+\)$"
        base_data = df[df.iloc[:, 0].str.match(pattern)].copy()
        
        base_data = base_data.sort_values(
            by=df.columns[0],
            key=lambda x: x.map(extract_frame_number)
        )
        
        ordered_data_frames.append(base_data)
    
    if ordered_data_frames:
        final_df = pd.concat(ordered_data_frames, axis=0, ignore_index=True)
        return final_df
    return pd.DataFrame()

def select_columns(df):
    """
    Interactive function to select specific columns (larva) from the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with selected columns
    list: List of selected column names
    """
    columns = df.columns[1:].tolist()
    
    print("\nAvailable columns (larva):")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")
    
    print("\nEnter the numbers of the columns you want to select (comma-separated),")
    print("or press Enter to keep all columns:")
    selection = input("Selection: ")
    
    if not selection.strip():
        print("Keeping all columns")
        return df, columns
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_columns = [columns[i] for i in selected_indices]
        
        filtered_df = df.iloc[:, [0] + [i + 1 for i in selected_indices]]
        
        print("\nSelected columns:")
        for col in selected_columns:
            print(f"- {col}")
        
        return filtered_df, selected_columns
    
    except (ValueError, IndexError) as e:
        print(f"Error in column selection: {e}")
        print("Keeping all columns")
        return df, columns

def save_filtered_data(filtered_df, original_file_path):
    """
    Open a save dialog and write the filtered DataFrame to disk.
    
    Parameters:
    filtered_df (pandas.DataFrame): The filtered data to save
    original_file_path (str): Original file path (used to suggest a save name)
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.update()
    
    original_file = os.path.basename(original_file_path)
    file_name, file_ext = os.path.splitext(original_file)
    new_file = f"{file_name}_formatted{file_ext}"
    
    output_path = filedialog.asksaveasfilename(
        initialfile=new_file,
        defaultextension=file_ext,
        filetypes=[('CSV files', '*.csv'), ('Excel files', '*.xlsx *.xls'), ('All files', '*.*')],
        title='Save Filtered Data As'
    )
    
    root.destroy()
    
    if output_path:
        if file_ext.lower() in ['.xlsx', '.xls']:
            filtered_df.to_excel(output_path, index=False)
        else:
            filtered_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    else:
        print("Save cancelled.")

def process_single_file():
    """Run the full pipeline for a single selected file."""
    file_path = select_file(multiple=False)
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")
    
    df = load_data(file_path)
    if df is None:
        return
    
    base_names = get_unique_base_names(df)
    
    print("\nAvailable base names:")
    for i, name in enumerate(base_names, 1):
        count = sum(df.iloc[:, 0].str.startswith(name))
        print(f"{i}. {name} ({count} frames)")
    
    print("\nEnter the numbers of the base names you want to select (comma-separated):")
    print("The data will be ordered in the sequence you specify.")
    print("Example: If you want all mom_x frames first, then all mom_y frames, enter: 13,14")
    selection = input("Selection: ")
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_names = [base_names[i] for i in selected_indices]
    except (ValueError, IndexError) as e:
        print(f"Error in selection: {e}")
        return
    
    filtered_df = select_and_order_base_names_data(df, selected_names)
    
    print("\nSelected base names (in order):")
    for name in selected_names:
        count = sum(filtered_df.iloc[:, 0].str.startswith(name))
        print(f"- {name} ({count} frames)")
    
    print("\nWould you like to select specific columns/larva? (yes/no):")
    if input().lower().startswith('y'):
        filtered_df, selected_columns = select_columns(filtered_df)
    else:
        selected_columns = filtered_df.columns[1:].tolist()
        print("Keeping all columns")
    
    print(f"\nFinal DataFrame shape: {filtered_df.shape}")
    
    print("\nVerifying order - first few rows of each selected parameter:")
    for name in selected_names:
        first_rows = filtered_df[filtered_df.iloc[:, 0].str.startswith(name)].head(3)
        print(f"\nFirst 3 rows for {name}:")
        print(first_rows.iloc[:, 0].tolist())
    
    print("\nFirst few rows of filtered data:")
    print(filtered_df.head())
    
    save = input("\nWould you like to save the filtered data? (yes/no): ")
    if save.lower().startswith('y'):
        save_filtered_data(filtered_df, file_path)

def process_multiple_files():
    """
    Run the full pipeline for multiple selected files.
    Lets you choose base names/columns independently per file,
    or apply the same selections to all files.
    """
    file_paths = select_file(multiple=True)
    if not file_paths:
        print("No files selected. Exiting...")
        return
    
    print(f"\n{len(file_paths)} file(s) selected:")
    for i, fp in enumerate(file_paths, 1):
        print(f"  {i}. {os.path.basename(fp)}")
    
    print("\nWould you like to apply the same base name / column selections to all files? (yes/no):")
    same_for_all = input().lower().startswith('y')
    
    shared_selected_indices = None
    shared_col_input = None
    
    for file_path in file_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(file_path)}")
        print('='*60)
        
        df = load_data(file_path)
        if df is None:
            print(f"Skipping {os.path.basename(file_path)} due to load error.")
            continue
        
        base_names = get_unique_base_names(df)
        
        # --- Base name selection ---
        if same_for_all and shared_selected_indices is not None:
            try:
                selected_names = [base_names[i] for i in shared_selected_indices]
                print(f"Applying shared base name selection: {selected_names}")
            except IndexError:
                print("Shared selection doesn't match this file's base names. Please re-select.")
                shared_selected_indices = None
        
        if not same_for_all or shared_selected_indices is None:
            print("\nAvailable base names:")
            for i, name in enumerate(base_names, 1):
                count = sum(df.iloc[:, 0].str.startswith(name))
                print(f"{i}. {name} ({count} frames)")
            
            print("\nEnter the numbers of the base names you want to select (comma-separated):")
            print("The data will be ordered in the sequence you specify.")
            selection = input("Selection: ")
            
            try:
                shared_selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_names = [base_names[i] for i in shared_selected_indices]
            except (ValueError, IndexError) as e:
                print(f"Error in selection: {e}. Skipping this file.")
                continue
        
        filtered_df = select_and_order_base_names_data(df, selected_names)
        
        print("\nSelected base names (in order):")
        for name in selected_names:
            count = sum(filtered_df.iloc[:, 0].str.startswith(name))
            print(f"- {name} ({count} frames)")
        
        # --- Column selection ---
        if same_for_all and shared_col_input is not None:
            col_input = shared_col_input
            print(f"Applying shared column selection.")
        else:
            print("\nWould you like to select specific columns/larva? (yes/no):")
            col_input = input().strip().lower()
            if same_for_all:
                shared_col_input = col_input
        
        if col_input.startswith('y'):
            filtered_df, selected_columns = select_columns(filtered_df)
        else:
            selected_columns = filtered_df.columns[1:].tolist()
            print("Keeping all columns")
        
        print(f"\nFinal DataFrame shape: {filtered_df.shape}")
        
        print("\nFirst few rows of filtered data:")
        print(filtered_df.head())
        
        # --- Save ---
        save = input("\nWould you like to save the filtered data? (yes/no): ")
        if save.lower().startswith('y'):
            save_filtered_data(filtered_df, file_path)
    
    print("\nAll files processed.")


if __name__ == "__main__":
    print("Run on single or multiple files?")
    print("1. Single file")
    print("2. Multiple files")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        process_single_file()
    elif choice == '2':
        process_multiple_files()
    else:
        print("Invalid choice. Exiting.")
