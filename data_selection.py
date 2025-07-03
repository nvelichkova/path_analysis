"""
Created on Thu Feb 13 15:54:48 2025

@author: N Velichkova
"""

import pandas as pd
import re
from tkinter import Tk, filedialog
import os

def select_file():
    """
    Open a file dialog to select a data file.
    
    Returns:
    str: Selected file path or None if cancelled
    """
    # Create and hide the tkinter root window
    root = Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title='Select Data File',
        filetypes=[
            ('CSV files', '*.csv'),
            ('Excel files', '*.xlsx *.xls'),
            ('All files', '*.*')
        ],
        initialdir=os.getcwd()  # Start in current working directory
    )
    
    # Destroy the hidden root window
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
    # Get file extension
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
    # Get all column names except the first (parameter name) column
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
        
        # Always include the first column (parameter names)
        filtered_df = df.iloc[:, [0] + [i + 1 for i in selected_indices]]
        
        print("\nSelected columns:")
        for col in selected_columns:
            print(f"- {col}")
        
        return filtered_df, selected_columns
    
    except (ValueError, IndexError) as e:
        print(f"Error in column selection: {e}")
        print("Keeping all columns")
        return df, columns

def analyze_and_select_data(file_path=None):
    """
    Interactive function to analyze data and select specific base names and columns.
    
    Parameters:
    file_path (str, optional): Path to the data file. If None, will prompt for file selection
    
    Returns:
    tuple: (original DataFrame, filtered DataFrame, selected base names, selected columns)
    """
    # If no file path provided, open file dialog
    if file_path is None:
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting...")
            return None, None, None, None
    
    # Load the data
    df = load_data(file_path)
    
    if df is not None:
        # Get unique base names
        base_names = get_unique_base_names(df)
        
        # Print available base names
        print("\nAvailable base names:")
        for i, name in enumerate(base_names, 1):
            count = sum(df.iloc[:, 0].str.startswith(name))
            print(f"{i}. {name} ({count} frames)")
        
        # Get base names selection
        print("\nEnter the numbers of the base names you want to select (comma-separated):")
        print("The data will be ordered in the sequence you specify.")
        print("Example: If you want all mom_x frames first, then all mom_y frames, enter: 13,14")
        selection = input("Selection: ")
        
        try:
            # Convert selection to list of base names
            selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_names = [base_names[i] for i in selected_indices]
            
            # Create filtered and ordered DataFrame
            filtered_df = select_and_order_base_names_data(df, selected_names)
            
            # Print summary of base names selection
            print("\nSelected base names (in order):")
            for name in selected_names:
                count = sum(filtered_df.iloc[:, 0].str.startswith(name))
                print(f"- {name} ({count} frames)")
            
            # Ask if user wants to select specific columns
            print("\nWould you like to select specific columns/larva? (yes/no):")
            if input().lower().startswith('y'):
                filtered_df, selected_columns = select_columns(filtered_df)
            else:
                selected_columns = filtered_df.columns[1:].tolist()
                print("Keeping all columns")
            
            # Print final summary
            print(f"\nFinal DataFrame shape: {filtered_df.shape}")
            
            # Verify ordering
            print("\nVerifying order - first few rows of each selected parameter:")
            for name in selected_names:
                first_rows = filtered_df[filtered_df.iloc[:, 0].str.startswith(name)].head(3)
                print(f"\nFirst 3 rows for {name}:")
                print(first_rows.iloc[:, 0].tolist())
            
            return df, filtered_df, selected_names, selected_columns
            
        except (ValueError, IndexError) as e:
            print(f"Error in selection: {e}")
            return df, None, None, None
    
    return None, None, None, None

# Example usage:
# Example usage:

if __name__ == "__main__":
   file_path = select_file()
   if file_path:
       original_df, filtered_df, selected_names, selected_columns = analyze_and_select_data(file_path)
       
       if filtered_df is not None:
           print("\nFirst few rows of filtered data:")
           print(filtered_df.head())
           
           save = input("\nWould you like to save the filtered data? (yes/no): ")
           if save.lower().startswith('y'):
               root = Tk()
               root.withdraw()
               
               original_file = os.path.basename(file_path)
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
                   print(f"Filtered data saved to: {output_path}")
               else:
                   print("Save cancelled")


