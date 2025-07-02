# Path_analysis

This repository contains the code for analyzing and visualizing trajectory data.

## Project Structure
- `analysis.py`: Statistical analysis on variables calculated and exported from trajectory plots.
- `column_filter.py`: Filter columns based on frame number.
- `data_selection.py`: Select specific columns (e.g., mom_x, mom_y) from data.
- `trajectory_plots.py`: Plot trajectories, adjust epsilon, export Excel with data, and export detailed turning angles.
- `notes`: Project notes and workflow.

## Usage
1. Use `data_selection.py` to select relevant columns from your data.
2. Use `column_filter.py` to filter data based on frame numbers.
3. Use `trajectory_plots.py` to visualize and export trajectory data.
4. Use `analysis.py` for statistical analysis on exported data.

## Requirements
- Python 3.6+
- (List any required packages here)

## License
MIT License 