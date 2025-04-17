import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

# Set style
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Arial"

# Create data for absolute errors - 13B-5T model
abs_data_13b = {
    'Task Name': [
        'MMLU', 'HellaSwag', 'ARC-Challenge', 'ARC-Easy',
        'PIQA', 'CommonsenseQA', 'SocialIQa', 'OpenBookQA'
    ],
    'Examples': [14042, 10042, 1172, 2376, 1838, 1221, 1954, 500],
    'Task Loss': [0.3, 2.1, 11.1, 9.9, 0.9, 3.5, 1.6, 3.8],
    'C4 Loss': [2.6, 4.7, 2.5, 1.7, 1.2, 5.2, 4.7, 5.1],
    'TaskCE Loss': [10.4, 8.7, 12.3, 5.4, 2.4, 2.0, 0.8, 0.1],
}

# Create data for absolute errors - 7B-4T model
abs_data_7b = {
    'Task Name': [
        'MMLU', 'HellaSwag', 'ARC-Challenge', 'ARC-Easy',
        'PIQA', 'CommonsenseQA', 'Social IQa', 'OpenBookQA'
    ],
    'Examples': [14042, 10042, 1172, 2376, 1838, 1221, 1954, 500],
    'Task Loss': [0.6, 1.2, 10.4, 8.0, 0.8, 3.1, 1.2, 5.2],
    'C4 Loss': [1.1, 3.7, 0.9, 1.0, 0.6, 4.2, 3.7, 1.0],
    'TaskCE Loss': [9.0, 5.9, 13.1, 4.5, 2.5, 1.9, 0.5, 3.5],
}

# Create data for relative errors - 13B-5T model
rel_data_13b = {
    'Task Name': [
        'MMLU', 'HellaSwag', 'ARC-Challenge', 'ARC-Easy',
        'PIQA', 'CommonsenseQA', 'Social IQa', 'OpenBookQA'
    ],
    'Examples': [14042, 10042, 1172, 2376, 1838, 1221, 1954, 500],
    'Task Loss': [0.7, 2.5, 17.5, 11.4, 1.1, 4.7, 2.7, 7.8],
    'C4 Loss': [5.0, 5.7, 3.9, 1.9, 1.5, 7.0, 7.6, 10.4],
    'TaskCE Loss': [20.2, 10.5, 19.5, 6.2, 2.9, 2.7, 1.2, 0.2],
}

# Create data for relative errors - 7B-4T model
rel_data_7b = {
    'Task Name': [
        'MMLU', 'HellaSwag', 'ARC-Challenge', 'ARC-Easy',
        'PIQA', 'CommonsenseQA', 'Social IQa', 'OpenBookQA'
    ],
    'Examples': [14042, 10042, 1172, 2376, 1838, 1221, 1954, 500],
    'Task Loss': [1.3, 1.4, 16.9, 9.4, 1.0, 4.2, 2.0, 10.6],
    'C4 Loss': [2.2, 4.5, 1.4, 1.2, 0.7, 5.8, 6.2, 2.0],
    'TaskCE Loss': [18.3, 7.2, 21.3, 5.4, 3.1, 2.6, 0.7, 7.1],
}

# Calculate row and column averages
for data in [abs_data_13b, abs_data_7b, rel_data_13b, rel_data_7b]:
    # Add row average
    data['Row Average'] = []
    for i in range(len(data['Task Name'])):
        row_avg = np.mean([data['Task Loss'][i], data['C4 Loss'][i], data['TaskCE Loss'][i]])
        data['Row Average'].append(round(row_avg, 2))
    
    # Add column averages
    for key in ['Task Loss', 'C4 Loss', 'TaskCE Loss', 'Row Average']:
        data[key].append(round(np.mean(data[key][:-1]), 2))
    data['Task Name'].append('Average')
    data['Examples'].append('-')

# Create figure with 2x2 grid - adjusted with more vertical space between plots
fig, axes = plt.subplots(2, 2, figsize=(20, 18))

# Create custom colormaps
def custom_cmap(max_val):
    """Create a custom colormap for the heatmap cells - red with lighter colors"""
    # Create a custom colormap from white to dark red
    colors = [(1, 1, 1), (0.98, 0.8, 0.8), (0.95, 0.6, 0.6), (0.9, 0.4, 0.4), (0.8, 0.2, 0.2)]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', colors)
    norm = mcolors.Normalize(vmin=0, vmax=max_val)
    return cmap, norm

# Set titles with smaller font size and more padding
axes[0, 0].set_title('7B-4T Model - Absolute Errors', fontsize=14, pad=10)
axes[0, 1].set_title('13B-5T Model - Absolute Errors', fontsize=14, pad=10)
axes[1, 0].set_title('7B-4T Model - Relative Errors (%)', fontsize=14, pad=10)
axes[1, 1].set_title('13B-5T Model - Relative Errors (%)', fontsize=14, pad=10)

# Create DataFrames
abs_df_7b = pd.DataFrame(abs_data_7b)
abs_df_13b = pd.DataFrame(abs_data_13b)
rel_df_7b = pd.DataFrame(rel_data_7b)
rel_df_13b = pd.DataFrame(rel_data_13b)

# Adjust figure layout for better spacing
plt.subplots_adjust(hspace=0.3)

# Create tables
# Absolute Error - 7B-4T (top left)
abs_cmap_7b, abs_norm_7b = custom_cmap(30)
abs_table_7b = axes[0, 0].table(
    cellText=abs_df_7b.values,
    colLabels=abs_df_7b.columns,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)
abs_table_7b.auto_set_font_size(False)
abs_table_7b.set_fontsize(12)

# Absolute Error - 13B-5T (top right)
abs_cmap_13b, abs_norm_13b = custom_cmap(30)
abs_table_13b = axes[0, 1].table(
    cellText=abs_df_13b.values,
    colLabels=abs_df_13b.columns,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)
abs_table_13b.auto_set_font_size(False)
abs_table_13b.set_fontsize(12)

# Relative Error - 7B-4T (bottom left)
rel_df_7b_display = rel_df_7b.copy()
# Format to add % sign to relative errors
for col in ['Task Loss', 'C4 Loss', 'TaskCE Loss', 'Row Average']:
    rel_df_7b_display[col] = rel_df_7b_display[col].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)

rel_cmap_7b, rel_norm_7b = custom_cmap(35)
rel_table_7b = axes[1, 0].table(
    cellText=rel_df_7b_display.values,
    colLabels=rel_df_7b_display.columns,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)
rel_table_7b.auto_set_font_size(False)
rel_table_7b.set_fontsize(12)

# Relative Error - 13B-5T (bottom right)
rel_df_13b_display = rel_df_13b.copy()
# Format to add % sign to relative errors
for col in ['Task Loss', 'C4 Loss', 'TaskCE Loss', 'Row Average']:
    rel_df_13b_display[col] = rel_df_13b_display[col].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)

rel_cmap_13b, rel_norm_13b = custom_cmap(35)
rel_table_13b = axes[1, 1].table(
    cellText=rel_df_13b_display.values,
    colLabels=rel_df_13b_display.columns,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)
rel_table_13b.auto_set_font_size(False)
rel_table_13b.set_fontsize(12)

# Color the cells for all tables
def color_table_cells(table, df, cmap, norm, is_relative=False):
    # Color headers
    for j, col in enumerate(df.columns):
        cell = table[0, j]
        cell.set_facecolor('#f0f0f0')  # Light gray for headers
        cell.set_text_props(weight='bold')
    
    # Color loss cells
    loss_cols = ['Task Loss', 'C4 Loss', 'TaskCE Loss', 'Row Average']
    loss_col_indices = [list(df.columns).index(col) for col in loss_cols]
    
    for i in range(len(df)):
        for j, col_idx in enumerate(loss_col_indices):
            cell = table[i+1, col_idx]
            
            # Get value for coloring
            if is_relative:
                val_str = df.iloc[i, col_idx]
                if isinstance(val_str, str) and '%' in val_str:
                    val = float(val_str.replace('%', ''))
                else:
                    val = float(val_str) if isinstance(val_str, (int, float)) else 0
            else:
                val = df.iloc[i, col_idx]
            
            # Set cell color based on value
            color = cmap(norm(val))
            cell.set_facecolor(color)
            
            # Make text white if background is dark
            if val > 20:  # Threshold for dark cells
                cell.set_text_props(color='white')
            
            # If average row, make bold
            if i == len(df) - 1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e5e5e5')  # Light gray for average row
        
        # Color task name and examples columns
        task_name_cell = table[i+1, 0]
        examples_cell = table[i+1, 1]
        
        if i == len(df) - 1:  # Average row
            task_name_cell.set_text_props(weight='bold')
            examples_cell.set_text_props(weight='bold')
            task_name_cell.set_facecolor('#e5e5e5')  # Light gray
            examples_cell.set_facecolor('#e5e5e5')  # Light gray
        else:
            task_name_cell.set_facecolor('#f5f5f5')  # Very light gray
            examples_cell.set_facecolor('#f5f5f5')  # Very light gray

# Apply cell coloring
color_table_cells(abs_table_7b, abs_df_7b, abs_cmap_7b, abs_norm_7b)
color_table_cells(abs_table_13b, abs_df_13b, abs_cmap_13b, abs_norm_13b)
color_table_cells(rel_table_7b, rel_df_7b, rel_cmap_7b, rel_norm_7b, is_relative=True)
color_table_cells(rel_table_13b, rel_df_13b, rel_cmap_13b, rel_norm_13b, is_relative=True)

# Turn off axes
for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout(pad=3.0)
plt.savefig('src/scripts/paper/figures/intermediate_feature_comparison.png', dpi=300, bbox_inches='tight')
plt.show()