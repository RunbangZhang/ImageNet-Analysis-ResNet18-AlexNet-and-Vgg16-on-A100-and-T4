import csv
import pandas as pd

EXPECTED_COLUMNS = 12  # Set EXPECTED_COLUMNS to the actual number of columns

# Automatically detect the starting row with the correct number of columns
def detect_start_row(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= EXPECTED_COLUMNS:  # Check if the row has the required number of columns
                return i
    return 0  # Default to the first row if no suitable row is found

file_path = "report.csv"
start_row = detect_start_row(file_path)

# Read the file, skipping rows up to the detected start row
df = pd.read_csv(file_path, skiprows=start_row)

# Ensure the 'Metric Value' column is numeric; non-numeric values will be converted to NaN
df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce')

# Group by 'Metric Name' and compute the sum of 'Metric Value' for each group
metric_sums = df.groupby('Metric Name')['Metric Value'].sum()

# Calculate total FLOPs using specific metric names, applying the appropriate weights
total_flops = (
    metric_sums.get('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum', 0) * 2 +
    metric_sums.get('smsp__sass_thread_inst_executed_op_fadd_p.sum', 0) +
    metric_sums.get('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum', 0) +
    metric_sums.get('smsp__sass_thread_inst_executed_op_fp16_pred_on.sum', 0)
)

print(metric_sums)
print("Total FLOPs: {}".format(int(total_flops)))
