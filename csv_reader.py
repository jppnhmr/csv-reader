import os
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

directory_path = os.path.dirname(os.path.abspath(__file__))
GRAPH_FOLDER = "graphs"
STATS_FOLDER = "stats"

def choose_from_list(vals):
    while True:
        for i, x in enumerate(vals):
            print(f"{i+1} - {x}")

        print("Enter choice\n: ", end="")
        try:
            choice = input()
            choice = int(choice) - 1
            return vals[choice]
        except Exception:
            print("Invalid Choice. Try again.\n")

def extract_cols_rows(csv_file):
    rows = []
    columns = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        start_pos = f.tell()
        first_line = f.readline()
        while first_line.startswith("#"):
            start_pos = f.tell()
            first_line = f.readline()

        f.seek(start_pos)

        reader = csv.reader(f)
        columns = next(reader) # header
        for row in reader:
            rows.append(row)

    return columns, rows

def clean_rows(cols, rows):
    clean_rows = []
    count = 0
    
    for row in rows:
        # Row with too many or too few columns
        if len(row) > len(cols) or len(row) < len(cols):
            count += 1
            continue

        clean_rows.append(row)

    print(f"Removed {count} unclean rows.")
    return clean_rows

def detect_type(value):
    val = value.strip()

    # None
    if val == "" or val.lower() in ("na", "null", "none"):
        return type(None)

    # Boolean
    if val.lower() in ("true", "false", "yes", "no"):
        return bool

    # Number
    for t in (int, float):
        try:
            t(val)
            return t
        except ValueError:
            pass

    # Datetime
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            datetime.strptime(val, fmt)
            return datetime
        except ValueError:
            pass 

    # String
    return str

def merge_types(types):
    """
    Return most appropriate data type given a set of types
    Detect Order: int -> float -> datetime -> bool -> str
    """
    types = {t for t in types if t is not type(None)}
    if not types:
        return type(None)
    if types == {int}:
        return int
    if types <= {int, float}:
        return float
    if types == {datetime}:
        return datetime
    if types == {bool}:
        return bool
    # default
    return str

def detect_column_types(rows):
    if not rows:
        return []
    num_cols = len(rows[0])
    col_type_sets = [set() for _ in range(num_cols)]

    limit = min(len(rows), 200) # scan many rows
    for r in rows[:limit]:
        for i, val in enumerate(r):
            col_type_sets[i].add(detect_type(val))

    return [merge_types(s) for s in col_type_sets]

def outlier_mask_iqr(series):
    """
    Return a boolean series, indicating True for values in the series that are outliers
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)

def remove_outliers(df):
    """
    Returns a DataFrame with all rows containing numeric outliers removed
    """
    numeric_cols = df.select_dtypes(include="number").columns

    global_outlier_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        if col == "ID": # ignore ID col
            continue

        col_mask = outlier_mask_iqr(df[col].dropna())
        # reindex incase NaNs were dropped
        col_mask = col_mask.reindex(df.index, fill_value=False)

        #if col_mask.any():
            #print(f"Removed {col_mask.sum()} rows because of {col}")

        global_outlier_mask = global_outlier_mask | col_mask

    # return rows that aren't outliers
    df_clean = df[~global_outlier_mask].copy()
    return df_clean

def get_num_stats(df):
    """
    Returns the Mean, Median & Mode of all number cols.
    """
    df = df.select_dtypes(include='number')

    num_stats = ""
    for col_name in df.describe():
        col = df[col_name]
        col_stats = f"""
{col_name}
Mean: {col.mean()}
Median: {col.median()}
Mode: {col.mode().to_list()}
"""
        num_stats += col_stats
    return num_stats

if __name__ == "__main__":

    all_files = os.listdir(directory_path)
    csv_files = []

    for item in all_files:
        if item[-4:] == ".csv":
            csv_files.append(item)
    
    csv_file = choose_from_list(csv_files)

    cols, rows = extract_cols_rows(csv_file)
    rows = clean_rows(cols, rows)
    data_types = detect_column_types(rows)

    df = pd.DataFrame(data=rows, columns=cols)  
    
    # convert columns to detected data types
    for col, dt in zip(cols, data_types):
        if dt is int:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')
        elif dt is float:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dt is datetime:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dt is bool:
            df[col] = df[col].str.lower().map({'true': True, 'false': False, 'yes': True, 'no': False})
        else:
            df[col] = df[col].astype("string")

    df = remove_outliers(df)

    filename = csv_file[:-4]
    if not (os.path.exists(filename)):
        os.makedirs(filename)
    if not (os.path.exists(f"{filename}/{GRAPH_FOLDER}")):
        os.makedirs(f"{filename}/{GRAPH_FOLDER}")

    # Create histogram for each Numeric value
    df_num = df.select_dtypes(include="number")
    for col in df_num.columns:
        if col == "id" or col == "ID": # ignore ID cols
            continue
        obj = df_num[col]
        obj.hist(figsize=(8,5))
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel("frequency")
        plt.savefig(fname=f"{filename}/{GRAPH_FOLDER}/{col}_histogram")
        plt.close()

    # Create HTML summary

    # Stats
    html_stats = get_num_stats(df)
    # Graphs
    html_graphs = ""
    graphs = os.listdir(f"{filename}/{GRAPH_FOLDER}")
    for graph in graphs:
        html_graphs += f'<img src="{GRAPH_FOLDER}/{graph}">'

    html_text = f"""
<html>
<head>
    <title>CSV Analysis Report: {filename}</title>
</head>
<body>
    <h1>Analysis Results</h1>
    
    <h2>Statistics</h2>
    <pre>{html_stats}</pre>
    
    <h2>Graphs</h2>
    {html_graphs}
</body>
</html>
"""
    with open(f"{filename}/{filename}_summary.html", "w") as f:
        f.write(html_text)

    print("--- Analysis Complete ---")  
