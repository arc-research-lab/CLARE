import pandas as pd
import sys

def load_xlsx_to_df(filepath):
    # Read Excel file with first row as header and first column as index
    df = pd.read_excel(filepath, header=0, index_col=0)
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_xlsx>")
        sys.exit(1)

    filepath = sys.argv[1]
    df = load_xlsx_to_df(filepath)
    print(df)
