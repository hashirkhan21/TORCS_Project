import pandas as pd
import ast
import argparse
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_csv(csv_path, output_path='preprocessed_data.csv', normalize=True):
    try:
        print(f"[INFO] Attempting to load: {csv_path}")
        data = pd.read_csv(csv_path)
        print(f"[✓] Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV file at '{csv_path}': {e}")
        return  # exit early since nothing can proceed without the file

    # Drop unused metadata columns
    for col in ['parser', 'sensors', 'actions', 'parser.1']:
        if col in data.columns:
            print(f"[INFO] Dropping column: {col}")
            data = data.drop(col, axis=1)

    # Expand list-like columns
    def expand_column(df, col_name):
        if col_name in df.columns:
            try:
                expanded = df[col_name].apply(ast.literal_eval)
                expanded_df = pd.DataFrame(expanded.tolist(), index=df.index)
                expanded_df.columns = [f"{col_name}_{i}" for i in range(expanded_df.shape[1])]
                df = df.drop(col_name, axis=1)
                df = pd.concat([df, expanded_df], axis=1)
                print(f"[✓] Expanded column: {col_name}")
            except Exception as e:
                print(f"[WARN] Failed to expand {col_name}: {e}")
        return df

    for col in ['track', 'opponents', 'wheelSpinVel', 'focus']:
        data = expand_column(data, col)

    # Convert columns to float where possible
    for col in data.columns:
        try:
            data[col] = data[col].astype(float)
        except:
            print(f"[WARN] Dropping unconvertible column: {col}")
            data = data.drop(col, axis=1)

    # Normalize data if required
   
    print("[INFO] Normalizing data...")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    joblib.dump(scaler, 'scaler.pkl')  # Save scaler for reuse
    print("[✓] Normalization complete and scaler saved.")

    # Save preprocessed output
    try:
        data.to_csv(output_path, index=False)
        print(f"[✓] Preprocessed data saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save preprocessed data: {e}")



if __name__ == "__main__":
    preprocess_csv(r"Odata1.csv")
   
