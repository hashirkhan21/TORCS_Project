import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_models(csv_path='preprocessed_data.csv', model_dir='models'):
    data = pd.read_csv(csv_path)

    target_columns = ['steer', 'accel', 'gear']
    feature_columns = [col for col in data.columns if col not in target_columns]

    models = {}
    for target in target_columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(data[feature_columns], data[target])
        models[target] = model
        joblib.dump(model, f"{target}_model.pkl")
        print(f"[✓] Saved model for {target} to {target}_model.pkl")

if __name__ == "__main__":
    train_models()
