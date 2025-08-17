import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class ModelManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.scaler = None
        self.models = {
            'steer': None,
            'gear': None,
            'accel': None
        }
        self.best_models = {
            'steer': None,
            'gear': None,
            'accel': None
        }
        self.best_scores = {
            'steer': float('inf'),
            'gear': float('inf'),
            'accel': float('inf')
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models if they exist
        self.load_models()
    
    def load_models(self):
        """Load existing models and scaler"""
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("[✓] Loaded existing scaler")
            
            # Load models
            for target in self.models.keys():
                model_path = os.path.join(self.model_dir, f'{target}_model.json')
                if os.path.exists(model_path):
                    self.models[target] = xgb.XGBRegressor()
                    self.models[target].load_model(model_path)
                    print(f"[✓] Loaded {target} model")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
    
    def save_models(self):
        """Save current models and scaler"""
        try:
            print("[INFO] Saving models to directory:", self.model_dir)
            
            # Save scaler
            if self.scaler is not None:
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
                print(f"[✓] Saved scaler to {scaler_path}")
            
            # Save models
            for target, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(self.model_dir, f'{target}_model.json')
                    model.save_model(model_path)
                    print(f"[✓] Saved {target} model to {model_path}")
            
            print("[✓] All models and scaler saved successfully")
        except Exception as e:
            print(f"[ERROR] Failed to save models: {e}")
            import traceback
            traceback.print_exc()
    
    def train_models(self, data_path, test_size=0.2):
        """Train models on preprocessed data"""
        try:
            print(f"[INFO] Loading data from {data_path}")
            data = pd.read_csv(data_path)
            print(f"[INFO] Loaded {len(data)} samples")
            
            # Prepare features and targets
            target_columns = ['steer', 'gear', 'accel']
            feature_columns = [col for col in data.columns if col not in target_columns]
            print(f"[INFO] Using features: {feature_columns}")
            print(f"[INFO] Number of features: {len(feature_columns)}")
            
            # Initialize scaler if not exists
            if self.scaler is None:
                print("[INFO] Initializing new scaler")
                self.scaler = StandardScaler()
                self.scaler.fit(data[feature_columns])
                print(f"[INFO] Scaler initialized with {self.scaler.n_features_in_} features")
                print(f"[INFO] Feature names: {self.scaler.feature_names_in_}")
            
            # Scale features
            X = self.scaler.transform(data[feature_columns])
            print("[INFO] Features scaled successfully")
            
            # Train each model
            for target in target_columns:
                if target not in data.columns:
                    print(f"[WARNING] Target column {target} not found in data")
                    continue
                
                print(f"\n[INFO] Training {target} model...")
                y = data[target].values
                
                # Create and train model
                model = xgb.XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.01,
                    max_depth=7,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42
                )
                
                # Train model
                model.fit(X, y)
                print(f"[INFO] {target} model trained")
                
                # Evaluate model
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                print(f"\n{target.upper()} Model Performance:")
                print(f"MSE: {mse:.4f}")
                print(f"R2 Score: {r2:.4f}")
                
                # Update model if better
                if mse < self.best_scores[target]:
                    self.best_scores[target] = mse
                    self.best_models[target] = model
                    self.models[target] = model
                    print(f"[✓] New best model for {target}")
                
            # Save models
            print("\n[INFO] Saving models...")
            self.save_models()
            print("[✓] All models saved successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to train models: {e}")
            import traceback
            traceback.print_exc()
    
    def predict(self, features):
        """Make predictions using the models"""
        try:
            if features is None:
                print("[ERROR] No features provided for prediction")
                return None
                
            # Convert features to numpy array if it's not already
            features = np.array(features).reshape(1, -1)
            
            # Verify feature count
            if self.scaler is not None:
                if features.shape[1] != self.scaler.n_features_in_:
                    print(f"[ERROR] Feature count mismatch. Expected {self.scaler.n_features_in_}, got {features.shape[1]}")
                    print(f"Scaler feature names: {self.scaler.feature_names_in_}")
                    return None
            
            # Scale features
            if self.scaler is not None:
                try:
                    features = self.scaler.transform(features)
                except Exception as e:
                    print(f"[ERROR] Failed to scale features: {e}")
                    print(f"Features shape: {features.shape}")
                    print(f"Scaler expecting: {self.scaler.n_features_in_} features")
                    return None
            
            predictions = {}
            for target, model in self.models.items():
                if model is not None:
                    try:
                        pred = model.predict(features)[0]
                        predictions[target] = pred
                    except Exception as e:
                        print(f"[ERROR] Failed to predict {target}: {e}")
                        predictions[target] = None
            
            return predictions
            
        except Exception as e:
            print(f"[ERROR] Failed to make predictions: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    
    # Train on preprocessed data
    if os.path.exists('preprocessed_data.csv'):
        manager.train_models('preprocessed_data.csv') 