import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.scaler = None
        self.feature_columns = [
            'angle', 'trackPos', 'speedX', 'speedY', 'speedZ', 'rpm', 'gear',
            'distRaced', 'fuel'
        ]
        # Add track sensors
        for i in range(19):
            self.feature_columns.append(f'track_{i}')
        
        # Load existing scaler if available
        self.load_scaler()
    
    def load_scaler(self):
        """Load existing scaler if available"""
        try:
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"[✓] Loaded existing scaler with {self.scaler.n_features_in_} features")
                print(f"[INFO] Feature names: {self.scaler.feature_names_in_}")
        except Exception as e:
            print(f"[ERROR] Failed to load scaler: {e}")
    
    def preprocess_data(self, input_file, output_file='preprocessed_data.csv'):
        """Preprocess raw data and save to output file"""
        try:
            print(f"[INFO] Loading raw data from {input_file}")
            data = pd.read_csv(input_file)
            print(f"[INFO] Loaded {len(data)} samples")
            
            # Verify required columns
            required_columns = self.feature_columns + ['steer', 'accel', 'gear']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"[ERROR] Missing required columns: {missing_columns}")
                return False
            
            # Initialize scaler if not exists
            if self.scaler is None:
                print("[INFO] Initializing new scaler")
                self.scaler = StandardScaler()
                self.scaler.fit(data[self.feature_columns])
                print(f"[INFO] Scaler initialized with {self.scaler.n_features_in_} features")
                
                # Save scaler
                os.makedirs(self.model_dir, exist_ok=True)
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
                print(f"[✓] Saved scaler to {scaler_path}")
            
            # Scale features
            X = self.scaler.transform(data[self.feature_columns])
            print("[INFO] Features scaled successfully")
            
            # Create preprocessed dataframe
            preprocessed_data = pd.DataFrame(X, columns=self.feature_columns)
            
            # Add target columns
            for target in ['steer', 'accel', 'gear']:
                preprocessed_data[target] = data[target]
            
            # Save preprocessed data
            preprocessed_data.to_csv(output_file, index=False)
            print(f"[✓] Saved preprocessed data to {output_file}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to preprocess data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def transform_features(self, features):
        """Transform a single feature vector using the scaler"""
        try:
            if self.scaler is None:
                print("[WARNING] No scaler available, returning original features")
                return features
            
            # Convert to numpy array if needed
            features = np.array(features).reshape(1, -1)
            
            # Verify feature count
            if features.shape[1] != self.scaler.n_features_in_:
                print(f"[ERROR] Feature count mismatch. Expected {self.scaler.n_features_in_}, got {features.shape[1]}")
                return None
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            return scaled_features[0]  # Return 1D array
            
        except Exception as e:
            print(f"[ERROR] Failed to transform features: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Preprocess raw data
    if os.path.exists('raw_data.csv'):
        preprocessor.preprocess_data('raw_data.csv') 