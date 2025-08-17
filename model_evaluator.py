import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        self.metrics_history = {
            'steer': [],
            'gear': [],
            'accel': []
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
    
    def evaluate_models(self, data, models):
        """Evaluate model performance and update if better"""
        results = {}
        
        for target, model in models.items():
            if target not in data.columns:
                continue
                
            X = data.drop(['steer', 'gear', 'accel'], axis=1, errors='ignore')
            y = data[target]
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results[target] = {
                'mse': mse,
                'r2': r2
            }
            
            # Update metrics history
            self.metrics_history[target].append({
                'timestamp': datetime.now(),
                'mse': mse,
                'r2': r2
            })
            
            # Check if this is the best model so far
            if mse < self.best_scores[target]:
                self.best_scores[target] = mse
                self.best_models[target] = model
                
                # Save the best model
                joblib.dump(model, f'best_{target}_model.pkl')
                print(f"[✓] New best model saved for {target} (MSE: {mse:.4f}, R2: {r2:.4f})")
        
        return results
    
    def get_model_improvements(self, data):
        """Suggest improvements for the models"""
        improvements = {}
        
        for target in ['steer', 'gear', 'accel']:
            if target not in data.columns:
                continue
                
            X = data.drop(['steer', 'gear', 'accel'], axis=1, errors='ignore')
            y = data[target]
            
            # Train a new model with more trees
            new_model = RandomForestRegressor(
                n_estimators=500,  # More trees
                max_depth=None,    # Allow full depth
                min_samples_split=2,
                min_samples_leaf=1,
                warm_start=True,
                random_state=42
            )
            
            new_model.fit(X, y)
            
            # Compare with current best
            y_pred = new_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            
            if mse < self.best_scores[target]:
                improvements[target] = {
                    'current_mse': self.best_scores[target],
                    'new_mse': mse,
                    'improvement': (self.best_scores[target] - mse) / self.best_scores[target] * 100,
                    'model': new_model
                }
        
        return improvements
    
    def save_metrics_history(self, filename='model_metrics_history.csv'):
        """Save metrics history to CSV"""
        history_data = []
        
        for target, metrics in self.metrics_history.items():
            for metric in metrics:
                history_data.append({
                    'target': target,
                    'timestamp': metric['timestamp'],
                    'mse': metric['mse'],
                    'r2': metric['r2']
                })
        
        df = pd.DataFrame(history_data)
        df.to_csv(filename, index=False)
        print(f"[✓] Metrics history saved to {filename}")
    
    def load_best_models(self):
        """Load the best models from disk"""
        for target in ['steer', 'gear', 'accel']:
            model_path = f'best_{target}_model.pkl'
            if os.path.exists(model_path):
                self.best_models[target] = joblib.load(model_path)
                print(f"[✓] Loaded best model for {target}")
    
    def get_feature_importance(self, data):
        """Get feature importance for each model"""
        importance_data = {}
        
        for target, model in self.best_models.items():
            if model is None or target not in data.columns:
                continue
                
            X = data.drop(['steer', 'gear', 'accel'], axis=1, errors='ignore')
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            importance_data[target] = feature_importance
        
        return importance_data

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Load existing models
    models = {}
    for target in ['steer', 'gear', 'accel']:
        model_path = f'{target}_model.pkl'
        if os.path.exists(model_path):
            models[target] = joblib.load(model_path)
    
    # Load data
    if os.path.exists('preprocessed_data.csv'):
        data = pd.read_csv('preprocessed_data.csv')
        
        # Evaluate current models
        results = evaluator.evaluate_models(data, models)
        
        # Get improvement suggestions
        improvements = evaluator.get_model_improvements(data)
        
        # Save metrics history
        evaluator.save_metrics_history()
        
        # Get feature importance
        importance = evaluator.get_feature_importance(data)
        
        # Print results
        print("\nModel Evaluation Results:")
        for target, metrics in results.items():
            print(f"\n{target.upper()} Model:")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"R2 Score: {metrics['r2']:.4f}")
            
            if target in improvements:
                print(f"\nPotential Improvement:")
                print(f"Current MSE: {improvements[target]['current_mse']:.4f}")
                print(f"New MSE: {improvements[target]['new_mse']:.4f}")
                print(f"Improvement: {improvements[target]['improvement']:.2f}%")
            
            if target in importance:
                print("\nTop 5 Important Features:")
                print(importance[target].head()) 