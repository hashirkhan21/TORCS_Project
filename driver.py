import msgParser
import carState
import carControl
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

class Driver(object):
    '''
    A driver object for the SCRC that uses a Random Forest model for predictions
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        
        # Feature list - what we'll use for prediction
        self.features = [
            'angle', 'trackPos', 'speedX', 'speedY', 'speedZ', 'rpm', 'gear',
            'distRaced', 'fuel'
        ]
        
        # Models for steering, gear, and acceleration
        self.steer_model = None
        self.gear_model = None
        self.accel_model = None
        
        # Try to load existing models or train new ones
        self.load_or_train_models()
        
    def load_or_train_models(self):
        """Load existing models if available, otherwise train new ones from CSV data"""
        models_exist = os.path.exists('steer_model.pkl') and os.path.exists('gear_model.pkl') and os.path.exists('accel_model.pkl')
        
        if models_exist:
            print("Loading existing models...")
            self.steer_model = pickle.load(open('steer_model.pkl', 'rb'))
            self.gear_model = pickle.load(open('gear_model.pkl', 'rb'))
            self.accel_model = pickle.load(open('accel_model.pkl', 'rb'))
        else:
            print("Training new models from CSV data...")
            try:
                # Try to find a CSV file in the current directory
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                if csv_files:
                    print(f"Found CSV file(s): {csv_files}")
                    self.train_models_from_csv(csv_files[0])  # Use the first CSV file found
                else:
                    print("No CSV files found. Using fallback driving logic.")
            except Exception as e:
                print(f"Error training models: {e}")
                print("Using fallback driving logic")
    
    def train_models_from_csv(self, csv_path):
        """Train Random Forest models from CSV data"""
        try:
            data = pd.read_csv(csv_path)
            for col in ['parser', 'sensor']:
                if col in data.columns:
                    data = data.drop(col, axis=1)

            available_features = []
            for feature in self.features:
                if feature in data.columns:
                    available_features.append(feature)
            
            if 'track' in data.columns:
                available_features.append('track')
            
            # Add additional features that exist in the dataset
            additional_features = ['curLapTime', 'damage', 'distFromS', 'distRaced', 
                                  'lastLapTime', 'opponents', 'racePos', 'wheelSpinVel']
            for feature in additional_features:
                if feature in data.columns:
                    available_features.append(feature)
                    if feature not in self.features:
                        self.features.append(feature)
            
            print(f"Using features: {available_features}")
            X = data[available_features]
            
            # Train steering model
            y_steer = data['steer']
            self.steer_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.steer_model.fit(X, y_steer)
            print("Steering model trained")
            
            # Train gear model
            y_gear = data['gear'].astype(int)
            self.gear_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.gear_model.fit(X, y_gear)
            print("Gear model trained")
            
            # Train acceleration model
            y_accel = data['accel']
            self.accel_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.accel_model.fit(X, y_accel)
            print("Acceleration model trained")
            
            # Save trained models
            pickle.dump(self.steer_model, open('steer_model.pkl', 'wb'))
            pickle.dump(self.gear_model, open('gear_model.pkl', 'wb'))
            pickle.dump(self.accel_model, open('accel_model.pkl', 'wb'))
            
            print("Models trained and saved successfully")
            
        except Exception as e:
            print(f"Failed to train models from CSV: {e}")
            raise
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        if self.steer_model and self.gear_model and self.accel_model:
            self.model_drive()
        else:
            # Fallback to original driving logic
            self.steer()
            self.gear()
            self.speed()
        
        return self.control.toMsg()
    
    def model_drive(self):
        """Use trained models to drive the car"""
        # Extract features from the current state
        features = self.extract_features()
        
        # Make predictions if we have all needed features
        try:
            # Predict steering
            steer_value = self.steer_model.predict([features])[0]
            self.control.setSteer(np.clip(steer_value, -1.0, 1.0))
            
            # Predict gear
            gear_value = int(round(self.gear_model.predict([features])[0]))
            # Ensure gear is valid (between -1 and 6)
            gear_value = max(-1, min(6, gear_value))
            self.control.setGear(gear_value)
            
            # Predict acceleration
            accel_value = self.accel_model.predict([features])[0]
            self.control.setAccel(np.clip(accel_value, 0.0, 1.0))
            
            # Set brake to 0 since we're only modeling acceleration and we don't want both applied
            self.control.setBrake(0)
            
            # Store the current RPM for the next cycle
            self.prev_rpm = self.state.getRpm()
            
            # Print occasional status
            if self.state.getDistRaced() % 100 < 1:
                print(f"Distance: {self.state.getDistRaced():.1f}, Speed: {self.state.getSpeedX():.1f}, " + 
                      f"Steering: {steer_value:.3f}, Accel: {accel_value:.2f}, Gear: {gear_value}")
            
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Fallback to original driving logic
            self.steer()
            self.gear()
            self.speed()
    
    def extract_features(self):
        """Extract features from the current state to feed into the model"""
        # Start with basic features
        feature_dict = {
            'angle': self.state.angle,
            'trackPos': self.state.trackPos,
            'speedX': self.state.getSpeedX(),
            'speedY': self.state.getSpeedY(),
            'speedZ': self.state.getSpeedZ(),
            'rpm': self.state.getRpm(),
            'gear': self.state.getGear(),
            'distRaced': self.state.getDistRaced(),
            'fuel': self.state.getFuel()
        }
        
        # Add track sensors if we're using them
        if 'track' in self.features:
            track_sensors = self.state.getTrack()
            if track_sensors and len(track_sensors) > 0:
                # Use the middle sensor reading if there are multiple
                feature_dict['track'] = track_sensors[len(track_sensors) // 2]
        
        # Add additional features if they're being used by our models
        if 'curLapTime' in self.features:
            feature_dict['curLapTime'] = self.state.getCurLapTime()
        if 'damage' in self.features:
            feature_dict['damage'] = self.state.getDamage()
        if 'distFromS' in self.features:
            feature_dict['distFromS'] = self.state.getDistFromStart()
        if 'lastLapTime' in self.features:
            feature_dict['lastLapTime'] = self.state.getLastLapTime()
        if 'racePos' in self.features:
            feature_dict['racePos'] = self.state.getRacePos()
        
        # Convert to list in the same order as self.features
        feature_values = []
        for feature in self.features:
            if feature in feature_dict:
                feature_values.append(feature_dict[feature])
            else:
                # Use a default value if the feature is not available
                feature_values.append(0)
        
        return feature_values
    
    # Original methods kept as fallback
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
        self.prev_rpm = rpm
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
            
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass