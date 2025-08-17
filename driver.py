import msgParser
import carState
import carControl
import pandas as pd
import numpy as np
import ast
from model_manager import ModelManager
import os
import threading
import queue
from datetime import datetime

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
        
        # Add track sensors to features (19 sensors)
        for i in range(19):
            self.features.append(f'track_{i}')
        
        print(f"[INFO] Initialized with {len(self.features)} features: {self.features}")
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Data collection
        self.data_queue = queue.Queue()
        self.collecting_data = False
        self.data_thread = None
        self.min_samples_for_training = 1000
        self.training_interval = 5000
        
        # Start data collection
        self.start_data_collection()
    
    def start_data_collection(self):
        """Start the data collection thread"""
        self.collecting_data = True
        self.data_thread = threading.Thread(target=self._collect_data_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
    
    def _collect_data_loop(self):
        """Background thread for collecting and processing data"""
        collected_data = []
        sample_count = 0
        print("[INFO] Starting data collection loop...")
        
        while self.collecting_data:
            try:
                data = self.data_queue.get(timeout=1.0)
                collected_data.append(data)
                sample_count += 1
                
                if sample_count % 100 == 0:  # Print progress every 100 samples
                    print(f"[INFO] Collected {sample_count} samples")
                
                if sample_count >= self.training_interval:
                    print(f"[INFO] Collected {sample_count} samples, starting training...")
                    # Save collected data
                    df = pd.DataFrame(collected_data)
                    df.to_csv('preprocessed_data.csv', index=False)
                    print(f"[INFO] Saved {len(df)} samples to preprocessed_data.csv")
                    
                    # Train models
                    self.model_manager.train_models('preprocessed_data.csv')
                    
                    collected_data = []
                    sample_count = 0
                    print("[INFO] Training completed, resetting sample counter")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Error in data collection: {e}")
                import traceback
                traceback.print_exc()
    
    def collect_current_state(self):
        """Collect current state data for training"""
        try:
            state_data = {
                'angle': self.state.angle,
                'trackPos': self.state.trackPos,
                'speedX': self.state.getSpeedX(),
                'speedY': self.state.getSpeedY(),
                'speedZ': self.state.getSpeedZ(),
                'rpm': self.state.getRpm(),
                'gear': self.state.getGear(),
                'distRaced': self.state.getDistRaced(),
                'fuel': self.state.getFuel(),
                'steer': self.control.getSteer(),
                'accel': self.control.getAccel(),
                'gear': self.control.getGear()
            }
            
            # Add track sensors
            track_sensors = self.state.getTrack()
            if track_sensors:
                for i, sensor in enumerate(track_sensors):
                    state_data[f'track_{i}'] = sensor
            
            # Ensure all features are present in collected data
            for feature in self.features:
                if feature not in state_data:
                    state_data[feature] = 0.0
            
            # Verify feature count
            if len(state_data) != len(self.features) + 3:  # +3 for steer, accel, gear
                print(f"[WARNING] Feature count mismatch. Expected {len(self.features) + 3}, got {len(state_data)}")
                print(f"Features: {list(state_data.keys())}")
            
            self.data_queue.put(state_data)
            
        except Exception as e:
            print(f"[ERROR] Error collecting state data: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Collect data for training
        self.collect_current_state()
        
        # Use model predictions if available, otherwise use basic driving
        if all(model is not None for model in self.model_manager.models.values()):
            self.model_drive()
        else:
            self.basic_drive()
        
        return self.control.toMsg()
    
    def model_drive(self):
        """Use trained models to drive the car"""
        # Initial movement sequence
        if self.state.getDistRaced() < 5:
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
            self.control.setSteer(0.0)
            self.control.setGear(1)
            return self.control.toMsg()
        
        # Additional check for very low speeds
        if self.state.getSpeedX() < 2.0:
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
            self.control.setGear(1)
            return self.control.toMsg()
        
        try:
            # Extract features
            features = self.extract_features()
            
            # Get predictions
            predictions = self.model_manager.predict(features)
            
            if predictions:
                # Apply predictions with safety checks
                steer_value = np.clip(predictions['steer'], -1.0, 1.0)
                self.control.setSteer(steer_value)
                
                # Gear control with safety
                gear_value = int(round(predictions['gear']))
                gear_value = max(-1, min(6, gear_value))
                
                # Additional gear safety
                rpm = self.state.getRpm()
                speed = self.state.getSpeedX()
                
                if speed < 3.0:
                    gear_value = 1
                elif rpm < 1000 and gear_value > 1:
                    gear_value = 1
                elif rpm > 7000 and gear_value < 6:
                    gear_value += 1
                elif rpm < 3000 and gear_value > 1:
                    gear_value -= 1
                
                self.control.setGear(gear_value)
                
                # Speed and acceleration control with safety
                accel_value = np.clip(predictions['accel'], 0.0, 1.0)
                
                # Speed control
                if speed > 95:  # Start braking at high speeds
                    accel_value = 0.0
                    self.control.setBrake(0.5)  # Apply moderate braking
                elif speed > 90:  # Reduce acceleration at high speeds
                    accel_value = min(accel_value, 0.3)
                    self.control.setBrake(0.0)
                elif speed < 5.0:  # Full acceleration at low speeds
                    accel_value = 1.0
                    self.control.setBrake(0.0)
                elif rpm < 2000 and gear_value == 1:  # High acceleration in first gear
                    accel_value = max(accel_value, 0.9)
                    self.control.setBrake(0.0)
                elif rpm > 6500:  # Reduce acceleration at high RPM
                    accel_value = min(accel_value, 0.8)
                    self.control.setBrake(0.0)
                else:
                    self.control.setBrake(0.0)
                
                # Apply acceleration
                self.set_throttle(accel_value)
                
                # Print status
                if self.state.getDistRaced() % 100 < 1:
                    print(f"Distance: {self.state.getDistRaced():.1f}, Speed: {self.state.getSpeedX():.1f}, " + 
                          f"Steering: {steer_value:.3f}, Accel: {accel_value:.2f}, Gear: {gear_value}, RPM: {rpm}")
            else:
                self.basic_drive()
                
        except Exception as e:
            print(f"Error during model prediction: {e}")
            self.basic_drive()
    
    def basic_drive(self):
        """Basic driving logic when models are not available"""
        # Initial movement
        if self.state.getSpeedX() < 2.0:
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
            self.control.setGear(1)
            return
        
        # Basic steering
        angle = self.state.angle
        dist = self.state.trackPos
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
        
        # Basic gear control
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm is None:
            up = True
        else:
            up = (self.prev_rpm - rpm) < 0
        
        if up and rpm > 6000:
            gear += 1
        elif not up and rpm < 4000:
            gear -= 1
        
        gear = max(-1, min(6, gear))
        self.control.setGear(gear)
        self.prev_rpm = rpm
        
        # Basic speed control with braking
        speed = self.state.getSpeedX()
        if speed > 95:  # Start braking at high speeds
            self.control.setAccel(0.0)
            self.control.setBrake(0.5)  # Apply moderate braking
        elif speed > 90:  # Reduce acceleration at high speeds
            self.control.setAccel(0.3)
            self.control.setBrake(0.0)
        elif speed < self.max_speed:
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
        else:
            self.control.setAccel(0.0)
            self.control.setBrake(0.0)
    
    def set_throttle(self, value):
        """Set throttle with brake/accel split"""
        if value <= 0:
            self.control.setBrake(-value)
            self.control.setAccel(0)
        else:
            self.control.setAccel(value)
            self.control.setBrake(0)
    
    def extract_features(self):
        """Extract features from current state"""
        try:
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
            
            # Add track sensors
            track_sensors = self.state.getTrack()
            if track_sensors:
                for i, sensor in enumerate(track_sensors):
                    feature_dict[f'track_{i}'] = sensor
            
            # Ensure all features are present in order
            feature_values = []
            for feature in self.features:
                if feature in feature_dict:
                    feature_values.append(feature_dict[feature])
                else:
                    feature_values.append(0.0)  # Default value for missing features
            
            # Verify feature count
            if len(feature_values) != len(self.features):
                print(f"[WARNING] Feature count mismatch in prediction. Expected {len(self.features)}, got {len(feature_values)}")
                print(f"Features: {self.features}")
                print(f"Values: {feature_values}")
            
            return feature_values
            
        except Exception as e:
            print(f"[ERROR] Failed to extract features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def onShutDown(self):
        """Clean up when shutting down"""
        self.collecting_data = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
    
    def onRestart(self):
        """Handle restart"""
        self.collecting_data = True
        self.start_data_collection()