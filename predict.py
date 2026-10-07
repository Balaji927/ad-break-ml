import joblib
import pandas as pd
import numpy as np

class PreBreakPredictor:
    def __init__(self, model_path='lightgbm_model.pkl', stats_path='historical_stats.csv'):
        self.model = joblib.load(model_path)
        self.stats_df = pd.read_csv(stats_path)
        
    def predict_cue(self, raw_cue: dict) -> dict:
        cue_df = pd.DataFrame([raw_cue])
        
        merged = pd.merge(
            cue_df, 
            self.stats_df, 
            on=['channel_id', 'timeslot_hour', 'genre'], 
            how='left'
        )
        
        merged['hist_mean_deviation'] = merged['hist_mean_deviation'].fillna(0.0)
        merged['hist_std_deviation'] = merged['hist_std_deviation'].fillna(4.0)
        
        for col in ['channel_id', 'genre']:
            merged[col] = merged[col].astype('category')
            
        features = [
            'declared_duration', 
            'timeslot_hour', 
            'day_of_week', 
            'channel_id', 
            'genre', 
            'hist_mean_deviation', 
            'hist_std_deviation'
        ]
        
        predicted_delta = float(self.model.predict(merged[features])[0])
        declared_dur = float(raw_cue['declared_duration'])
        adjusted_duration = max(5.0, declared_dur + predicted_delta)
        
        hist_std = float(merged['hist_std_deviation'].iloc[0])
        confidence_score = float(np.exp(-0.08 * hist_std))
        
        return {
            "declared_duration": declared_dur,
            "predicted_delta": round(predicted_delta, 2),
            "adjusted_duration": round(adjusted_duration, 2),
            "confidence_score": round(confidence_score, 4)
        }

if __name__ == "__main__":
    predictor = PreBreakPredictor()
    sample_cue = {
        'declared_duration': 60.0,
        'channel_id': 'CNN',
        'timeslot_hour': 20,
        'day_of_week': 2,
        'genre': 'news'
    }
    print("🎯 Output Test:", predictor.predict_cue(sample_cue))