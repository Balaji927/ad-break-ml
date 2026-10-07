import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

def train_model():
    df = pd.read_csv('historical_ads.csv')
    
    stats = df.groupby(['channel_id', 'timeslot_hour', 'genre'])['delta'].agg(
        hist_mean_deviation='mean',
        hist_std_deviation='std'
    ).reset_index()
    
    stats['hist_std_deviation'] = stats['hist_std_deviation'].fillna(4.0)
    df = pd.merge(df, stats, on=['channel_id', 'timeslot_hour', 'genre'], how='left')
    
    categorical_cols = ['channel_id', 'genre']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        
    features = [
        'declared_duration', 
        'timeslot_hour', 
        'day_of_week', 
        'channel_id', 
        'genre', 
        'hist_mean_deviation', 
        'hist_std_deviation'
    ]
    
    X = df[features]
    y = df['delta']
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X, y, categorical_feature=categorical_cols)
    
    joblib.dump(model, 'lightgbm_model.pkl')
    stats.to_csv('historical_stats.csv', index=False)
    print("✅ LightGBM model trained and saved to lightgbm_model.pkl!")

if __name__ == "__main__":
    train_model()