import pandas as pd
import numpy as np

def generate_historical_ad_logs(n_samples=3000, seed=42):
    np.random.seed(seed)
    
    channels = ['ESPN', 'CNN', 'HBO', 'NBC', 'TNT']
    genres = ['sports', 'news', 'drama', 'sitcom', 'comedy']
    
    data = []
    for _ in range(n_samples):
        channel = np.random.choice(channels)
        genre = np.random.choice(genres)
        timeslot_hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        declared_duration = float(np.random.choice([30, 60, 90, 120, 180]))
        
        genre_factor = 1.10 if genre == 'news' else (0.92 if genre == 'sports' else 1.0)
        time_factor = 1.05 if 18 <= timeslot_hour <= 22 else 0.98
        
        actual_duration = declared_duration * genre_factor * time_factor + np.random.normal(0, 3)
        actual_duration = max(10.0, round(actual_duration, 2))
        
        delta = round(actual_duration - declared_duration, 2)
        
        data.append({
            'channel_id': channel,
            'genre': genre,
            'timeslot_hour': timeslot_hour,
            'day_of_week': day_of_week,
            'declared_duration': declared_duration,
            'actual_duration': actual_duration,
            'delta': delta
        })
        
    df = pd.DataFrame(data)
    df.to_csv('historical_ads.csv', index=False)
    print("✅ Created historical_ads.csv!")

if __name__ == "__main__":
    generate_historical_ad_logs()