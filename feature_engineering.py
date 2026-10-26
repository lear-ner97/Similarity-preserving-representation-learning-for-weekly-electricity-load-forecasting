# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:27:35 2026

@author: umroot
"""
import pandas as pd
import numpy as np
import holidays




def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'
    

def missing_temperature_imputation(weather_data,site_id,time_range):
    weather_subset=weather_data[weather_data['site_id']==site_id]
    weather_subset['timestamp']=pd.to_datetime(weather_subset['timestamp'])
    missing_hours = time_range.difference(pd.to_datetime(weather_subset['timestamp'])) 
    diffs = missing_hours.to_series().diff()
    groups = (diffs != pd.Timedelta(hours=1)).cumsum()
    streak_lengths = missing_hours.to_series().groupby(groups).size()
    max_consecutive_missing = streak_lengths.max()
    weather_subset = weather_subset.set_index('timestamp')
    weather_subset = weather_subset.reindex(time_range)
    if max_consecutive_missing<=5:
        #consecutive gaps, so we apply time interpolation as temp data is highly autocorrelated hourly, 684 for moose with long
        airTemp = weather_subset['airTemperature'].interpolate(method='time')
    else:
        # create season column from datetime index
        weather_subset['season'] = weather_subset.index.month.map(get_season)
        # fill missing temperatures using seasonal hourly means
        airTemp = weather_subset['airTemperature'].fillna(weather_subset.groupby(['season',
                weather_subset.index.hour])['airTemperature'].transform('mean'))
    return airTemp

def calendar_features(full_range,src_site_id,tgt_site_id):
    src_df = pd.DataFrame({'timestamp': full_range})
    tgt_df = pd.DataFrame({'timestamp': full_range})
    # weekend 
    src_df['weekend'] = src_df['timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    time_series_weekend=np.array(src_df['weekend'])
    # Cyclical encoding
    src_df['hour'] = src_df['timestamp'].dt.hour
    src_df['hour_sin'] = np.sin(2 * np.pi * src_df['hour'] / 24)
    src_df['hour_cos'] = np.cos(2 * np.pi * src_df['hour'] / 24)
    src_df['day'] = src_df['timestamp'].dt.dayofweek
    src_df['day_sin'] = np.sin(2 * np.pi * src_df['day'] / 7)
    src_df['day_cos'] = np.cos(2 * np.pi * src_df['day'] / 7)
    time_series_hour_sin=np.array(src_df['hour_sin'])
    time_series_hour_cos=np.array(src_df['hour_cos'])
    time_series_day=np.array(src_df['day'])
    time_series_day_sin=np.array(src_df['day_sin'])
    time_series_day_cos=np.array(src_df['day_cos'])
    # if src_site_id=='Hog':
    #     return time_series_weekend,time_series_hour_sin,time_series_hour_cos,time_series_day,time_series_day_sin,time_series_day_cos
    # #holiday column
    if src_site_id=='Robin': #london, uk
        uk_holidays = holidays.UK(years=[2016, 2017], subdiv='England')
        src_df['is_holiday'] = src_df['timestamp'].dt.date.isin(uk_holidays).astype(int)
    elif src_site_id=='Hog':
        us_holidays = holidays.US(years=[2016, 2017])
        src_df['is_holiday'] = src_df['timestamp'].dt.date.isin(us_holidays).astype(int)
    elif src_site_id=='Lamb': #cardiff, wales
        uk_holidays = holidays.UK(years=[2016, 2017], subdiv='WLS')
        src_df['is_holiday'] = src_df['timestamp'].dt.date.isin(uk_holidays).astype(int)  
    src_time_series_holiday=np.array(src_df['is_holiday'])

    if tgt_site_id=='Robin': #london, uk
        uk_holidays = holidays.UK(years=[2016, 2017], subdiv='England')
        tgt_df['is_holiday'] = tgt_df['timestamp'].dt.date.isin(uk_holidays).astype(int)
    elif tgt_site_id=='Hog': #US central
        us_holidays = holidays.US(years=[2016, 2017])
        tgt_df['is_holiday'] = tgt_df['timestamp'].dt.date.isin(us_holidays).astype(int)
    elif tgt_site_id=='Lamb': #cardiff, wales
        uk_holidays = holidays.UK(years=[2016, 2017], subdiv='WLS')
        tgt_df['is_holiday'] = tgt_df['timestamp'].dt.date.isin(uk_holidays).astype(int)  
    tgt_time_series_holiday=np.array(tgt_df['is_holiday'])
    return time_series_weekend,src_time_series_holiday,tgt_time_series_holiday,time_series_hour_sin,time_series_hour_cos,time_series_day,time_series_day_sin,time_series_day_cos


