# Detecting and Analyzing Chinook Wind Events in Calgary Using Unsupervised Machine Learning

**ATMS 523 - Module 8 Project**  
**Author:** Craig MacDonald - craigdm2

## Overview

I live in Calgary, Alberta. The winters here are long and cold - except when they very quickly aren't! It's not uncommon for Chinook winds to raise temperatures by 20 to 30C in the span of a few hours (https://en.wikipedia.org/wiki/Chinook_windLinks to an external site.). Warm winds blow in from the west over the mountains, interact with low pressure systems in the east, and the result is rapid increases in temperature and steep drops in air pressure and humidity. From a temperature standpoint, this is lovely. Less lovely is that they give me terrible migraines. This year has been particularly bad. I wanted to know whether Chinooks have become more or less likely occur to occur over time.

The catch is, there's no nicely labeled datasets of Chinook events. It's clear to tell when a Chinook is happening, you can just look outside and see the characteristic Chinook arch. But I wanted to look into the unlabeled data and try and identify the events. This seemed like a great opportunity to do unsupervised machine learning to see whether Chinooks can be identified in the hourly historic weather station data. With some timeseries data wrangling from Modules 2-3, machine learning strategies from Module 5 and feature engineering from Module 6, this seemed like an appropriate survey of the techniques I've learned in this class.

## Data

**Source:** Environment and Climate Change Canada Historical Weather Data  
**Station:** Calgary International Airport (Station ID: 3031092)  
**Period:** 1974–2024 (50 years)  
**Resolution:** Hourly observations  
**DOI:** [https://doi.org/10.17616/R3N012](https://doi.org/10.17616/R3N012)

### Variables Used

`temp` - Air temperature - C 
`rel_hum` - Relative humidity - % 
`wind_spd` - Wind speed - km/h 
`wind_dir` - Wind direction - tens of degrees (0–36) 
`pressure` - Station pressure - hPa 

### Engineered Features

`temp_delta_6h` - 6-hour temperature change - Looking for: Large positive (+10–20°C) 
`pressure_delta_6h` - 6-hour pressure change - Looking for: Negative (pressure drop) 
`humidity_delta_6h` - 6-hour humidity change - Looking for: Large negative (rapid drying) 
`wind_sin`, `wind_cos` - Wind components - Looking for: Westerly winds (sin ~ 0, cos < 0) 

## Methodology

### Cleaning and Fit for Use

I downloaded the data from ECCC weather stations using R's weathercan package (this step isn't included in the jupyter notebook). I did a check for rows with missing data (~7% of rows) and dropped them. The missing rows were relatively evenly distributed over time, so no worries that I was dropping too much data from a single time period I also checked for outliers (max, min) and the averages, and all seemed reasonable. The input data is the cleaned hourly dataset. 

### Two-Stage Unsupervised Learning Approach

For this project I used a two-stage unsupervised learning approach to isolate chinooks. First, I used an isolation forest methodology with a 2% anomaly threshold ('contamination' in sklearn's language) to pick out anomalous hours during winter months (Oct-Mar). On it's own this wasn't enough to identify chinooks, and the characteristics from the anomalies didn't appear to map clearly to chinook events. It was clear that not all anomalies were chinooks.

Next I used k-means clustering with k=3 on the isolation forest anomalies to try and differentiate between Chinooks and other events. This worked very well! Looking at the characteristics of the centroids for each event, there was a clear differentiation between chinooks and other events (roughly northerly cold air events, and winter storms/blizzards).

### Trend Determination

Next I classified any day with >=3 or more hours which were tagged to the chinook cluster as a chinook day. I used linear regression and then a non-parametric Mann-Kendall test to confirm whther there was a significant trend. 

## Results

### Isolated Forest

temp_delta_6h
    -Anomaly: +3.64
pressure_delta_6h
    -Anomaly: +0.09
humidity_delta_6h
    -Anomaly: -10.29
rel_hum
    -Anomaly: +51.43
wind_spd
    -Anomaly: +28.19
wind_dir
    -Anomaly: 321.9°

This clearly isn't picking out a chinkook. Moderate warming, increased pressure (we're looking for a decrease), a slight drop in humidity but relatively high average humidity.  Moderate northerly winds. 

### Detected Anomaly Sub-Clusters

| Sub-cluster | Events | Temp Δ | Humidity | Wind Dir | Interpretation |
|-------------|--------|--------|----------|----------|----------------|
| 0 (Chinook) | 2,330 hrs | +13.0C | 31% RH | 244 deg (WSW) | Chinook winds |
| 1 (Cold snap) | 905 hrs | -10.4°C | 85% RH | 19 deg (N) | Cold air outbreaks |
| 2 (Winter storms) | 872 hrs | -6.8°C | 71% RH | 344 deg (NNW) | Winter storms |

This seems like it's picking out a chinook more clearly!

### Chinook Climatology

- **Average:** 6.9 Chinook days per year
- **Range:** 2–13 days per year
- **Peak months:** October (28%) and March (23%)
- **Shoulder months (Oct–Nov, Feb–Mar):** 77.5% of chinooks in these months rather than dec-jan

### Trend Analysis

| Test | Statistic | P-value | Result |
|------|-----------|---------|--------|
| Linear Regression | slope = -0.036 days/year | 0.23 | No significant trend |
| Mann-Kendall | tau = -0.13 | 0.20 | No significant trend |

**Conclusion:** No statistically significant trend in Chinook frequency over the 50-year record, though there's high year-to-year variability.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

### Configuration

Key parameters that I adjusted, which are placed at the top of the script:

```python
CONTAMINATION = 0.02           # Isolation Forest: proportion of anomalies
CHINOOK_MIN_HOURS_PER_DAY = 3  # Minimum hours to count as a "Chinook day"
CHINOOK_MONTHS = [10, 11, 12, 1, 2, 3]  # Winter season definition
N_SUBCLUSTERS = 3              # Number of K-means sub-clusters
```

## Hypothesis Evaluation

-K-means identifies distinct Chinook cluster - YES (5/5 signature indicators)
-Two-stage approach improves over single-stage (new hyptohesis during the exercise) - YES (3/5 -> 5/5 score)
- ~10–20 Chinook days per winter - SOMEWHAT (6.9 days/year average) 
- Peak in shoulder months (Oct–Nov, Feb–Mar) - YES (77.5% of events)
- Increasing trend over time - NO (no significant trend)

## References

- Environment and Climate Change Canada. Historical Climate Data. DOI: [10.17616/R3N012](https://doi.org/10.17616/R3N012)
- Nkemdirim, L. C. (1996). Canada's chinook belt. *International Journal of Climatology*, 16(4), 441-462.
- Scikit Learn documentation (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

## License

This project is for academic purposes (ATMS 523 coursework).