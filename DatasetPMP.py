import numpy as np
import pandas as pd

def clip_eps(value, min_val, max_val):
    return np.clip(value, min_val, max_val)

num_rows = 50
num_files = 1000
product_names = ['Product1', 'Product2', 'Product3', 'Product4']
seasonal_factor = 10  # Adjust the effect of seasonality
min_demand_forecast = 5  # Minimum threshold for forecast and demand

for file_num in range(num_files):
    data = {}

    # Initialize forecast and demand for each product
    for product in product_names:
        data[product] = {
            'fcst': [20],
            'dmd': [20],
            'fcst_sd': 5,
            'dmd_sd': 5
        }

    # Generate data for each time step
    for i in range(1, num_rows):
        for product in product_names:
            alpha = 0.7  # Fixed decay factor for stability
            fcst_sd = data[product]['fcst_sd']
            dmd_sd = data[product]['dmd_sd']

            # Add a seasonal component using a sinusoidal function
            seasonality = seasonal_factor * np.sin(2 * np.pi * i / num_rows)

            # Add random noise with more variability
            fcst_eps = clip_eps(np.random.normal(loc=0, scale=np.sqrt(fcst_sd)), -10, 10)
            dmd_eps = clip_eps(np.random.normal(loc=0, scale=np.sqrt(dmd_sd)), -10, 10)

            # Update forecast and demand with seasonality and random noise
            new_fcst = max(min_demand_forecast, data[product]['dmd'][-1] + dmd_eps + seasonality)
            new_dmd = max(min_demand_forecast, data[product]['fcst'][-1] + fcst_eps)

            # Smooth out abrupt changes using a moving average with the previous value
            smoothed_fcst = 0.5 * data[product]['fcst'][-1] + 0.5 * new_fcst
            smoothed_dmd = 0.5 * data[product]['dmd'][-1] + 0.5 * new_dmd

            # Append smoothed values to lists
            data[product]['fcst'].append(smoothed_fcst)
            data[product]['dmd'].append(smoothed_dmd)

            # Update the standard deviation for forecast and demand
            data[product]['fcst_sd'] = (1 - alpha) * fcst_sd + alpha * (smoothed_fcst - data[product]['fcst'][-2])**2
            data[product]['dmd_sd'] = (1 - alpha) * dmd_sd + alpha * (smoothed_dmd - data[product]['dmd'][-2])**2

    # Create a DataFrame from the generated data
    df = pd.DataFrame()
    for product in product_names:
        df[f'{product}_Forecast'] = np.round(data[product]['fcst']).clip(min=min_demand_forecast)
        df[f'{product}_Demand'] = np.round(data[product]['dmd']).clip(min=min_demand_forecast)

    # Save the DataFrame as a CSV file
    df.to_csv(f'dataMP/dataset_{file_num}.csv', index=False)
