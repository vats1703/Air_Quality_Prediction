### In this module, we define the Extract, Transform, Load (ETL) pipeline for the project. 
### We are going to use the OpenAQ API to extract the dat

import requests
import pandas as pd


def extract_data(city, country, parameter, date_from, date_to, limit = 1000):
    """
    Extracts data from the OpenAQ API.
    
    Parameters:"""

    url = f'https://api.openaq.org/v1/measurements'

    params = {
        'city': city,
        'country': country,
        'parameter': parameter,
        'date_from': date_from,
        'date_to': date_to,
        'limit': limit,
    }
    response = requests.get(url, params=params)
    data = response.json().get('results', [])
    data = data_to_df(data)
    return data

def data_to_df(data, drop_city_country = True):
    """
    Transforms the data extracted from the OpenAQ API into a pandas DataFrame.
    Use the drop_city_country parameter to drop the city and country columns from the DataFrame if we only
    want to keep a single city in the dataframe.
    """
    records = []
   
    if drop_city_country:
        for record in data:

            records.append({
                'location': record['location'],
                'date': record['date']['utc'],
                'parameter': record['parameter'],
                'value': record['value'],
                'unit': record['unit'],
            })
    else:
        for record in data:
            records.append({
                'location': record['location'],
                'city': record['city'],
                'country': record['country'],
                'date': record['date']['utc'],
                'parameter': record['parameter'],
                'value': record['value'],
                'unit': record['unit'],
            })
    
    df = pd.DataFrame(data)

    df['date'] = (
                    pd.to_datetime(df['date'])
                    .sort_values('date')
                    .reset_index(drop=True)   
                )
    return df

if __name__ == '__main__':
    data = extract_data('Machala', 'EC', 'pm25', '2020-01-01', '2020-01-02', limit=1000)
    
    data.to_csv('data_Machala.csv', index=False)
    print("Data extracted and saved to data.csv")


