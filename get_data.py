import datetime
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

#%% static data
files = {'sources':'Data Sources.csv', 'final':'Data Cleaned.csv'}

#%% classes and functions
#%%
class Variables:
    def __init__(self, df):
        self.dep = df[df['Dependent']=='Y']['Name'].tolist()

        self.indep = df[df['Dependent']!='Y']['Name'].tolist()
        self.indep = [i for i in self.indep if 'business expectations' not in i.lower()]

        self.freq = dict(zip(df['Name'],df['Frequency']))

        self.non_stat = []

#%% singapore department of statistics
def get_singstat(url):
    response = requests.get(url=url).json()
    return({i['Key']:float(i['Value']) for i in response})

#%% singapore exchange
def get_sgx(url):
    data = {}
    response = requests.get(url=url).json()
    for d in response['data']:
        date = d['trading_time'].split('_')[0]
        date = datetime.datetime.strptime(date, '%Y%m%d').date()
        data[date] = float(d['lp'])
    return(data)

#%% monetary authority of singapore
def get_mas(url, params={'limit':100,'offset':0}):
    response = requests.get(url=url, params=params).json()

    freq = [i for i in response['result']['records'][0].keys() if i.startswith('end_of')][0]
    series_name = [i for i in response['result']['records'][0].keys() if not i in [freq,'timestamp']][0]
    no_records = int(response['result']['total'])
    pagesize = int(response['result']['limit'])
    pages = no_records//pagesize + (no_records%pagesize>0)

    def process_records(data_collect, response_obj):
        for i in response_obj:
            data_collect[i[freq]] = float(i[series_name]) if i[series_name] is not None else np.nan

    # parse first page
    data = {}
    process_records(data, response['result']['records'])

    # parse rest of data to get complete series
    for i in range(1,pages):
        params['offset'] = i*100
        response = requests.get(url=url, params=params).json()
        process_records(data, response['result']['records'])

    return(data)

#%% load data sources and create dicts to store metadata of each source
sources = pd.read_csv(files['sources'], encoding='utf-8')
variables = Variables(sources)

requests_funcs = {'mas':get_mas, 'sgx':get_sgx, 'singstat':get_singstat}
apis = {'mas':[], 'sgx':[], 'singstat':[]}
for k in apis.keys():
    for i in zip(sources['Name'], sources['Frequency'], sources['API']):
        if k in i[-1]:
            apis[k].append({'Name':i[0], 'Frequency':i[1], 'API':i[2]})

#%% pull data
ts_data = {}
for k,v in apis.items():
    for dim in v:
        ts_data[dim['Name']] = requests_funcs[k](dim['API'])

#%% pass data into pandas series
ts_pd = {}
for k,v in apis.items():
    for dim in v:

        if dim['Frequency']=='Q':
            periods = [p.split()[0]+p.split()[-1][::-1] for p in list(ts_data[dim['Name']])]
            periods = pd.PeriodIndex(periods, freq=dim['Frequency'])
            ts_pd[dim['Name']] = pd.Series(ts_data[dim['Name']].values(), index=periods)

        elif dim['Frequency']=='M':
            periods = pd.to_datetime(list(ts_data[dim['Name']])) + pd.tseries.offsets.MonthEnd(0)
            ts_pd[dim['Name']] = pd.Series(ts_data[dim['Name']].values(), index=periods)

        elif dim['Frequency']=='D':
            periods = pd.to_datetime(list(ts_data[dim['Name']]))
            ts_pd[dim['Name']] = pd.Series(ts_data[dim['Name']].values(), index=periods).resample('M').last()
            variables.freq[dim['Name']] = 'M' # update to 'M' since resampled

        # start all series from first valid index
        ts_pd[dim['Name']] = ts_pd[dim['Name']][ts_pd[dim['Name']].first_valid_index():]

#%% check and ensure series' stationarity
# if p-value >0.05, variable is non-stationary
for i in variables.indep:
    print(i)
    # if unit root, take % yoy growth (which also removes seasonality)
    if adfuller(ts_pd[i])[1]>0.05:

        if variables.freq[i]=='M':
            ts_pd[i] = ts_pd[i].pct_change(periods=12) * 100
        elif variables.freq[i]=='Q':
            ts_pd[i] = ts_pd[i].pct_change(periods=4) * 100
        print('Non-stationary', end='\n\n')
        variables.non_stat.append(i)

    else:
        print('Stationary', end='\n\n')

#%% resample all series and pass into dataframe
'''
order adhered to as defined by statsmodels docs for dynamic factor modelling:
[https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html]
- dependent variable leftmost
- monthly data in the first columns
- quarterly data in the last columns
'''
# resample all series to monthly and start from first valid index
for series, freq in zip(sources['Name'], sources['Frequency']):

    if freq=='Q':
        ts_pd[series] = ts_pd[series].resample('M', convention='end').asfreq()
        ts_pd[series].index = pd.to_datetime(ts_pd[series].index.strftime('%Y-%m-%d'))

    if ts_pd[series].index[0]!=ts_pd[series].first_valid_index():
        ts_pd[series] = ts_pd[series][ts_pd[series].first_valid_index():]

# pass all series into dataframe, start dataframe from first year of GDP growth data
ts_df = pd.DataFrame(ts_pd)
ts_df = ts_df.loc[ts_df.index.year>=ts_df.loc[:,variables.dep[0]].first_valid_index().year]

# rearrange columns in correct order for factor modelling as explained in markdown above
ts_df = ts_df[variables.dep+\
              [k for k,v in variables.freq.items() if v=='M' and k!=variables.dep[0]]+\
              [k for k,v in variables.freq.items() if v=='Q' and k!=variables.dep[0]]]

ts_df.index.name = 'Period'
ts_df = ts_df.reset_index()
ts_df['Period'] = ts_df['Period'].dt.date

#%% export
ts_df.to_csv(files['final'], encoding='utf-8', index=False)