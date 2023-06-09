import datetime
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
import json

#%% static data
files = {'sources':'Data Sources.csv', 'final':'Data Cleaned.json'}

#%% classes and functions
#%%
class Variables:
    def __init__(self, df):
        self.dep = df[df['Dependent']=='Y']['Name'].tolist()[0] # only 1 dependent var!

        self.indep = df[df['Dependent']!='Y']['Name'].tolist()
        self.indep = [i for i in self.indep if 'business expectations' not in i.lower()]

        self.freq = dict(zip(df['Name'],df['Frequency']))

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

#%% yfinance
def get_yf(ticker, start_date='1900-01-01'):
    data = yf.download(ticker, start=start_date)
    close = dict(zip(data.index.date, data['Close']))
    return(close)

#%% load data sources and create dicts to store metadata of each source
requests_funcs = {'mas':get_mas, 'sgx':get_sgx, 'singstat':get_singstat, 'yf':get_yf}

sources = pd.read_csv(files['sources'], encoding='utf-8')
for k in requests_funcs:
    sources.loc[sources['URL'].str.contains(k)|sources['API'].str.contains(k), 'Source'] = k

variables = Variables(sources)

apis = {k:sources[sources['Source']==k][['Name','Frequency','API']].to_dict('records') for k in requests_funcs}

#%% pull data
ts_data = {}
for k,v in apis.items():
    for dim in v:
        ts_data[dim['Name']] = requests_funcs[k](dim['API'])

#%% M1 money supply from MAS API stops at 2021-06. download post-2021-06 data using requests.
# change date range if desired
post_data = {'__VIEWSTATE':None,'__VIEWSTATEGENERATOR':None,'__EVENTVALIDATION':None,
             'ctl00$ContentPlaceHolder1$StartYearDropDownList':'2021',
             'ctl00$ContentPlaceHolder1$StartMonthDropDownList':'1',
             'ctl00$ContentPlaceHolder1$EndYearDropDownList':'2023',
             'ctl00$ContentPlaceHolder1$EndMonthDropDownList':'12',
             'ctl00$ContentPlaceHolder1$FrequencyDropDownList':'M',
             'ctl00$ContentPlaceHolder1$DownloadButton':'Download',
             'ctl00$ContentPlaceHolder1$OptionsList$3':'on'} # M1

url = 'https://eservices.mas.gov.sg/statistics/msb-xml/Report.aspx?tableSetID=I&tableID=I.1'
with requests.Session() as session:
    response = session.get(url)
    headers = {'Cookie':' '.join(f'{k}={v} ' for k, v in response.cookies.get_dict().items())}
    bs_obj = BeautifulSoup(response.content, 'html.parser')
    for k,v in post_data.items():
        if v==None:
            post_data[k] = bs_obj.find(id=k)['value']

    response = session.post(url=url, data=post_data, headers=headers)

add_m1_money = [i.split(',') for i in response.text.split('\n\n')[0].split('\n')][2:]
add_m1_money = dict(zip([i[0] for i in add_m1_money], [i[1] for i in add_m1_money]))
add_m1_money = {pd.to_datetime(k).strftime('%Y-%m'):float(v) for k,v in add_m1_money.items()}

ts_data['M1 Money Supply'].update(add_m1_money)

#%% change key datatypes to string
for k in ts_data:
    ts_data[k] = {str(v):dim for v,dim in ts_data[k].items()}

#%% export to json
with open(files['final'], 'w', encoding='utf-8') as f:
    f.write(json.dumps(ts_data, ensure_ascii=False, indent=4, sort_keys=True, default=str))