
# Importing the required libraries:

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yahoo_fin as yhfin
import plotly.express as px
import datetime as dt
import requests
import matplotlib.pyplot as plt
import finnhub
import plotly.graph_objs as go
from datetime import timedelta
from bs4 import BeautifulSoup
from streamlit_option_menu import option_menu
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from dateutil.relativedelta import relativedelta
from yahoo_fin.stock_info import get_data
from yahoo_fin import stock_info as si
from pandas_datareader import DataReader
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# FundamentalData's key:

key = '5O2RB0NXONEEN12V'

# Finnhub key:

finnhub_client = finnhub.Client(api_key='ch2jk9hr01qs9g9uqcpgch2jk9hr01qs9g9uqcq0')

# Defining the app's features:

st.set_page_config(page_title = 'Stock Dashboard', page_icon = '📈')

hide_menu = ''' <style> #MainMenu {visibility:hidden;} footer {visibility:hidden;} </style>'''
st.markdown(hide_menu, unsafe_allow_html=True)

header = st.container()
content = st.container()
footer = st.container()

# Creating the alternative pages:

with st.sidebar:
    selected = option_menu(menu_title = None, options = ['Home', 'Stock Ideas', 'Summary', 'Daily Prices', 'Monthly Returns', 'Financial Statements', 'Analysts Recommendations', 'News', 'Predictions'], default_index=0)

# Creating the sidebar:

ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Set default values:

if ticker == '':
    ticker = 'AAPL'

else:
    pass

if start_date == end_date: 
    start_date = datetime.now() - relativedelta(years=10)
    end_date = datetime.now()

else:
    pass

# Footer function:

def write_footer(ticker):
    st.markdown('')
    st.markdown('------------------------------------------')
    st.markdown('')
    st.markdown(f'**Companies similar to {ticker}:**')
    st.markdown('')
    st.markdown(finnhub_client.company_peers(ticker))
    st.markdown('------------------------------------------')
    st.markdown('*Application created for educational purposes only, by Arthur Petry and Corentin Delcourt.*')

# Defining the page 'Home':

if selected == 'Home':

    # Creating the header:

    with header:

        st.title('Stock Dashboard')
        st.header('HEC Liège - Data Management Course Project')
        st.markdown('')

    # Content of the page:

    with content:

        st.markdown('''
        The aim of our project is to provide help to individuals with in order to solve
        the dynamic question: **"Should I invest in this stock or rather sell it?"**
        
        Our app provides many tools giving information and analyses on specific stocks, depending on your input. 
        You can navigate through the other pages of the app after inputing the ticker (name) of the asset and the desired start and end dates.
        Under every page in our app, you will also be able to see a list of 10 similar stocks if you wanna compare them.
        
        On "Stock Ideas", you can find trending stock to inspire yourself. You will see the Top 100 daily gainers (the stocks with the higest daily return).
        You will see the Top 100 daily gainers (It is possible it is not displayed as the API seems not to works efficiently everytime with this).
        There is also the Top 100 more traded today and Top 100 undervalued according to Yahoo finance.
        
        On the page "Summary", you can find the 12 basic stock indicators. Those are there to help you understand the stocks without getting involved with more complicated information.
        
        Under "Daily Prices", you can firstly see a dynamic graph where you can zoom on a targeted zone. It helps seeing a more precise time zone.
        Under that graph you can observe a dataframe with the historical data if needed. Next to that, we computed 3 differents return metrics to help you interpret the data.
        
        Regarding "Monthly Returns", it is exactly the same principle as for "Daily Prices", differing in the time frame.
        
        Concerning "Financials Statements", you are able to see 2 differents graphs and 3 dataframes. The first graph is showing you the actual earnings per share compared to the ones expected by analysts.
        It provides you a sort of feeling toward the stocks. For example, if it beats every quarters the expectations, It is a preety good indicator. That is why the second graph plots the surprise of the actual vs expectation.
        It gives you an idea in "%" of that difference. The dataframes represent respectively the balance sheet, the income statement and the cash flow statement.
        Those data are there for the ones willing to analyze more deeply the numbers.
        
        For "Analysts Recommandations", you can find a graph showing you the rating given by a number of analysts for the ticker you entered. 
        Again, it is supposed to give you a tendency of what people expect from this stock. We also display the mean recommandation and the mean targeted price by those analysts.
        
        As for the "News" page, you find the 20 latests news regarding the ticker you entered. You see the headline, the date, a summary of the news and the url to go directly to it. 
        News are a really helpful tool to analyse the future of a stock. If you know that the stock is having a important board meeting tomorrow, 
        there is a lot of chance that the volatility will be quite high in the next day for bad or for good depending on what comes out of it.
        
        For the prediction part, we decided to explore 2 differents ways to "predict" the future price of the stock. The first method is the simplest one. 
        It is an exponential moving average rolling over periodically. It is based on the last 60 days prices and will return prices for the next following month. 
        The goal is to compare graphically the difference between this method and the more complex one which is a model analyzing the last year data. 
        It trains the data and then test them to give you an expected future price. This time series modelling approach is supposed to do a better job predicting than the EMA. 
        Complementary we should say that stock markets are unpredictable and our models cannot be used with certainty. Their goal is more to give you a tendency of what is expected by those models for the future.
        ''')

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Stock Ideas':

if selected == 'Stock Ideas':

    # Creating the header:

    with header:

        st.title('Stock Ideas')
        st.markdown('')

    # Content of the page:

    with content:

        # Get and display the titles and dataframes in streamlit (handling possible errors):
        
        try:
            df_winners = si.get_day_gainers()
            dfw_clean = df_winners.drop(['Avg Vol (3 month)', 'PE Ratio (TTM)', 'Change'], axis=1)
            st.subheader('Top 100 stocks with higher profitability today')
            st.dataframe(dfw_clean)
            st.markdown('')
        
        except:
            st.markdown('**:red[Not able to display "Top 100 stocks with higher profitability today" due to API unavailability]**')
            st.markdown('')
        
        try:
            df_losers = si.get_day_losers()
            dfl_clean = df_losers.drop(['Avg Vol (3 month)', 'PE Ratio (TTM)', 'Change'], axis=1)
            st.subheader('Top 100 stocks with higher losses today')
            st.dataframe(dfl_clean)
            st.markdown('')
        
        except:
            st.markdown('**:red[Not able to display "Top 100 stocks with higher losses today" due to API unavailability]**')
            st.markdown('')
        
        try:
            df_active = si.get_day_most_active()
            dfa_clean = df_active.drop(['Market Cap', 'PE Ratio (TTM)', 'Change'], axis=1)
            st.subheader('Top 100 stocks with higher trading volume today')
            st.dataframe(dfa_clean)
            st.markdown('')
        
        except:
            st.markdown('**:red[Not able to display "Top 100 stocks with higher trading volume today" due to API unavailability]**')
            st.markdown('')
        
        try:
            df_value = si.get_undervalued_large_caps()
            dfv_clean = df_value.drop(['Avg Vol (3 month)', 'PE Ratio (TTM)', 'Change', '52 Week Range'], axis=1)
            st.subheader('Top 100 undervalued large cap stocks')
            st.dataframe(dfv_clean)
            st.markdown('')
        
        except:
            st.markdown('**:red[Not able to display "Top 100 undervalued large cap stocks" due to API unavailability]**')
            st.markdown('')

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Summary':

if selected == 'Summary':

    # Creating the header:

    with header:

        st.title(f'Summary for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        # Webscarping the data:

        url = f'https://finance.yahoo.com/quote/{ticker}'

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        left_table_rows = soup.find('div', {'data-test': 'left-summary-table'}).find_all('tr')

        right_table_rows = soup.find('div', {'data-test': 'right-summary-table'}).find_all('tr')

        # Create a dictionary to store the extracted data:

        data_summary = {}

        for row in left_table_rows:
            cols = row.find_all('td')
            data_summary[cols[0].text.strip()] = cols[1].text.strip()

        for row in right_table_rows:
            cols = row.find_all('td')
            data_summary[cols[0].text.strip()] = cols[1].text.strip()

        # Create and format a dataframe:

        df_summary = pd.DataFrame(data_summary, index=[0])
        df_summary = df_summary.transpose()
        df_summary = df_summary.rename(columns={0: 'Value'})

        # Display the dataframe:
    
        col1, col2, col3 = st.columns([1.5,1,1.5])
        
        for index, row in df_summary.iloc[0:4].iterrows():
               col1.metric(label=index, value=row['Value'])

        for index, row in df_summary.iloc[8:12].iterrows():
               col2.metric(label=index, value=row['Value'])
 
        for index, row in df_summary.iloc[4:8].iterrows():
               col3.metric(label=index, value=row['Value'])

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Daily Prices':

if selected == 'Daily Prices':

    # Creating the header:

    with header:

        st.title(f'Daily Prices for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        # Getting the data out of yahoo finance:

        df1 = si.get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval='1d')
        df2 = si.get_dividends(ticker, start_date=start_date, end_date=end_date, index_as_date=True)
        df = pd.concat([df1, df2], axis=1)

        # Manage the columns in the dataframe:

        df = df[['adjclose','dividend']]
        df['dividend'] = df['dividend'].fillna(0)
        df['% Change'] = df['adjclose'] / df['adjclose'].shift(1) - 1
        df = df.rename_axis('Date')
        df = df.rename(columns={'adjclose': 'Adjusted Close Price', 'dividend': 'Dividends', 'index':'Date'})

        # Saving a copy of the dataframe for later:

        dfb = df

        # Formatting the columns:

        df = df.rename(columns={'index': 'Date'})
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
        df.index = df.index.strftime('%d/%m/%Y')
        df['Adjusted Close Price'] = df['Adjusted Close Price'].round(2)
        df['Adjusted Close Price'] = df['Adjusted Close Price'].apply(lambda x: '${:.2f}'.format(x))
        df['Dividends'] = df['Dividends'].round(2)
        df['Dividends'] = df['Dividends'].apply(lambda x: '${:.2f}'.format(x))
        df['% Change'] = df['% Change'].apply(lambda x: '{:.2%}'.format(x))

        daily_df = df
        daily_df2 = daily_df.drop('Dividends', axis=1)

        # Displaying the graph:

        fig_daily = px.line(daily_df2, x=daily_df2.index, y='Adjusted Close Price', title=ticker)
        st.plotly_chart(fig_daily)
        st.markdown('------------------------------------------')
        st.markdown('')
        st.markdown('')

        space, dataframe_daily, space2, calc_daily = st.columns([1,7,1,4])

        # Displaying the dataframe:

        with dataframe_daily:
            st.dataframe(daily_df2)

        # Displaying the calculations:

        with calc_daily:
            annual_return = dfb['% Change'].mean()*252
            annual_returnp = '{:.2%}'.format(annual_return)
            st.metric('Annual Return is',annual_returnp)
            stdev = np.std(dfb['% Change'])*np.sqrt(252)
            stdev_round = round(stdev*100,2)
            st.metric('Standard Devation is',stdev_round)
            risk_adj_return = annual_return/stdev_round
            risk_adj_returnp = '{:.2%}'.format(risk_adj_return)
            st.metric('Risk Adjusted Return is',risk_adj_returnp)

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Monthly Returns':

if selected == 'Monthly Returns':

    # Creating the header:

    with header:

        st.title(f'Montlhy Returns for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        # Getting the data out of yahoo finance:

        df1 = si.get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval='1d')
        df2 = si.get_dividends(ticker, start_date=start_date, end_date=end_date, index_as_date=True)
        df = pd.concat([df1, df2], axis=1)

        # Manage the columns in the dataframe:

        df = df[['adjclose','dividend']]
        df['dividend'] = df['dividend'].fillna(0)
        df['% Change'] = df['adjclose'] / df['adjclose'].shift(1) - 1
        df = df.rename_axis('Date')
        df = df.rename(columns={'adjclose': 'Adj. Close Price', 'dividend': 'Dividends', 'index':'Date'})

        # Saving a copy of the dataframe for later:

        dfb = df

        # Adjusting the dataframe to show monthly data:

        dfb.index = pd.to_datetime(dfb.index, format='%d/%m/%Y')
        monthly_df = dfb.resample('M').last()
        monthly_df['Dividends'] = dfb['Dividends'].resample('M').sum()
        monthly_df = monthly_df.drop('% Change', axis=1)
        monthly_df.index = monthly_df.index.strftime('%m/%Y')

        # Calculate the total monthly return considering dividends:

        monthly_df['Monthly Adj. Total Return'] = (monthly_df['Adj. Close Price'] + \
                                                   monthly_df['Dividends']) / \
                                                   monthly_df['Adj. Close Price'].shift(1) - 1

        # Formatting the columns:

        monthly_df['Adj. Close Price'] = monthly_df['Adj. Close Price'].apply(lambda x: '${:.2f}'.format(x))
        monthly_df['Dividends'] = monthly_df['Dividends'].apply(lambda x: '${:.2f}'.format(x))
        monthly_df['Monthly Adj. Total Return'] = monthly_df['Monthly Adj. Total Return'].fillna(0)
        monthly_df['Monthly Adj. Total Return'] = monthly_df['Monthly Adj. Total Return'].apply(lambda x: '{:.2%}'.format(x))

        # Displaying the graph:

        fig_monthly = px.line(monthly_df, x=monthly_df.index, y='Monthly Adj. Total Return', title=ticker)
        st.plotly_chart(fig_monthly)
        st.markdown('------------------------------------------')
        st.markdown('')
        st.markdown('')

        space, dataframe_daily, space2, calc_daily = st.columns([1,9,3,6])

        # Displaying the dataframe:

        with dataframe_daily:
            st.dataframe(monthly_df)

        # Displaying the calculations:

        with calc_daily:
            annual_return = dfb['% Change'].mean()*252
            annual_returnp = '{:.2%}'.format(annual_return)
            st.metric('Annual Return is',annual_returnp)
            stdev = np.std(dfb['% Change'])*np.sqrt(252)
            stdev_round = round(stdev*100,2)
            st.metric('Standard Devation is',stdev_round)
            risk_adj_return = annual_return/stdev_round
            risk_adj_returnp = '{:.2%}'.format(risk_adj_return)
            st.metric('Risk Adjusted Return is',risk_adj_returnp)

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Financial Statements':

if selected == 'Financial Statements':

    # Creating the header:

    with header:

        st.title(f'Financial Statements for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        # Data for the plot:

        earnings_data = finnhub_client.company_earnings(ticker, limit=5) # Call the method to get the data
        quarters = [round(x['quarter'],2) for x in earnings_data] # Iterate over the data to extract the 'quarter' property
        actual_eps = [round(x['actual'],2) for x in earnings_data]
        estimated_eps = [round(x['estimate'],2) for x in earnings_data]
        surprise_percent = [round(x['surprisePercent'],2) for x in earnings_data]

        # Create the figure for the first plot:

        fig_actual_eps = go.Figure()

        sorted_data = sorted(zip(quarters, estimated_eps), key=lambda x: x[0])
        sorted_quarters, sorted_estimated_eps = zip(*sorted_data)

        fig_actual_eps.add_trace(go.Bar(x=quarters, y=actual_eps, width=0.4, name='Actual EPS', marker_color='#006837', text=[f"{x}%" for x in actual_eps], textposition='auto', textfont=dict(color='#DCD6D0')))
        fig_actual_eps.add_trace(go.Scatter(x=sorted_quarters, y=sorted_estimated_eps, mode='lines', name='Estimated EPS', line=dict(color='#b1de71', width=2), text=[f'{x}%' for x in sorted_estimated_eps], textposition='top center', textfont=dict(color='#DCD6D0')))

        fig_actual_eps.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), xaxis_title='Quarter', yaxis_title='Earnings Per Share (EPS)', title=f'Actual Earnings Per Share for {ticker}', legend=dict(y=-0.3, yanchor='top', x=0.5, xanchor='center'))
                
        # Create the figure for the second plot:

        mod_surprise_percent = [abs(sp*20) for sp in surprise_percent]
                
        fig_surprise_percent = go.Figure()

        fig_surprise_percent.add_trace(go.Scatter(x=quarters, y=surprise_percent, mode='markers', marker=dict(size=mod_surprise_percent, opacity=1.0, color='#006837'), name='Surprise Percentage'))

        for i in range(len(quarters)):
            fig_surprise_percent.add_annotation(x=quarters[i], y=surprise_percent[i], text=f'{surprise_percent[i]}%', showarrow=False, font=dict(size=16, color='#DCD6D0'))

        fig_surprise_percent.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), xaxis_title='Quarter', yaxis_title='Surprise Percentage (%)', title=f'Surprise Percentage for {ticker}', showlegend=True, legend=dict(y=-0.3, yanchor='top', x=0.5, xanchor='center'))
        
        # Display the charts using st.plotly_chart():

        st.plotly_chart(fig_actual_eps, use_container_width=True, config={'displayModeBar': False}, name='Actual EPS Chart')
        st.plotly_chart(fig_surprise_percent, use_container_width=True, config={'displayModeBar': False}, name='Surprise Percentage Chart')

        # Getting the data out of alpha vantage:

        fd = FundamentalData(key)

        # Balance sheet:
   
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]

        # Formatting the balance sheet:

        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        index_data = ['Total Assets', 'Total Current Assets',
                    'Cash and Cash Equivalents at Carrying Value', 'Cash and Short-Term Investments',
                    'Inventory', 'Current Net Receivables', 'Total Non-Current Assets',
                    'Property, Plant, Equipment', 'Accumulated Depreciation Amortization PPE',
                    'Intangible Assets', 'Intangible Assets Excluding Goodwill', 'Goodwill',
                    'Investments', 'Long-Term Investments', 'Short-Term Investments',
                    'Other Current Assets', 'Other Non-Current Assets', 'Total Liabilities',
                    'Total Current Liabilities', 'Current Accounts Payable', 'Deferred Revenue',
                    'Current Debt', 'Short-Term Debt', 'Total Non-Current Liabilities',
                    'Capital Lease Obligations', 'Long-Term Debt', 'Current Long-Term Debt',
                    'Long-Term Debt Non-Current', 'Short/Long-Term Debt Total',
                    'Other Current Liabilities', 'Other Non-Current Liabilities',
                    'Total Shareholder Equity', 'Treasury Stock', 'Retained Earnings',
                    'Common Stock', 'Common Stock Shares Outstanding']

        bs.index = index_data

        new_cols = []
        for col in bs.columns:
            date = dt.datetime.strptime(col, '%Y-%m-%d')
            new_col = date.strftime('%m/%Y')
            new_cols.append(new_col)

        bs.columns = new_cols

        bs = bs.replace('None', np.nan)
        bs = bs.dropna()
        bs = bs.apply(pd.to_numeric, errors='coerce')
        bs = bs.dropna()
        bs = bs.applymap(lambda x: '{:,.2f}'.format(x/1000000).replace(',', 'x').replace('.', ',').replace('x', '.'))

        # Assigning the balance sheet to the page:

        st.subheader('Balance Sheet (in Millions)')
        st.dataframe(bs)
        st.markdown('')
        st.markdown('------------------------------------------')
        st.markdown('')

        # Income statement:
 
        income_statement = fd.get_income_statement_annual(ticker)[0]

        # Formatting the income statement:

        ist= income_statement.T[2:]
        ist.columns = list(income_statement.T.iloc[0])
        index_data = ['Gross Profit', 'Total Revenue', 'Cost of Revenue',
                    'Cost of Goods and Services Sold', 'Operating Income',
                    'Selling, General and Administrative Expenses', 'Research and Development Expenses',
                    'Operating Expenses', 'Investment Income, Net', 'Net Interest Income',
                    'Interest Income', 'Interest Expense', 'Non-Interest Income',
                    'Other Non-Operating Income', 'Depreciation',
                    'Depreciation and Amortization', 'Income Before Tax', 'Income Tax Expense',
                    'Interest and Debt Expense', 'Net Income from Continuing Operations',
                    'Comprehensive Income, Net of Tax', 'Earnings Before Interest and Taxes (EBIT)',
                    'Earnings Before Interest, Taxes, Depreciation and Amortization (EBITDA)', 'Net Income']

        ist.index = index_data

        new_cols = []
        for col in ist.columns:
            date = dt.datetime.strptime(col, '%Y-%m-%d')
            new_col = date.strftime('%m/%Y')
            new_cols.append(new_col)

        ist.columns = new_cols

        ist = ist.replace('None', np.nan)
        ist = ist.dropna()
        ist = ist.apply(pd.to_numeric, errors='coerce')
        ist = ist.dropna() # drop rows with NaN values
        ist = ist.applymap(lambda x: '{:,.2f}'.format(x/1000000).replace(',', 'x').replace('.', ',').replace('x', '.'))

        # Assigning the income statement to the page:

        st.subheader('Income Statement (in Millions)')
        st.dataframe(ist)
        st.markdown('')
        st.markdown('------------------------------------------')
        st.markdown('')

        # Cash flow:

        cash_flow = fd.get_cash_flow_annual(ticker)[0]

        # Formatting the cash flow statement:

        cf = cash_flow.T[2:]
        cf.columns = list(cash_flow.T.iloc[0])
        index_data = ['Operating Cash Flow', 'Payments for Operating Activities', 'Proceeds from Operating Activities', 
                    'Change in Operating Liabilities', 'Change in Operating Assets', 
                    'Depreciation, Depletion and Amortization', 'Capital Expenditures', 
                    'Change in Receivables', 'Change in Inventory', 'Profit/Loss', 
                    'Cash Flow from Investment', 'Cash Flow from Financing', 
                    'Proceeds from Repayments of Short Term Debt', 
                    'Payments for Repurchase of Common Stock', 'Payments for Repurchase of Equity', 
                    'Payments for Repurchase of Preferred Stock', 'Dividend Payout', 
                    'Dividend Payout Common Stock', 'Dividend Payout Preferred Stock', 
                    'Proceeds from Issuance of Common Stock', 
                    'Proceeds from Issuance of Long Term Debt and Capital Securities, Net', 
                    'Proceeds from Issuance of Preferred Stock', 'Proceeds from Repurchase of Equity', 
                    'Proceeds from Sale of Treasury Stock', 'Change in Cash and Cash Equivalents', 
                    'Change in Exchange Rate', 'Net Income']
        cf.index = index_data

        new_cols = []
        for col in cf.columns:
            date = dt.datetime.strptime(col, '%Y-%m-%d')
            new_col = date.strftime('%m/%Y')
            new_cols.append(new_col)

        cf.columns = new_cols

        cf = cf.replace("None", np.nan)
        cf = cf.dropna()
        cf = cf.apply(pd.to_numeric, errors='coerce')
        cf = cf.dropna() # drop rows with NaN values
        cf = cf.applymap(lambda x: '{:,.2f}'.format(x/1000000).replace(',', 'x').replace('.', ',').replace('x', '.'))

        # Assigning the cash flow statement to the page:

        st.subheader('Cash Flow Statement')
        st.dataframe(cf)

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Analysts Recommendations':

if selected == 'Analysts Recommendations':

    # Creating the header:

    with header:

        st.title(f'Analyst Recommendations for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        # Analyst recommandation Yahoo Finance:

        recommendations = []
        targetMeanPrices = []
        currentPrices = []

        lhs_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'
        rhs_url = '?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&' \
                  'modules=upgradeDowngradeHistory,recommendationTrend,' \
                  'financialData,earningsHistory,earningsTrend,industryTrend&' \
                  'corsDomain=finance.yahoo.com'

        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        url1 =  lhs_url + ticker + rhs_url
        r = requests.get(url1, headers=headers)

        if not r.ok:
            recommendation = 6

        try:
            result = r.json()['quoteSummary']['result'][0]
            recommendation =result['financialData']['recommendationMean']['fmt']
            targetMeanPrice = result['financialData']['targetMeanPrice']['fmt'].replace(',', '')
            currentPrice = result['financialData']['currentPrice']['fmt'].replace(',', '')

        except:
            recommendation = 6

        recommendations.append(recommendation)
        targetMeanPrices.append(targetMeanPrice)
        currentPrices.append(currentPrice)

        st.markdown ('**According to analysts :**')
        st.markdown ('{} has an average recommendation of: '.format(ticker), recommendation)
        st.markdown ('{} has a target mean price of: '.format(ticker), targetMeanPrice)
        st.markdown ('{} has a current price of: '.format(ticker), currentPrice)

        # Analyst Recommendation Finnhub:

        def plot_stock_data(stock_data):
            symbol = stock_data[0]['symbol']
            buy_values = [data['buy'] for data in stock_data]
            hold_values = [data['hold'] for data in stock_data]
            sell_values = [data['sell'] for data in stock_data]
            strong_buy_values = [data['strongBuy'] for data in stock_data]
            strong_sell_values = [data['strongSell'] for data in stock_data]

            latest_data = stock_data[0]
            labels = ['Buy', 'Hold', 'Sell', 'Strong Buy', 'Strong Sell']
            values = [latest_data['buy'], latest_data['hold'], latest_data['sell'], latest_data['strongBuy'], latest_data['strongSell']]
            explode = [0] * len(labels)  # Initialize explode values to 0.
            explode[values.index(max(values))] = 0.1  # Explode the highest value.

            fig2, ax = plt.subplots()
            ax.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title(f'Stock Ratings for {symbol} ({latest_data["period"]})')
            st.plotly_chart(fig2)

        plot_stock_data(finnhub_client.recommendation_trends(ticker)) 

        def create_stock_dataframe(stock_data):
            periods = [data['period'] for data in stock_data]
            buys = [data['buy'] for data in stock_data]
            holds = [data['hold'] for data in stock_data]
            sells = [data['sell'] for data in stock_data]
            strong_buys = [data['strongBuy'] for data in stock_data]
            strong_sells = [data['strongSell'] for data in stock_data]

            df = pd.DataFrame({'Period': periods,
                               'Buy': buys,
                               'Hold': holds,
                               'Sell': sells,
                               'Strong Buy': strong_buys,
                               'Strong Sell': strong_sells})

            df.set_index('Period', inplace=True)  # Set the Period column as the index.
            return df

        df = create_stock_dataframe(finnhub_client.recommendation_trends(ticker))
        st.write(df)

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'News':

if selected == 'News':

    # Creating the header:

    with header:

        st.title(f'News for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        # Calculate the date of one month ago from the current date:

        today = datetime.now()
        one_month_ago = today - relativedelta(days=15)

        # Format the date in the required format (i.e., YYYY-MM-DD):

        from_date = one_month_ago.strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')

        # Get the news from the last month only:

        news = finnhub_client.company_news(ticker, _from=from_date, to=to_date)

        # Display the news in the desired format:
        
        for n in news:
            st.header(n['headline'])
            st.markdown('')
            st.markdown(n['summary'])
            st.markdown('')
            st.markdown(f'**Source:** {n["source"]}')
            st.markdown('')
            st.markdown(f'**URL:** {n["url"]}')
            st.markdown('')
            st.markdown('------------------------------------------')
            st.markdown('')

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Predictions':

if selected == 'Predictions':

    # Creating the header:

    with header:

        st.title(f'Predictions for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        st.write('coming soon')

    # Creating the footer:

    with footer:

        write_footer(ticker)