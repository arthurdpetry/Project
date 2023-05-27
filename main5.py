
# Importing the required libraries:

import yfinance as yf
import datetime as dt
import finnhub
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import requests
import streamlit as st
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit_option_menu import option_menu
from xgboost import XGBClassifier
from yahoo_fin import stock_info as si
from alpha_vantage.fundamentaldata import FundamentalData
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta

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

    selected = option_menu(menu_title = None, options = ['Home', 'Stock Ideas', 'Summary',
                                                         'Daily Prices', 'Monthly Returns', 'Financial Statements',
                                                         'Analysts Recommendations', 'News', 'Predictions',
                                                         'Final Analysis'], default_index=0)

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
        The aim of our project is to provide help to individuals with answering
        the dynamic question: **"Should I invest in this stock, keep it or rather sell it?"**

        Our app provides many tools giving information and analyses on specific stocks, depending on your input. 
        You can navigate through the other pages of the app after inputing the ticker (name) of the stock and the desired start and end dates.
        Under every page in our app, you will also be able to see a list of 10 similar stocks if you wanna compare them to the one you entered.

        On "Stock Ideas", you can find trending stock to inspire yourself if you don't know what you are looking for yet. You will see the Top 100 daily gainers (the stocks with the higest daily return).
        You will see the Top 100 daily losers. There is also the Top 100 more traded today and Top 100 undervalued according to Yahoo finance.

        On the page "Summary", you can find the 15 basic stock indicators. Those are there to help you understand the stocks without getting involved with more complicated information.
        Information as the "52 weeks range" or the "EPS (earning per share)" can help you answering the question. We will develop on that page what each info can be used for.
        
        Under "Daily Prices", you can firstly see a dynamic graph where you can zoom on a targeted zone. It helps seeing a more precise time zone.
        Under that graph you can observe a dataframe with the historical data if needed. Next to that, we computed 3 differents return metrics to help you interpret the data.

        Regarding "Monthly Returns", it is exactly the same principle as for "Daily Prices", differing in the time frame.

        Concerning "Financials Statements", you are able to see 2 differents graphs and 3 dataframes. The first graph is showing you the actual earnings per share compared to the ones expected by analysts.
        It provides you a sort of feeling toward the stocks. For example, if it beats every quarters the expectations, it is a preety good indicator that this i a solid company. 
        That is why the second graph plots the surprise of the actual earnings vs expectation.
        It gives you an idea in "%" of that difference. The more positive and big the numer is, the best the surprise was. The dataframes represent respectively the balance sheet, the income statement and the cash flow statement.
        Those data are there for the ones willing to analyze more deeply the numbers.

        For "Analysts Recommandations", you can find a graph showing you the rating given by a number of analysts for the ticker you entered. 
        Again, it is supposed to give you a tendency of what expert people expect from this stock. We also display the mean recommandation, the analyst verdict and the mean targeted price by those analysts.
        The analyst verdict is calculated with the mean recommendation, and is based on this scale: Strong buy - from 1 to 1.5; Buy - from 1.5 to 2.5; Hold - from 2.5 and 3.5; Sell - from 3.5 to 4.5; Strong sell - from 4.5 to 5.

        As for the "News" page, you find the 20 latests news regarding the ticker you entered. You see the headline, the date, a summary of the news and the url to go directly to it. 
        News are a really helpful tool to analyse the future of a stock. If you know that the stock is having a important board meeting tomorrow, 
        there is a lot of chance that the volatility will be quite high in the next day for bad or for good depending on what comes out of it.

        For the prediction part, we decided to explore differents ways to "predict" the future price of the stock. The first method is the simplest one. 
        It is an exponential moving average, rolled over periodically. It is based on the last 60 days prices and will return prices for the next following month. 
        The goal of this part is to compare graphically the difference between this method and more complex models which analyse last year data. 
        It trains the data and then test them to give you an expected future price. Those modelling approaches are supposed to do a better job predicting than the EMA. 
        Complementary we should say that stock markets are unpredictable and our models cannot be used with certainty as they are not "really efficient" to predict the real future.
        If it was the case, everyone would be rich ;)  Their goal is more to give you a tendency of what is expected by those models for the future.

        A quotation we really like regarding the prediction of stock prices comes from the famous statistician George Box who said "All models are wrong but some are useful". We don't aim
        at being precise, just testing different approach to give tendency to users. 
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

        def convert_market_cap(value):

            if isinstance(value, str):
                suffixes = {'T': 1e12, 'B': 1e9}

                if value[-1] in suffixes:
                    factor = suffixes[value[-1]]
                    value = value[:-1]
                    value = int(float(value) * factor)
                else:
                    value = int(value.replace(',', ''))

                return '{:,}'.format(value)

            else:
                return '{:,}'.format(value)

        def format_numbers(value):

            if isinstance(value, (int, float)):
                return '{:,.0f}'.format(value)

            else:
                return value

        def convert_market_cap2(value, column_name):

            if isinstance(value, str):

                if 'B' in value:
                    factor = 1e9

                elif 'M' in value:
                    factor = 1e6

                elif 'K' in value:
                    factor = 1e3

                else:
                    factor = 1

                value = float(value.replace(',', '').replace('$', '').replace(column_name, '').replace('M','').replace('B','')) * factor

            if column_name in ['Market Cap']:
                return '{:,.0f}'.format(value)

            else:
                return '{:,.2f}'.format(value)
 
        def convert_volume2(value, column_name):

            if isinstance(value, str):

                if 'B' in value:
                    factor = 1e9

                elif 'M' in value:
                    factor = 1e6

                elif 'K' in value:
                    factor = 1e3

                else:
                    factor = 1

                value = float(value.replace(',', '').replace('$', '').replace(column_name, '').replace('M','').replace('B','')) * factor

            if column_name in ['Volume']:
                return '{:,.0f}'.format(value)

            else:
                return '{:,.2f}'.format(value)

        # Get and display the titles and dataframes in streamlit (handling possible errors):
        
        try:
            df_winners = si.get_day_gainers()
            dfw_clean = df_winners.drop(['Avg Vol (3 month)', 'PE Ratio (TTM)', 'Change'], axis=1)
            dfw_clean['% Change'] = dfw_clean['% Change'].apply(lambda x: '{:,}'.format(x))
            dfw_clean['Volume'] = dfw_clean['Volume'].apply(lambda x: '{:,.0f}'.format(x))
            dfw_clean[['Volume', 'Market Cap']] = dfw_clean[['Volume', 'Market Cap']].applymap(format_numbers)
            dfw_clean['Market Cap'] = dfw_clean['Market Cap'].apply(convert_market_cap)
            st.subheader('Top 100 stocks with higher profitability today')
            st.dataframe(dfw_clean)
            st.markdown('')
     
        except:
            st.markdown('**:red[Not able to display "Top 100 stocks with higher profitability today" due to API unavailability]**')
            st.markdown('')
     
        try:
            df_losers = si.get_day_losers()
            dfl_clean = df_losers.drop(['Avg Vol (3 month)', 'PE Ratio (TTM)', 'Change'], axis=1)
            dfl_clean['% Change'] = dfl_clean['% Change'].apply(lambda x: '{:,}'.format(x))
            dfl_clean['Volume'] = dfl_clean['Volume'].apply(lambda x: '{:,.0f}'.format(x))
            dfl_clean[['Volume', 'Market Cap']] = dfl_clean[['Volume', 'Market Cap']].applymap(format_numbers)
            dfl_clean['Market Cap'] = dfl_clean['Market Cap'].apply(convert_market_cap)
            st.subheader('Top 100 stocks with higher losses today')
            st.dataframe(dfl_clean)
            st.markdown('')
    
        except:
            st.markdown('**:red[Not able to display "Top 100 stocks with higher losses today" due to API unavailability]**')
            st.markdown('')
    
        try:
            df_active = si.get_day_most_active()
            dfa_clean = df_active.drop(['Market Cap', 'PE Ratio (TTM)', 'Change'], axis=1)
            dfa_clean['% Change'] = dfa_clean['% Change'].apply(lambda x: '{:,}'.format(x))
            dfa_clean['Volume'] = dfa_clean['Volume'].apply(lambda x: '{:,.0f}'.format(x))
            st.subheader('Top 100 stocks with higher trading volume today')
            st.dataframe(dfa_clean)
            st.markdown('')
  
        except:
            st.markdown('**:red[Not able to display "Top 100 stocks with higher trading volume today" due to API unavailability]**')
            st.markdown('')
    
        try:
            df_value = si.get_undervalued_large_caps()
            dfv_clean = df_value.drop(['Avg Vol (3 month)', 'PE Ratio (TTM)', 'Change', '% Change', '52 Week Range'], axis=1)
            dfv_clean['Market Cap'] = dfv_clean['Market Cap'].apply(convert_market_cap2, column_name='Market Cap')
            dfv_clean['Volume'] = dfv_clean['Volume'].apply(convert_volume2, column_name='Volume')
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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.127 Safari/537.36',}

        response = requests.get(url, headers=headers)
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

        col1, col2, col3, = st.columns([1,1,1])
    
        for index, row in df_summary.iloc[0:5].iterrows():
               col1.metric(label=index, value=row['Value'])

        for index, row in df_summary.iloc[10:15].iterrows():
               col2.metric(label=index, value=row['Value'])

        for index, row in df_summary.iloc[5:10].iterrows():
               col3.metric(label=index, value=row['Value'])

        # Explaining the different data:

        st.markdown('------------------------------------------')
        st.markdown('')
        st.markdown('''
        Those data above are useful, basic data regarding a stock. They are most of the time very helpful to decide what to do with the stock.
        - Previous close, Open, Bid, Ask are data related to today's trading. 
        - Day's range helps you seing where you situate yourself at the moment you want to buy. It is mostly useful when there are volatile days as it increases the range.
        - 52 week range is super helpful to know where you are but this time on a yearly basis. It helps knowing it the stock is cheap or expensive at the moment. You need other data to determine that.
        - Volume and average volume give you information on "Is this stock traded a lot today? Compared to what it is normally".
        - Market Cap allows you to estimate if it is a big company or not. Most of the time big company are more secured as they experience high liquidity and "low" volatility.
        - Beta is a measure of the volatility of this stock compared to the market. A beta < 1 is less volatile and > 1 is more. Volatility is a measure of risk.
        - PE ratio (price to earnings) is simply the multiple at which the stock is traded compared to its revenue. It is a comparative ratio to peers stock.
        A small PE ratio consider the stock as cheap compared to peers and inversely.
        - EPS (earnings per share) is the revenue divided by the number of stock shares.
        - Earnings date is the next date at which the company publish results.
        - Dividend yield is the return you can expect from this stock just regarding dividend not a possible stock price increase. 
        - Ex-dividend date is the date at which the dividend it detached from the stock price.
        ''')

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
            stdev_round = round(stdev,2)
            st.metric('Standard Devation is',stdev_round)
            risk_adj_return = annual_return/stdev_round
            risk_adj_returnp = '{:.2%}'.format(risk_adj_return)
            st.metric('Risk Adjusted Return is',risk_adj_returnp)

        st.markdown('------------------------------------------')
        st.markdown('')
        st.markdown('The annual return gives you the return the stock experienced during that period.')
        st.markdown('The standard deviation is the measure of risk and volatility used in finance.')
        st.markdown('The risk adjusted return is measured as the annual return divided by the standard deviation. The goal is to compare it to cash return.')

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
            stdev_round = round(stdev,2)
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

        earnings_data = finnhub_client.company_earnings(ticker, limit=5) # Call the method to get the data.
        quarters = [round(x['quarter'],2) for x in earnings_data] # Iterate over the data to extract the "quarter" property.
        actual_eps = [round(x['actual'],2) for x in earnings_data]
        estimated_eps = [round(x['estimate'],2) for x in earnings_data]
        surprise_percent = [round(x['surprisePercent'],2) for x in earnings_data]
        mod_surprise_percent = [abs(sp*20) for sp in surprise_percent]  

        # Create the figure for the first plot:

        fig_actual_eps = go.Figure()

        sorted_data = sorted(zip(quarters, estimated_eps), key=lambda x: x[0])
        sorted_quarters, sorted_estimated_eps = zip(*sorted_data)
        fig_actual_eps.add_trace(go.Bar(x=quarters, y=actual_eps, width=0.4, name='Actual EPS', marker_color='#006837', text=[f"{x}%" for x in actual_eps], textposition='auto', textfont=dict(color='#DCD6D0')))
        fig_actual_eps.add_trace(go.Scatter(x=sorted_quarters, y=sorted_estimated_eps, mode='lines', name='Estimated EPS', line=dict(color='#b1de71', width=2), text=[f'{x}%' for x in sorted_estimated_eps], textposition='top center', textfont=dict(color='#DCD6D0')))
        fig_actual_eps.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), xaxis_title='Quarter', yaxis_title='Earnings Per Share (EPS)', title=f'Actual Earnings Per Share for {ticker}', legend=dict(y=-0.3, yanchor='top', x=0.5, xanchor='center'))

        # Create the figure for the second plot:

        fig_surprise_percent = go.Figure()
        legend_labels = set()

        for q, sp, mod_sp in zip(quarters, surprise_percent, mod_surprise_percent):

            if sp < 0:
                color = '#8B0000'
                legend_label = 'Surprise Percentage (-)'

            else:
                color = '#006837'
                legend_label = 'Surprise Percentage (+)'

            # Check if the combination of color and legend label already exists:

            if (color, legend_label) not in legend_labels:
                legend_labels.add((color, legend_label))
                fig_surprise_percent.add_trace(go.Scatter(x=[q], y=[sp], mode='markers', marker=dict(size=[mod_sp], opacity=1.0, color=color), name=legend_label))

            else:
                fig_surprise_percent.add_trace(go.Scatter(x=[q], y=[sp], mode='markers', marker=dict(size=[mod_sp], opacity=1.0, color=color), showlegend=False))

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

        st.subheader('Cash Flow Statement (in Millions)')
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

        # Analyst Recommendation Finnhub:

        graph, metrics = st.columns([5,2])
      
        with graph:

            def plot_stock_data(stock_data):
                symbol = stock_data[0]['symbol']
                latest_data = stock_data[0]
                labels = ['Buy', 'Hold', 'Sell', 'Strong Buy', 'Strong Sell']
                values = [latest_data['buy'], latest_data['hold'], latest_data['sell'], latest_data['strongBuy'], latest_data['strongSell']]
                
                # Calculate the index of the slice with the highest value:

                max_index = values.index(max(values))
                
                # Build the plotly pie chart:

                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                period_str = datetime.strptime(latest_data['period'], '%Y-%m-%d').strftime('%d/%m/%Y')
                fig.update_layout(title=f'Stock Ratings for {symbol} ({period_str})')
                
                # Pull the slice with the highest value:

                pull_list = [0] * len(labels)
                pull_list[max_index] = 0.1
                fig.update_traces(pull=pull_list)
                
                # Set the legend orientation and position:

                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="right", x=0.5))
                fig.update_traces(domain=dict(x=[0, 0.5]))
                st.plotly_chart(fig)

            plot_stock_data(finnhub_client.recommendation_trends(ticker))

        with metrics:

            recommendation = float(recommendation)

            if recommendation >= 1 and recommendation < 1.5:
                status = 'Strong Buy'

            elif recommendation >= 1.5 and recommendation < 2.5:
                status = 'Buy'

            elif recommendation >= 2.5 and recommendation < 3.5:
                status = 'Hold'

            elif recommendation >= 3.5 and recommendation < 4.5:
                status = 'Sell'

            else:
                status = 'Strong Sell'

            st.markdown('')
            st.markdown('')
            st.markdown ('**According to analysts :**')
            st.metric (label='Average recommendation', value=f'{recommendation}')
            st.metric (label='Analyst verdict', value=f'{status}')
            st.metric (label='Mean target price', value=f'${targetMeanPrice}')
            st.metric (label='Current price', value=f'${currentPrice}')

        def create_stock_dataframe(stock_data):
            periods = [data['period'] for data in stock_data]
            buys = [data['buy'] for data in stock_data]
            holds = [data['hold'] for data in stock_data]
            sells = [data['sell'] for data in stock_data]
            strong_buys = [data['strongBuy'] for data in stock_data]
            strong_sells = [data['strongSell'] for data in stock_data]

            df = pd.DataFrame({'Period': periods, 'Strong Buy': strong_buys, 'Buy': buys, 'Hold': holds, 'Sell': sells, 'Strong Sell': strong_sells})

            df.set_index('Period', inplace=True)  # Set the Period column as the index.
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d').strftime('%d/%m/%Y')
            return df

        df = create_stock_dataframe(finnhub_client.recommendation_trends(ticker))
        st.write(df)

        st.markdown('------------------------------------------')
        st.markdown('')
        st.markdown('You can use those data made by experts to help you decide what to do with that stock. But be aware that they are no medium.')
        st.markdown('You should analyse other data then that before making a decision but is is a useful source of information.')

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

        if ticker == '':
            ticker = 'AAPL'

        else:
            pass

        end_date = datetime.today().strftime('%d/%m/%Y')
        start_date = (datetime.today() - relativedelta(years=1)).strftime('%d/%m/%Y')

        # Load data:

        df1 = si.get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval='1d')
        df2 = si.get_dividends(ticker, start_date=start_date, end_date=end_date, index_as_date=True)
        df = pd.concat([df1, df2], axis=1)

        # Manage the columns in the dataframe:

        df['dividend'] = df['dividend'].fillna(0)
        df['% change'] = df['adjclose'] / df['adjclose'].shift(1) - 1
        df['date'] = df.index
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['close'] = df['adjclose']
        df = df.drop(['adjclose'], axis=1)

        ema = st.container()
        other_models = st.container()

        # EMA model:

        with ema:

            st.subheader(f'EMA model for {ticker}')
            st.markdown('')
            st.markdown('''
            You can find here the Exponential Moving Average (EMA). The EMA is a line that helps identify trends in the price of the asset. 

            Here's what the graph shows:

            The actual prices of the stock over time, the EMA, which is a smoothed average of the prices giving more power to latest data. It helps highlight the overall trend of the stock.
            The graph is interactive, allowing you to hover over the points to see the specific values at different dates. 
            The EMA is pertinent for stock price prediction because it smoothes price data and helps identify trends, support/resistance levels, crossovers, and potential reversals or breakouts.
            It's important to note that while the EMA is a useful tool (used a lot in finance), it should not be used in isolation for stock price prediction. 
            It is often used in conjunction with other technical indicators, fundamental analysis, and market trends to form a more comprehensive view of the stock's potential future movements.
            ''')

            # Calculate the EMA for a given time period:

            time_period = 60
            alpha = 2 / (time_period + 1)
            ema = df['close'].ewm(alpha=alpha, adjust=False).mean()

            # Calculate the threshold value:

            threshold = ema[-1]

            # Calculate the rolling EMA for the next 30 days:

            rolling_ema = df['close'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
            ema_predictions = []

            for i in range(30):
                ema_predictions.append(rolling_ema)
                rolling_ema = alpha * df['close'].iloc[-30 + i] + (1 - alpha) * rolling_ema

            # Create a Plotly figure:

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Actual'))
            fig.add_trace(go.Scatter(x=df.index, y=ema, name='EMA'))
            fig.add_trace(go.Scatter(x=df.index[-30:] + pd.DateOffset(days=42), y=ema_predictions, name='EMA Predictions'))
            fig.add_shape(type='line', x0=df.index[0], y0=threshold, x1=df.index[-1], y1=threshold, line=dict(color='Red', dash='dash'), name='Threshold')
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price'), hovermode='x', title='EMA Model for ' + ticker)

            # Display the Plotly figure in Streamlit:

            st.plotly_chart(fig)

            # Display the distribution of features of the data:

            st.markdown('')
            st.markdown('**Distribution of input values:**')
            st.markdown('')
            st.markdown(':red[*You can select and disselect the features shown in the graph through the legend in the left upper corner.]')

            # Plot distributions of features:

            fig = go.Figure()
            features = ['open', 'high', 'low', 'close', 'volume']

            for col in features:
                trace_name = col.capitalize()
                fig.add_trace(go.Histogram(x=df[col], name=trace_name))
                fig.update_layout(barmode='overlay', bargap=0.1)

            fig.update_layout(title='Distribution of the Selected Features', xaxis_title='Price/Volume', yaxis_title='Value Count')
            fig.update_traces(visible='legendonly')
            fig.update_traces(visible=True, selector=dict(name='Open'))
            st.plotly_chart(fig)

        # Other models:

        with other_models:

            st.subheader(f'Different models trying to predict future prices for {ticker}')
            st.markdown('')
            st.markdown('''
            Here you can find the training and validation accuracy of the models. It is more for the teacher purpose than for a potential user as it has no value to him. 
            On the last graph you can analyse the last year price of the stock and after the threshold the predictions made by the 3 following models. 

            **Logistic Regression:**
            Logistic Regression is useful for stock price prediction as it can model the relationship between input variables and binary outcomes, such as predicting whether a stock will increase or decrease in price, providing a probabilistic prediction.

            **Support Vector Classifier (SVC):**
            SVC with a polynomial kernel and probability estimation is beneficial for stock price prediction due to its ability to handle non-linear relationships and provide probability estimates of different price movements, aiding in decision-making.

            **XGBoost Classifier:**
            XGBoost Classifier is valuable for stock price prediction due to its ability to handle complex relationships and capture non-linear patterns in the data, resulting in accurate predictions by leveraging gradient boosting techniques.
            ''')

            end_date_model = datetime.today().strftime('%d/%m/%Y')
            start_date_model = (datetime.today() - relativedelta(years=1)).strftime('%d/%m/%Y')

            # Load data:

            df1 = si.get_data(ticker, start_date=start_date_model, end_date=end_date_model, index_as_date=True, interval='1d')
            df2 = si.get_dividends(ticker, start_date=start_date_model, end_date=end_date_model, index_as_date=True)
            df = pd.concat([df1, df2], axis=1)

            # Manage the columns in the dataframe:

            df['dividend'] = df['dividend'].fillna(0)
            df['% change'] = df['adjclose'] / df['adjclose'].shift(1) - 1
            df['date'] = df.index
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df['close'] = df['adjclose']
            df = df.drop(['adjclose'], axis=1)

            # Add date-related columns:

            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df['day'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['year']= df['date'].dt.year

            # Add is_quarter_end column:

            df['is_quarter_end'] = np.where(df['month']%3==0,1,0)

            # Add additional columns:

            df['open-close'] = df['open'] - df['close']
            df['low-high'] = df['low'] - df['high']
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

            # Prepare data for modeling:

            features = df[['open-close', 'low-high', 'is_quarter_end']]
            target = df['target']
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            # Split data into training and validation sets:

            X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

            # Train models and evaluate performance:

            models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
            for i in range(3):
                models[i].fit(X_train, Y_train)

            # Set the rolling window size:

            window_size = 230

            # Initialize a list to store the predicted prices:

            predicted_prices = []

            # Perform the rolling forecast:

            for i in range(window_size, len(df)):
                X_train = features[i-window_size:i]
                X_valid = features[i:i+1]
                Y_train = target[i-window_size:i]
                Y_valid = target[i:i+1]

                # Train models and evaluate performance:

                models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

                for model in models:
                    model.fit(X_train, Y_train)

                # Making predictions on the validation data:

                last_close = df['close'].iloc[i-1]
                preds = []
                for model in models:
                    preds.append(model.predict_proba(X_valid)[:, 1])

                # Computing the predicted prices based on the last closing price and the predicted probabilities:

                day_predicted_prices = []

                for pred in preds:
                    day_predicted_prices.append(last_close * (1 + 0.01 * pred))

                predicted_prices.append(day_predicted_prices)

            # Plotting the actual and predicted prices together:

            fig = go.Figure()

            # Add the actual data:

            fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Actual', line=dict(color='blue')))

            colors = ['orange', 'green', 'red']
            model_names = ['Logistic Regression', 'SVM', 'XGBoost']

            # Add the predicted data for each model:

            for i, model_name in enumerate(model_names):
                fig.add_trace(go.Scatter(x=df['date'].iloc[window_size:], y=[pred[i][0] for pred in predicted_prices],
                                        mode='lines', name=model_name, line=dict(color=colors[i])))

            # Configure the layout:

            fig.update_layout(title='Actual vs Predicted Stock Prices', yaxis_title='Price', showlegend=True,
                              xaxis=dict(tickformat='%Y-%m-%d', showticklabels=False ),
                              shapes=[dict(type='line', x0=df['date'].iloc[window_size],y0=min(df['close']), x1=df['date'].iloc[window_size],y1=max(df['close']),
                                           line=dict(color='#DCD6D0', width=1,dash='dash')) for window_size in range(window_size, len(df['date']), window_size)])

            # Display the figure using Streamlit:

            st.plotly_chart(fig)

    # Creating the footer:

    with footer:

        write_footer(ticker)

# Defining the page 'Final Analysis':

if selected == 'Final Analysis':

    # Creating the header:

    with header:

        st.title(f'Final Analysis for {ticker}')
        st.markdown('')

    # Content of the page:

    with content:

        st.subheader('These following analyses are a summary of the whole app and can help you make an educated choice on whether buy, sell or wait on a stock...')
        st.markdown('')
        st.markdown('')
        st.markdown ('**According to analysts:**')

        metric, metric2, graph = st.columns ([1,1,1])

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

        recommendation = float(recommendation)

        if recommendation >= 1 and recommendation < 1.5:
            status = 'Strong Buy'

        elif recommendation >= 1.5 and recommendation < 2.5:
            status = 'Buy'

        elif recommendation >= 2.5 and recommendation < 3.5:
            status = 'Hold'

        elif recommendation >= 3.5 and recommendation < 4.5:
            status = 'Sell'

        else:
            status = 'Strong Sell'

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

        annual_return = dfb['% Change'].mean()*252
        annual_returnp = '{:.2%}'.format(annual_return)

        # Webscarping the data:

        url = f'https://finance.yahoo.com/quote/{ticker}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.127 Safari/537.36',}

        response = requests.get(url, headers=headers)
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
    
        with metric:

            st.metric (label='Average recommendation', value=f'{recommendation}')
            st.metric (label='Analyst verdict', value=f'{status}')
            st.metric (label='Mean target price', value=f'${targetMeanPrice}')
            st.metric (label='Current price', value=f'${currentPrice}')

        with metric2:

            st.metric('Annual Return is',annual_returnp)

            for index, row in df_summary.iloc[9:12].iterrows():
                st.metric(label=index, value=row['Value'])

        with graph:

            # Retrieve the historical stock data for the past 52 weeks and the current price:

            stock_data = yf.download(ticker, period='1y')
            current_price = stock_data['Close'].iloc[-1]

            # Calculate the 52 week range:

            high_52_weeks = stock_data['High'].rolling(window=52).max().iloc[-1]
            low_52_weeks = stock_data['Low'].rolling(window=52).min().iloc[-1]

            # Create a vertical colorbar style plot to display the 52 week range:

            fig = go.Figure()

            # Plot the 52-week range bar with a higher z-index to ensure it's below the current price bar:

            fig.add_trace(go.Bar(x=[0], y=[high_52_weeks - low_52_weeks],
                                base=[low_52_weeks], width=[0.1], marker_color='#006837', name='52 Week Range'))

            # Plot the line representing the current price on top of the bars:

            fig.add_trace(go.Scatter(x=[0], y=[current_price], mode='markers',
                                    line=dict(color='#8B0000', width=2), name='Current Price'))

            fig.update_layout(yaxis=dict(title='Stock Price'), xaxis=dict(visible=False),
                              legend=dict(orientation='h',yanchor='bottom', y=-0.15, xanchor='right', x=0.9),
                              title=dict(text=f'{ticker} Price 52 Week Range'), height=350, width=300, margin=dict(l=0, r=0, t=40, b=0))

            st.plotly_chart(fig)

    # Creating the footer:

    with footer:

        write_footer(ticker)
