from flask import Flask, render_template, request, redirect, url_for, jsonify
import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import requests
import numpy as np
from scipy import stats

app = Flask(__name__)

# Stock tickers list
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'DIS', 'MA', 'PYPL',
           'NFLX', 'ADBE', 'PFE', 'KO', 'MRK', 'NKE', 'INTC', 'CMCSA', 'PEP', 'T', 'ABT', 'CSCO', 'VZ', 'XOM', 'CVX',
           'LLY', 'AVGO', 'COST', 'MDT', 'WMT', 'MCD', 'BAC', 'ACN', 'QCOM', 'CRM', 'DHR', 'NEE', 'TXN', 'LIN', 'HON',
           'PM', 'ABBV', 'UNP', 'SBUX']
user_portfolios = {}

# Defensive and growth stocks for recommendations
defensive_stocks = ['JNJ', 'PG', 'KO', 'WMT', 'PEP']
growth_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period='1d', interval='1m')
    return stock_data.tail(1)

def calculate_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    metrics = {
        "Ticker": ticker,
        "EPS": info.get("trailingEps"),
        "PE_ratio": info.get("trailingPE"),
        "PB_ratio": info.get("priceToBook"),
        "Dividend_yield": info.get("dividendYield"),
        "Debt_to_equity": info.get("debtToEquity"),
        "ROE": info.get("returnOnEquity"),
        "Revenue_growth": info.get("revenueGrowth"),
        "Profit_margin": info.get("profitMargins")
    }
    return metrics

def preprocess_data(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if len(data) < 10:  # Ensure minimum data points
        return pd.DataFrame()  # Return empty DataFrame if insufficient data
    data['Return'] = data['Close'].pct_change().fillna(0)
    data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean().fillna(0)
    data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean().fillna(0)
    data['Lag_1'] = data['Close'].shift(1).fillna(data['Close'].iloc[0])
    data['DayOfWeek'] = data.index.dayofweek
    return data.dropna()

def train_predict(data, prediction_length):
    if data.empty or len(data) < 10:  # Check for sufficient data
        return pd.Series(), 0  # Return empty series and zero accuracy if insufficient data
    
    data['DateInt'] = data.index.map(pd.Timestamp.toordinal)
    X = data[['DateInt', 'SMA_5', 'SMA_10', 'Lag_1', 'DayOfWeek']]
    y = data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = 100 - mean_absolute_percentage_error(y_test, y_pred) * 100
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_length, freq='D')
    future_data = pd.DataFrame({'DateInt': future_dates.map(pd.Timestamp.toordinal).values})
    future_data['SMA_5'] = data['SMA_5'].iloc[-1]
    future_data['SMA_10'] = data['SMA_10'].iloc[-1]
    future_data['Lag_1'] = data['Close'].iloc[-1]
    future_data['DayOfWeek'] = future_dates.dayofweek
    future_predictions = model.predict(future_data)
    
    return pd.Series(future_predictions, index=future_dates), accuracy

def preprocess_gold_data(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if len(data) < 10:
        return pd.DataFrame()
    data['Return'] = data['Close'].pct_change().fillna(0)
    data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean().fillna(0)
    data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean().fillna(0)
    data['Lag_1'] = data['Close'].shift(1).fillna(data['Close'].iloc[0])
    data['DayOfWeek'] = data.index.dayofweek
    return data.dropna()

def predict_gold_prices(data, prediction_length):
    if data.empty or len(data) < 10:
        return pd.Series(), 0
    
    data['DateInt'] = data.index.map(pd.Timestamp.toordinal)
    X = data[['DateInt', 'SMA_5', 'SMA_10', 'Lag_1', 'DayOfWeek']]
    y = data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = 100 - mean_absolute_percentage_error(y_test, y_pred) * 100
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_length, freq='D')
    future_data = pd.DataFrame({'DateInt': future_dates.map(pd.Timestamp.toordinal).values})
    future_data['SMA_5'] = data['SMA_5'].iloc[-1]
    future_data['SMA_10'] = data['SMA_10'].iloc[-1]
    future_data['Lag_1'] = data['Close'].iloc[-1]
    future_data['DayOfWeek'] = future_dates.dayofweek
    future_predictions = model.predict(future_data)
    
    return pd.Series(future_predictions, index=future_dates), accuracy

def analyze_portfolio_risk(portfolio):
    if not portfolio:
        return "Empty", ["Add some stocks to your portfolio to assess risk"], 0, 0
    
    volatilities = []
    total_value = sum(stock['total_value'] for stock in portfolio)
    stock_weights = {}
    
    for stock in portfolio:
        stock_data = yf.Ticker(stock['stock']).history(period="1y", interval="1d")
        returns = stock_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        weight = stock['total_value'] / total_value
        volatilities.append(volatility * weight)
        stock_weights[stock['stock']] = weight
    
    portfolio_volatility = sum(volatilities)
    num_stocks = len(portfolio)
    concentration = max(stock['total_value']/total_value for stock in portfolio)
    
    suggestions = []
    
    if portfolio_volatility < 0.15 and num_stocks >= 10 and concentration < 0.3:
        risk_level = "Low"
        suggestions.extend([
            "Well-diversified portfolio with low volatility - excellent risk management",
            "Consider adding selective growth stocks like {} for potential returns".format(', '.join(growth_stocks[:2])),
            "Maintain exposure to defensive stocks like {} for stability".format(', '.join(defensive_stocks[:2])),
            "Review portfolio quarterly to maintain this balanced approach"
        ])
    elif (portfolio_volatility < 0.25 and num_stocks >= 5 and concentration < 0.5) or \
         (portfolio_volatility < 0.15 and num_stocks >= 5):
        risk_level = "Medium"
        suggestions.extend([
            "Moderately diversified portfolio with acceptable risk level",
            "Consider adding {}-{} more stocks to improve diversification".format(2, 5-num_stocks if num_stocks < 5 else 3),
            "Look into stable stocks like {} to reduce volatility".format(', '.join(defensive_stocks[:3])),
            "Consider trimming positions where single stock exceeds 20% of portfolio",
            "Monitor market trends and rebalance semi-annually"
        ])
    else:
        risk_level = "High"
        high_vol_stocks = [stock['stock'] for stock in portfolio if 
                          yf.Ticker(stock['stock']).history(period="1y").Close.pct_change().std() * np.sqrt(252) > 0.3]
        suggestions.extend([
            "Portfolio shows high risk due to volatility and/or concentration",
            "Diversify by adding {}-{} more stocks from different sectors".format(5-num_stocks if num_stocks < 5 else 3, 10-num_stocks if num_stocks < 10 else 5),
            "Consider reducing exposure to high-volatility stocks like {}".format(', '.join(high_vol_stocks[:2]) if high_vol_stocks else "identified volatile positions"),
            "Add defensive stocks like {} for stability".format(', '.join(defensive_stocks[:3])),
            "Review and rebalance portfolio monthly until risk level decreases",
            "Consider dollar-cost averaging to mitigate entry point risks"
        ])
    
    beta_values = []
    for stock in portfolio:
        stock_info = yf.Ticker(stock['stock']).info
        beta = stock_info.get('beta', 1.0)
        beta_values.append(beta * (stock['total_value']/total_value))
    
    portfolio_beta = sum(beta_values)
    if portfolio_beta > 1.2:
        suggestions.append("Portfolio has high market sensitivity (beta > 1.2) - consider adding low-beta stocks like {}".format(defensive_stocks[0]))
    elif portfolio_beta < 0.8:
        suggestions.append("Portfolio has low market sensitivity (beta < 0.8) - opportunity to add growth stocks like {}".format(growth_stocks[0]))
    
    if concentration > 0.5:
        suggestions.append("Reduce concentration risk - largest position is {:.1f}% of portfolio".format(concentration*100))
    if num_stocks < 3:
        suggestions.append("Urgent: Too few stocks ({}) - aim for at least 5-10 for proper diversification".format(num_stocks))
    
    return risk_level, suggestions, portfolio_volatility, portfolio_beta

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/top5')
def top5_stocks():
    metrics_list = []
    for ticker in tickers:
        metrics = calculate_metrics(ticker)
        stock_data = fetch_stock_data(ticker)
        real_time_value = stock_data['Close'].values[0] if not stock_data.empty else None
        metrics['RealTimeValue'] = real_time_value
        metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    df['Score'] = (df['EPS'] * 0.2 + df['PE_ratio'] * 0.1 + df['PB_ratio'] * 0.1 +
                   df['Dividend_yield'] * 0.1 + df['Debt_to_equity'] * -0.1 +
                   df['ROE'] * 0.2 + df['Revenue_growth'] * 0.1 + df['Profit_margin'] * 0.1)
    top_stocks = df.sort_values(by='Score', ascending=False).head(5)
    return render_template('top5.html', top_stocks=top_stocks)

@app.route('/predictions')
def predictions():
    predictions_dict = {}
    accuracies = {}
    prediction_dates = {}
    total_accuracy = 0
    prediction_length = 30
    current_date = datetime.now().date()
    start_date = current_date - timedelta(days=365)  # Use past year of data
    
    for ticker in tickers:
        stock_data = yf.Ticker(ticker).history(start=start_date, end=current_date, interval="1d")
        if stock_data.empty:
            print(f"No data available for {ticker}")
            continue
        processed_data = preprocess_data(stock_data)
        if processed_data.empty:
            print(f"Processed data empty for {ticker}")
            continue
        predictions, accuracy = train_predict(processed_data, prediction_length)
        if not predictions.empty:
            predictions_dict[ticker] = predictions
            accuracies[ticker] = accuracy
            prediction_dates[ticker] = predictions.index.strftime('%Y-%m-%d').tolist()
            total_accuracy += accuracy
    
    if not predictions_dict:
        return render_template('predictions.html', error="No valid prediction data available for any ticker")
    
    average_accuracy = total_accuracy / len(predictions_dict)
    
    return render_template('predictions.html', predictions=predictions_dict, accuracies=accuracies, 
                           prediction_dates=prediction_dates, average_accuracy=average_accuracy)

@app.route('/candlesticks')
def candlesticks():
    candlestick_figs = []
    current_date = datetime.now().date()
    start_date = current_date - timedelta(days=365)  # Use past year
    
    for ticker in tickers:
        stock_data = yf.Ticker(ticker).history(start=start_date, end=current_date, interval="1d")
        if stock_data.empty:
            continue
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
        candlestick_figs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
    return render_template('candlesticks.html', candlestick_figs=candlestick_figs)

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    if request.method == 'POST':
        user = request.form.get('username')
        stock = request.form.get('stock')
        quantity = int(request.form.get('quantity', 0))
        buying_price = float(request.form.get('buying_price', 0.0))
        
        if user not in user_portfolios:
            user_portfolios[user] = []
        user_portfolios[user].append({'stock': stock, 'quantity': quantity, 'buying_price': buying_price})
        
        return redirect(url_for('view_portfolio', username=user))
    
    return render_template('portfolio.html', tickers=tickers)

@app.route('/view_portfolio/<username>')
def view_portfolio(username):
    if username in user_portfolios:
        portfolio = user_portfolios[username]
        total_profit_loss = 0
        
        for stock_entry in portfolio:
            stock_data = fetch_stock_data(stock_entry['stock'])
            current_price = stock_data['Close'].values[0] if not stock_data.empty else 0.0
            stock_entry['current_price'] = current_price
            stock_entry['profit_loss'] = ((current_price - stock_entry['buying_price']) / stock_entry['buying_price']) * 100
            stock_entry['total_value'] = current_price * stock_entry['quantity']
            stock_entry['initial_value'] = stock_entry['buying_price'] * stock_entry['quantity']
            stock_entry['profit_loss_value'] = stock_entry['total_value'] - stock_entry['initial_value']
            total_profit_loss += stock_entry['profit_loss_value']
        
        df = pd.DataFrame(portfolio)
        fig = px.line(df, x='stock', y=['buying_price', 'current_price'], title='Buying Price vs Current Price')
        line_graph_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        return render_template(
            'view_portfolio.html',
            portfolio=portfolio,
            total_profit_loss=total_profit_loss,
            line_graph_html=line_graph_html,
            username=username
        )
    
    return f"No portfolio found for {username}"

@app.route('/portfolio_classification/<username>')
def portfolio_classification(username):
    if username not in user_portfolios:
        return f"No portfolio found for {username}"
    
    portfolio = user_portfolios[username]
    
    total_value = 0
    for stock_entry in portfolio:
        stock_data = fetch_stock_data(stock_entry['stock'])
        current_price = stock_data['Close'].values[0] if not stock_data.empty else 0.0
        stock_entry['current_price'] = current_price
        stock_entry['total_value'] = current_price * stock_entry['quantity']
        total_value += stock_entry['total_value']
    
    risk_level, suggestions, volatility, beta = analyze_portfolio_risk(portfolio)
    
    fig = px.pie(
        values=[stock['total_value'] for stock in portfolio],
        names=[stock['stock'] for stock in portfolio],
        title='Portfolio Allocation'
    )
    pie_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return render_template(
        'portfolio_classification.html',
        username=username,
        portfolio=portfolio,
        risk_level=risk_level,
        suggestions=suggestions,
        volatility=round(volatility*100, 2),
        beta=round(beta, 2),
        pie_chart_html=pie_chart_html,
        total_value=round(total_value, 2)
    )

@app.route('/gold_predictions')
def gold_predictions():
    try:
        gold_ticker = "GC=F"
        current_date = datetime.now().date()
        start_date = current_date - timedelta(days=365)
        gold_data = yf.Ticker(gold_ticker).history(start=start_date, end=current_date, interval="1d")
        processed_data = preprocess_gold_data(gold_data)
        
        prediction_length = 30
        predictions, accuracy = predict_gold_prices(processed_data, prediction_length)
        
        if predictions.empty:
            return render_template('gold_predictions.html', error="No valid prediction data available for gold")
        
        fig = px.line(
            x=predictions.index, 
            y=predictions.values,
            title="Gold Price Predictions (Next 30 Days)",
            labels={"x": "Date", "y": "Predicted Price (USD)"}
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

        return render_template('gold_predictions.html', predictions=predictions, accuracy=round(accuracy, 2), chart_html=chart_html)
    except Exception as e:
        print(f"Error fetching or predicting gold prices: {e}")
        return "Error fetching or predicting gold prices. Please try again later."

if __name__ == '__main__':
    app.run(debug=True)