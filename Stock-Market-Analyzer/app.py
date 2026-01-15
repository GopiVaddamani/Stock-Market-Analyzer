import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, session, jsonify
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import mysql.connector
import time
import requests
from dotenv import load_dotenv
 
app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for sessions

# Load environment variables
load_dotenv()

# API Keys for news services
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")  # optional fallback

# Simple in-memory cache to avoid calling the news API too often
NEWS_CACHE = {
    'ts': 0,
    'data': None,
    'symbol': None
}

def fetch_market_news(symbol: str | None = None, cache_ttl: int = 60):
    """
    Returns a list of news articles (dicts). If symbol is provided, fetches company-news
    (last 7 days). Otherwise fetches general market news.
    """
    now = time.time()
    # return cached if within TTL and same symbol
    if NEWS_CACHE['data'] and (now - NEWS_CACHE['ts'] < cache_ttl) and (NEWS_CACHE['symbol'] == symbol):
        return NEWS_CACHE['data']

    headers = {"Accept": "application/json"}
    try:
        if FINNHUB_API_KEY is None:
            raise RuntimeError("FINNHUB_API_KEY not configured in environment")

        if symbol:
            # fetch last 7 days of company news (Finnhub company-news)
            to_date = dt.date.today()
            from_date = to_date - dt.timedelta(days=7)
            url = (
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={symbol}&from={from_date.isoformat()}&to={to_date.isoformat()}&token={FINNHUB_API_KEY}"
            )
        else:
            # general market news: category can be general, forex, crypto etc. (Finnhub market news)
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()

        # Normalize the returned list to a consistent schema for the frontend
        normalized = []
        for a in payload:
            normalized.append({
                'id': a.get('id') or a.get('uuid') or '',
                'headline': a.get('headline') or a.get('title') or '',
                'source': a.get('source') or a.get('news_source') or '',
                'image': a.get('image') or None,
                'datetime': a.get('datetime') or a.get('published_utc') or None,
                'summary': a.get('summary') or a.get('description') or '',
                'url': a.get('url') or a.get('article_url') or ''
            })

        # update cache
        NEWS_CACHE['ts'] = now
        NEWS_CACHE['data'] = normalized
        NEWS_CACHE['symbol'] = symbol
        return normalized

    except Exception as e:
        # Optional: fallback to Marketaux if configured (simple fallback)
        try:
            if MARKETAUX_API_KEY:
                url_m = "https://api.marketaux.com/v1/news/all"
                params = {"api_token": MARKETAUX_API_KEY, "language": "en", "limit": 50}
                if symbol:
                    params['symbols'] = symbol
                r = requests.get(url_m, params=params, timeout=8)
                r.raise_for_status()
                p = r.json()
                items = p.get('data') or p.get('news') or []
                normalized = []
                for a in items:
                    normalized.append({
                        'id': a.get('uuid') or '',
                        'headline': a.get('title') or a.get('headline') or '',
                        'source': a.get('source') or '',
                        'image': a.get('image'),
                        'datetime': a.get('published_at') and int(dt.datetime.fromisoformat(a.get('published_at')).timestamp()) or None,
                        'summary': a.get('summary') or '',
                        'url': a.get('url') or ''
                    })
                NEWS_CACHE['ts'] = now
                NEWS_CACHE['data'] = normalized
                NEWS_CACHE['symbol'] = symbol
                return normalized
        except Exception:
            pass

        # if everything fails, return an empty list so UI can handle gracefully
        print(f"Error fetching news: {e}")
        return []

# Stock Recommendations Helper Functions
def calculate_rsi(prices, window=14):
    """Calculate RSI without TA-Lib dependency"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return 50

def calculate_technical_indicators(df):
    """Calculate technical indicators for stock analysis"""
    try:
        indicators = {}
        
        # RSI (Relative Strength Index)
        if len(df) >= 14:
            indicators['rsi'] = calculate_rsi(df['Close'])
        else:
            indicators['rsi'] = None
            
        # EMAs
        if len(df) >= 20:
            ema_20 = df['Close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['Close'].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else None
            current_price = df['Close'].iloc[-1]
            
            if ema_50:
                if current_price > ema_20 > ema_50:
                    indicators['ema_signal'] = "Bullish"
                elif current_price < ema_20 < ema_50:
                    indicators['ema_signal'] = "Bearish"
                else:
                    indicators['ema_signal'] = "Neutral"
            else:
                indicators['ema_signal'] = "Bullish" if current_price > ema_20 else "Bearish"
        else:
            indicators['ema_signal'] = "Insufficient Data"
            
        # Volume trend (compare recent average to historical average)
        if len(df) >= 20:
            recent_vol_avg = df['Volume'].tail(5).mean()
            historical_vol_avg = df['Volume'].mean()
            
            if historical_vol_avg > 0:
                vol_ratio = recent_vol_avg / historical_vol_avg
                
                if vol_ratio > 1.2:
                    indicators['volume_trend'] = "High"
                elif vol_ratio < 0.8:
                    indicators['volume_trend'] = "Low"
                else:
                    indicators['volume_trend'] = "Normal"
            else:
                indicators['volume_trend'] = "Unknown"
        else:
            indicators['volume_trend'] = "Unknown"
            
        # Volatility (standard deviation of returns)
        if len(df) >= 20:
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * 100
                
                if volatility > 3:
                    indicators['volatility'] = "High"
                elif volatility < 1:
                    indicators['volatility'] = "Low"
                else:
                    indicators['volatility'] = "Medium"
            else:
                indicators['volatility'] = "Unknown"
        else:
            indicators['volatility'] = "Unknown"
            
        return indicators
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return {
            'rsi': None,
            'ema_signal': 'Error',
            'volume_trend': 'Error',
            'volatility': 'Error'
        }

def generate_recommendation(symbol, df, indicators):
    """Generate buy/sell/hold recommendation based on technical analysis"""
    try:
        score = 0
        reasoning_parts = []
        
        # RSI Analysis
        if indicators.get('rsi') is not None:
            rsi = indicators['rsi']
            if rsi < 30:
                score += 2
                reasoning_parts.append(f"RSI ({rsi:.1f}) indicates oversold conditions")
            elif rsi > 70:
                score -= 2
                reasoning_parts.append(f"RSI ({rsi:.1f}) indicates overbought conditions")
            else:
                reasoning_parts.append(f"RSI ({rsi:.1f}) is in neutral range")
                
        # EMA Signal Analysis
        ema_signal = indicators.get('ema_signal', 'Unknown')
        if ema_signal == "Bullish":
            score += 1
            reasoning_parts.append("EMA crossover shows bullish trend")
        elif ema_signal == "Bearish":
            score -= 1
            reasoning_parts.append("EMA crossover shows bearish trend")
        else:
            reasoning_parts.append("EMA signals are neutral")
            
        # Volume Analysis
        volume_trend = indicators.get('volume_trend', 'Unknown')
        if volume_trend == "High":
            score += 1
            reasoning_parts.append("Above-average trading volume supports the move")
        elif volume_trend == "Low":
            score -= 0.5
            reasoning_parts.append("Below-average volume shows weak conviction")
            
        # Price momentum (simple 5-day vs 20-day comparison)
        if len(df) >= 20:
            recent_avg = df['Close'].tail(5).mean()
            medium_avg = df['Close'].tail(20).mean()
            
            if recent_avg > medium_avg * 1.02:
                score += 1
                reasoning_parts.append("Recent price momentum is positive")
            elif recent_avg < medium_avg * 0.98:
                score -= 1
                reasoning_parts.append("Recent price momentum is negative")
                
        # Generate recommendation and confidence
        if score >= 3:
            recommendation = "Strong Buy"
            confidence = min(85 + (score - 3) * 3, 95)
        elif score >= 1:
            recommendation = "Buy"
            confidence = 65 + score * 5
        elif score <= -3:
            recommendation = "Strong Sell"
            confidence = min(85 + abs(score + 3) * 3, 95)
        elif score <= -1:
            recommendation = "Sell"
            confidence = 65 + abs(score) * 5
        else:
            recommendation = "Hold"
            confidence = 60
            
        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Analysis based on technical indicators."
        
        return {
            'recommendation': recommendation,
            'confidence': int(confidence),
            'reasoning': reasoning,
            'score': score
        }
        
    except Exception as e:
        print(f"Error generating recommendation for {symbol}: {e}")
        return {
            'recommendation': 'Hold',
            'confidence': 50,
            'reasoning': 'Unable to complete technical analysis due to insufficient data.',
            'score': 0
        }

def analyze_stock_for_recommendation(symbol, period='3mo'):
    """Analyze a single stock and return recommendation data"""
    try:
        # Download stock data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            return None
            
        # Get current price and calculate change
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(df)
        
        # Generate recommendation
        rec_data = generate_recommendation(symbol, df, indicators)
        
        # Get company info
        try:
            info = stock.info
            company_name = info.get('longName', symbol)
        except:
            company_name = symbol
            
        return {
            'symbol': symbol,
            'name': company_name,
            'current_price': float(current_price),
            'price_change': float(price_change),
            'recommendation': rec_data['recommendation'],
            'confidence': rec_data['confidence'],
            'reasoning': rec_data['reasoning'],
            'technical_indicators': indicators
        }
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return {
            'symbol': symbol,
            'name': symbol,
            'current_price': None,
            'price_change': 0,
            'recommendation': 'Hold',
            'confidence': 50,
            'reasoning': f'Analysis failed: {str(e)}',
            'technical_indicators': {
                'rsi': None,
                'ema_signal': 'Error',
                'volume_trend': 'Error',
                'volatility': 'Error'
            }
        }

# Connect MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",         # your MySQL username
    password="Sathwika07@",  # your MySQL password
    database="gopi_db"
)
cursor = db.cursor(dictionary=True)

# News API Routes
@app.route('/api/news')
def api_news():
    """
    Returns news articles as JSON. Optional query param:
        - symbol: ticker (e.g. 'TCS.NS', 'RELIANCE.NS', 'AAPL')
    """
    symbol = request.args.get('symbol')
    articles = fetch_market_news(symbol=symbol, cache_ttl=60)
    return jsonify(status='ok', count=len(articles), articles=articles)

@app.route('/news')
def news_page():
    """
    Renders the news page. The page's JS calls /api/news to populate the feed.
    """
    # Check if user is logged in
    if 'user_ID' not in session:
        return redirect(url_for('login'))
    
    user_name = session.get('user_name')
    return render_template('news.html', user_name=user_name)

# Stock Recommendations Routes
@app.route('/recommendations')
def recommendations_page():
    """Renders the stock recommendations page"""
    # Check if user is logged in
    if 'user_ID' not in session:
        return redirect(url_for('login'))
    
    user_name = session.get('user_name')
    return render_template('recommendations.html', user_name=user_name)

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """API endpoint to generate stock recommendations"""
    try:
        data = request.get_json()
        symbols_str = data.get('symbols', '')
        period = data.get('period', '3mo')
        
        if not symbols_str:
            return jsonify(status='error', message='No symbols provided'), 400
            
        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
        
        if not symbols:
            return jsonify(status='error', message='No valid symbols provided'), 400
            
        # Limit to prevent abuse
        if len(symbols) > 20:
            return jsonify(status='error', message='Maximum 20 symbols allowed'), 400
            
        # Analyze each stock
        recommendations = []
        for symbol in symbols:
            rec = analyze_stock_for_recommendation(symbol, period)
            if rec:
                recommendations.append(rec)
                
        if not recommendations:
            return jsonify(status='error', message='No valid recommendations generated'), 400
            
        return jsonify(
            status='ok',
            count=len(recommendations),
            recommendations=recommendations,
            period=period,
            generated_at=dt.datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error in recommendations API: {e}")
        return jsonify(status='error', message=f'Server error: {str(e)}'), 500

# ------------------ Signup ------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        password = request.form['password']

        try:
            cursor.execute(
                "INSERT INTO stockdb (first_Name, last_Name, Email_ID, Pass_Word) VALUES (%s, %s, %s, %s)",
                (fname, lname, email, password)
            )
            db.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            flash(f"Error: {err}", "danger")

    return render_template('signup.html')

# ------------------ Login ------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM stockdb WHERE Email_ID = %s AND Pass_Word = %s", (email, password))
        user = cursor.fetchone()

        if user:
            session['user_ID'] = user['user_ID']
            session['user_name'] = user['first_Name']
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password!", "danger")

    return render_template('login.html')

# ------------------ Logout ------------------
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

plt.style.use("fivethirtyeight")

# Load the model (make sure your model is in the correct path)
model = load_model('models/my_model.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_ID' not in session:
        return redirect(url_for('login'))

    error_msg = None
    user_name = session.get('user_name')
    if request.method == 'POST':
        stock = request.form.get('stock')

        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()

        # Download stock data
        df = yf.download(stock, start=start, end=end)

        # Check if data is empty
        if df.empty or 'Close' not in df.columns:
            error_msg = f"No data found for symbol '{stock}'. Please check the symbol and try again."
            return render_template('index.html', error=error_msg, user_name=user_name)

        # Descriptive Data
        data_desc = df.describe()

        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        # Check if training/testing data is sufficient
        if data_training.empty or data_testing.empty or len(data_training) < 100:
            error_msg = "Not enough data for training/prediction. Please try another symbol."
            return render_template('index.html', error=error_msg, user_name=user_name)

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        if x_test.size == 0:
            error_msg = "Not enough data for prediction. Please try another symbol."
            return render_template('index.html', error=error_msg, user_name=user_name)

        y_predicted = model.predict(x_test)

        # Inverse scaling for predictions
        scaler_scale = scaler.scale_
        scale_factor = 1 / scaler_scale[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)

        # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        # Return the rendered template with charts and dataset
        return render_template('index.html',
                               plot_path_ema_20_50=ema_chart_path,
                               plot_path_ema_100_200=ema_chart_path_100_200,
                               plot_path_prediction=prediction_chart_path,
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path,
                               error=None,
                               user_name=user_name)

    return render_template('index.html', error=None, user_name=user_name)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)