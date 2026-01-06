
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.RandomForest import RandomForest

ticker = "AAPL"
start_date = "2016-01-01"
end_date = "2025-12-31"
df = yf.download(ticker, start=start_date, end=end_date)
df.columns = df.columns.droplevel(1)
display(df.head())

display(df.describe())
display(df.info())

plt.figure(figsize=(10,4))
plt.plot(df.index, df['Close'])
plt.title('Giá đóng cửa cổ phiếu AAPL')
plt.xlabel('Thời gian')
plt.ylabel('Giá')
plt.grid(True)
plt.show()

macro_tickers = ['^VIX', '^GSPC', 'DX-Y.NYB']
macro_data = yf.download(macro_tickers, start=start_date, end=end_date)['Close']
macro_data.columns = ['vix_close', 'sp500_close', 'dxy_close']

df = df.join(macro_data, how='left')
df[['vix_close', 'sp500_close', 'dxy_close']] = df[['vix_close', 'sp500_close', 'dxy_close']].ffill()

df['vix_change'] = df['vix_close'].diff()
df['sp500_return'] = df['sp500_close'].pct_change()
df['dxy_return'] = df['dxy_close'].pct_change()

df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility_5'] = df['log_return'].rolling(5).std()
df['ma_5'] = df['Close'].rolling(5).mean()
df['ma_10'] = df['Close'].rolling(10).mean()
df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
df['open_close_change'] = (df['Close'] - df['Open']) / df['Open']

# RSI 14
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
loss = -delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

#MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# OBV
df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

df = df.dropna()
df['target'] = df['log_return'].shift(-1)
df = df.dropna()

display(df.head())
display(df.info())
display(df.describe())

plt.figure(figsize=(10,4))
plt.plot(df.index, df['log_return'])
plt.title('log_return theo thời gian')
plt.xlabel('Thời gian')
plt.ylabel('log_return')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['log_return'], bins=60, kde=True)
plt.title('Phân phối log_return')
plt.xlabel('log_return')
plt.ylabel('Tần suất')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df['sp500_return'],
    y=df['log_return'],
    alpha=0.4
)
plt.title('log_return vs sp500_return')
plt.xlabel('sp500_return')
plt.ylabel('log_return')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df['vix_change'],
    y=df['log_return'],
    alpha=0.4
)
plt.title('log_return vs vix_change')
plt.xlabel('vix_change')
plt.ylabel('log_return')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng")
plt.show()

selected_features = [
    'Close',
    'log_return',
    'sp500_return',
    'vix_change',
    'open_close_change',
    'obv',
    'ma_5',
    'volatility_5',
]

X = df[selected_features].values
y = df['target'].values

split_idx = int(len(X) * 0.85)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

rf = RandomForest(
    n_trees=100,
    max_depth=10,
    min_samples_split=5,
    n_features=int(X.shape[1]/3),
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

base_prices = X_test[:, selected_features.index('Close')]

predicted_prices = base_prices * np.exp(y_pred)
actual_prices = base_prices * np.exp(y_test)

dates_test = df.index[split_idx:]

plt.figure(figsize=(14, 7))
plt.plot(dates_test, actual_prices, label='Giá thực tế', linewidth=2)
plt.plot(dates_test, predicted_prices, label='Giá dự báo', linestyle='--')

plt.title('So sánh giá thực tế và giá dự báo')
plt.xlabel('Thời gian')
plt.ylabel('Giá cổ phiếu')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
