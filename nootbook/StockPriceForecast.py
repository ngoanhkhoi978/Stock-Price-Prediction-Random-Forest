
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.RandomForest import RandomForest

ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-12-31"
df = yf.download(ticker, start="2016-01-01", end="2025-12-31")
df.columns = df.columns.droplevel(1)
print(df)

df.describe()

plt.figure(figsize=(10,4))
plt.plot(df.index, df['Close'])
plt.title(f'Giá đóng cửa cổ phiếu {ticker}')
plt.xlabel('Thời gian')
plt.ylabel('Giá')
plt.grid(True)
plt.show()

# --- 2. BỔ SUNG DỮ LIỆU VĨ MÔ (VIX, SP500, USD Index) ---
# Tải dữ liệu vĩ mô cùng khoảng thời gian
macro_tickers = ['^VIX', '^GSPC', 'DX-Y.NYB']
print("Đang tải dữ liệu vĩ mô...")
macro_data = yf.download(macro_tickers, start=start_date, end=end_date)['Close']

# Đổi tên cột cho dễ hiểu
macro_data.columns = ['dxy_close', 'sp500_close', 'vix_close']

# Gộp vào DataFrame chính (dùng left join để ưu tiên ngày giao dịch của AAPL)
df = df.join(macro_data, how='left')

# Xử lý dữ liệu thiếu (do lệch múi giờ hoặc ngày nghỉ lễ khác nhau) bằng cách lấy giá trị ngày trước đó
df[['dxy_close', 'sp500_close', 'vix_close']] = df[['dxy_close', 'sp500_close', 'vix_close']].ffill()

# Tạo đặc trưng biến động cho vĩ mô
df['vix_change'] = df['vix_close'].diff()        # Thay đổi của chỉ số sợ hãi
df['sp500_return'] = df['sp500_close'].pct_change() # Lợi nhuận thị trường chung
df['dxy_return'] = df['dxy_close'].pct_change()   # Sức mạnh đồng USD



# --- 1. Tính toán các đặc trưng cơ bản (Code cũ của bạn) ---
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility_5'] = df['log_return'].rolling(5).std()
df['ma_5'] = df['Close'].rolling(5).mean()
df['ma_10'] = df['Close'].rolling(10).mean()
df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
df['open_close_change'] = (df['Close'] - df['Open']) / df['Open']

# --- 2. BỔ SUNG: Các chỉ báo nâng cao (Tâm lý & Dòng tiền) ---

# A. RSI (Relative Strength Index) - Chỉ báo Sợ hãi & Tham lam
# Chu kỳ thường dùng là 14 ngày. RSI > 70: Quá mua, RSI < 30: Quá bán.
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# B. Stochastic Oscillator (%K) - Động lượng giá
# So sánh giá đóng cửa với biên độ giá trong 14 ngày.
low_14 = df['Low'].rolling(window=14).min()
high_14 = df['High'].rolling(window=14).max()
df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))

# C. Williams %R - Điểm đảo chiều
# Tương tự Stochastic nhưng thang đo từ -100 đến 0. > -20 là quá mua, < -80 là quá bán.
df['williams_r'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100

# D. MACD (Moving Average Convergence Divergence) - Xu hướng
# Chênh lệch giữa EMA 12 và EMA 26.
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean() # Đường tín hiệu

# E. On Balance Volume (OBV) - Dòng tiền thông minh
# Tích lũy khối lượng dựa trên việc giá tăng hay giảm.
df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

# --- 3. Xử lý dữ liệu và tạo Target ---

# Loại bỏ các giá trị NaN sinh ra do rolling window (lớn nhất là 26 ngày của MACD)
df = df.dropna()

# Tạo target: Bạn đang dùng log_return của ngày hôm sau
df['target'] = df['log_return'].shift(-1)

# Loại bỏ hàng cuối cùng (vì shift(-1) sẽ tạo ra NaN ở dòng cuối)
df = df.dropna()

# Hiển thị kết quả
df.tail()

plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng")
plt.show()

corr_matrix = df.corr()
target_corr = corr_matrix['target'].abs().sort_values(ascending=False)
top_features = target_corr.drop('target')
print(top_features.head(10))

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

X = df[selected_features]
y = df['target']

# --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH TỰ BUILD ---

# 1. Chuyển đổi sang Numpy Array (Bắt buộc vì class Node dùng cú pháp array)
X_values = X.values
y_values = y.values

# 2. Chia tập dữ liệu theo thời gian (Time Series Split)
# Lấy 85% đầu để train, 15% sau để test
split_idx = int(len(X_values) * 0.85)

X_train, X_test = X_values[:split_idx], X_values[split_idx:]
y_train, y_test = y_values[:split_idx], y_values[split_idx:]

# 3. Lấy giá Close gốc của tập Test để phục vụ việc quy đổi giá (Price Reconstruction)
# Tìm vị trí cột 'Close' trong X để lấy giá nền
try:
    close_col_idx = selected_features.index('Close')
    base_prices_test = X_test[:, close_col_idx]
except ValueError:
    print("LỖI: Cần có cột 'Close' trong selected_features để quy đổi giá!")
    # Fallback: Lấy từ df gốc nếu không tìm thấy trong X
    base_prices_test = df['Close'].iloc[split_idx:].values

print(f"Kích thước Train: {len(X_train)} | Kích thước Test: {len(X_test)}")

# --- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST TỰ BUILD ---

print("\nĐang khởi tạo và huấn luyện Random Forest (Custom)...")
# Lưu ý: n_jobs=-1 để chạy song song đa nhân CPU
rf_custom = RandomForest(
    n_trees=100,            # Số lượng cây
    max_depth=10,           # Độ sâu tối đa
    min_samples_split=5,    # Số mẫu tối thiểu để tách nút
    n_features=int(X.shape[1] / 3), # Rule of thumb: features/3
    random_state=42,
    n_jobs=-1
)

rf_custom.fit(X_train, y_train)
print("Huấn luyện xong!")

# --- BƯỚC 3: DỰ BÁO VÀ QUY ĐỔI GIÁ ---

print("Đang dự báo trên tập Test...")
# 1. Dự báo Log Return
y_pred_log_return = rf_custom.predict(X_test)

# 2. Quy đổi từ Log Return về Giá dự báo (Predicted Price)
# Công thức: Giá_dự_báo = Giá_hiện_tại * e^(log_return_dự_báo)
predicted_prices = base_prices_test * np.exp(y_pred_log_return)

# 3. Tính giá thực tế (Actual Price) từ log_return thực tế
actual_prices = base_prices_test * np.exp(y_test)

# --- BƯỚC 4: ĐÁNH GIÁ VÀ VẼ BIỂU ĐỒ ---

# Tính các chỉ số đánh giá
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# Tính độ chính xác xu hướng (Directional Accuracy)
pred_direction = np.sign(y_pred_log_return)
true_direction = np.sign(y_test)
directional_acc = np.mean(pred_direction == true_direction) * 100

print("\n" + "="*40)
print("KẾT QUẢ ĐÁNH GIÁ (CUSTOM RANDOM FOREST)")
print("="*40)
print(f"RMSE (Sai số giá):       {rmse:.4f}")
print(f"MAPE (Sai số %):         {mape:.2f}%")
print(f"Directional Accuracy:    {directional_acc:.2f}%")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(14, 7))
dates_test = df.index[split_idx:] # Lấy ngày tháng để trục hoành đẹp hơn

plt.plot(dates_test, actual_prices, label='Giá Thực Tế (Actual)', color='#1f77b4', linewidth=2)
plt.plot(dates_test, predicted_prices, label='Giá Dự Báo (Predicted)', color='#ff7f0e', linestyle='--', linewidth=1.5)

plt.title(f'Dự báo Giá Cổ Phiếu - Random Forest Tự Build (MAPE: {mape:.2f}%)')
plt.xlabel('Thời gian')
plt.ylabel('Giá')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vẽ biểu đồ sai số (Residuals)
plt.figure(figsize=(14, 4))
errors = actual_prices - predicted_prices
plt.plot(dates_test, errors, color='red', alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.title('Biểu đồ Sai số (Thực tế - Dự báo)')
plt.ylabel('Chênh lệch giá')
plt.show()