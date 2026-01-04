import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

# Thêm path để import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.RandomForest import RandomForest

# Cấu hình trang
st.set_page_config(
    page_title="Dự Báo Giá Cổ Phiếu - Random Forest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Dự Báo Giá Cổ Phiếu với Random Forest</p>', unsafe_allow_html=True)

# Sidebar - Cấu hình
st.sidebar.header("Cấu hình Mô hình")

# Chọn cổ phiếu
ticker = st.sidebar.text_input("Mã cổ phiếu", value="AAPL", help="Nhập mã cổ phiếu (VD: AAPL, GOOGL, MSFT, TSLA)")

# Chọn khoảng thời gian
st.sidebar.subheader("Khoảng thời gian")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Từ ngày", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("Đến ngày", value=pd.to_datetime("2025-12-31"))

# Tham số mô hình
st.sidebar.subheader("Tham số Random Forest")
n_trees = st.sidebar.slider("Số lượng cây (n_trees)", min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider("Độ sâu tối đa (max_depth)", min_value=3, max_value=20, value=10)
min_samples_split = st.sidebar.slider("Min samples split", min_value=2, max_value=20, value=5)
train_ratio = st.sidebar.slider("Tỷ lệ Train (%)", min_value=70, max_value=90, value=85)

# Chọn đặc trưng
st.sidebar.subheader("Chọn Đặc trưng")
all_features = [
    'Close', 'log_return', 'sp500_return', 'vix_change', 'open_close_change',
    'obv', 'ma_5', 'ma_10', 'volatility_5', 'rsi_14', 'macd', 'stoch_k',
    'atr_14', 'bb_width', 'cci', 'adx'
]

default_features = ['Close', 'log_return', 'sp500_return', 'vix_change', 'open_close_change', 'obv', 'ma_5', 'volatility_5']
selected_features = st.sidebar.multiselect(
    "Chọn các đặc trưng để huấn luyện",
    options=all_features,
    default=default_features
)

# Nút huấn luyện
train_button = st.sidebar.button("Huấn luyện Mô hình", type="primary", use_container_width=True)

# Hàm tính toán đặc trưng
@st.cache_data
def load_and_prepare_data(ticker, start_date, end_date):
    """Tải và chuẩn bị dữ liệu"""
    # Tải dữ liệu cổ phiếu
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None, "Không tìm thấy dữ liệu cho mã cổ phiếu này"

    # Xử lý multi-level columns nếu có
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Tải dữ liệu vĩ mô
    macro_tickers = ['^VIX', '^GSPC', 'DX-Y.NYB']
    try:
        macro_data = yf.download(macro_tickers, start=start_date, end=end_date, progress=False)['Close']
        macro_data.columns = ['dxy_close', 'sp500_close', 'vix_close']
        df = df.join(macro_data, how='left')
        df[['dxy_close', 'sp500_close', 'vix_close']] = df[['dxy_close', 'sp500_close', 'vix_close']].ffill()

        # Đặc trưng vĩ mô
        df['vix_change'] = df['vix_close'].diff()
        df['sp500_return'] = df['sp500_close'].pct_change()
        df['dxy_return'] = df['dxy_close'].pct_change()
    except:
        df['vix_change'] = 0
        df['sp500_return'] = 0
        df['dxy_return'] = 0

    # Tính các đặc trưng
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility_5'] = df['log_return'].rolling(5).std()
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_200'] = df['Close'].rolling(200).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_change'] = (df['Close'] - df['Open']) / df['Open']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Williams %R
    df['williams_r'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # OBV
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr14 = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (abs(minus_dm).rolling(14).sum() / tr14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # Loại bỏ NaN
    df = df.dropna()

    # Tạo target
    df['target'] = df['log_return'].shift(-1)
    df = df.dropna()

    return df, None

def train_model(df, n_trees, max_depth, min_samples_split, train_ratio, selected_features):
    """Huấn luyện mô hình"""
    # Đảm bảo Close luôn có trong features
    if 'Close' not in selected_features:
        selected_features = ['Close'] + list(selected_features)

    # Lọc features có trong df
    available_features = [f for f in selected_features if f in df.columns]

    X = df[available_features].values
    y = df['target'].values

    # Chia dữ liệu
    split_idx = int(len(X) * train_ratio / 100)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Lấy giá Close
    close_col_idx = available_features.index('Close')
    base_prices_test = X_test[:, close_col_idx]

    # Huấn luyện
    rf = RandomForest(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_features=max(1, len(available_features) // 3),
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Dự báo
    y_pred_log_return = rf.predict(X_test)
    predicted_prices = base_prices_test * np.exp(y_pred_log_return)
    actual_prices = base_prices_test * np.exp(y_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    mae = mean_absolute_error(actual_prices, predicted_prices)

    # Directional Accuracy
    pred_direction = np.sign(y_pred_log_return)
    true_direction = np.sign(y_test)
    directional_acc = np.mean(pred_direction == true_direction) * 100

    return {
        'model': rf,
        'y_test': y_test,
        'y_pred': y_pred_log_return,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'dates_test': df.index[split_idx:],
        'rmse': rmse,
        'mape': mape,
        'mae': mae,
        'directional_acc': directional_acc,
        'split_idx': split_idx,
        'features': available_features
    }

# Tabs chính
tab1, tab2, tab3 = st.tabs(["Dữ liệu", "Kết quả Dự báo", "Phân tích Kỹ thuật"])

# Tab 1: Dữ liệu
with tab1:
    st.subheader(f"Dữ liệu cổ phiếu {ticker}")

    with st.spinner("Đang tải dữ liệu..."):
        df, error = load_and_prepare_data(ticker, str(start_date), str(end_date))

    if error:
        st.error(error)
    elif df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Số mẫu dữ liệu", f"{len(df):,}")
        with col2:
            st.metric("Giá hiện tại", f"${df['Close'].iloc[-1]:.2f}")
        with col3:
            change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            st.metric("Thay đổi", f"{change:.2f}%")
        with col4:
            st.metric("Số đặc trưng", f"{df.shape[1] - 1}")

        # Biểu đồ giá với Plotly
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ))
        fig.update_layout(
            title=f'Biểu đồ nến cổ phiếu {ticker}',
            xaxis_title='Thời gian',
            yaxis_title='Giá ($)',
            template='plotly_white',
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị dữ liệu
        st.subheader("Bảng dữ liệu")
        st.dataframe(df.tail(50), use_container_width=True)

        # Ma trận tương quan với Plotly
        st.subheader("Ma trận tương quan")
        numeric_cols = ['Close', 'Volume', 'log_return', 'rsi_14', 'macd', 'stoch_k', 'obv', 'atr_14', 'adx', 'cci']
        available_cols = [c for c in numeric_cols if c in df.columns]
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig_corr.update_layout(
                title="Ma trận tương quan giữa các đặc trưng",
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# Tab 2: Kết quả dự báo
with tab2:
    if train_button or 'results' in st.session_state:
        if train_button:
            if len(selected_features) < 2:
                st.error("Vui lòng chọn ít nhất 2 đặc trưng để huấn luyện mô hình")
            else:
                with st.spinner("Đang huấn luyện mô hình..."):
                    df, error = load_and_prepare_data(ticker, str(start_date), str(end_date))
                    if error:
                        st.error(error)
                    else:
                        results = train_model(df, n_trees, max_depth, min_samples_split, train_ratio, selected_features)
                        st.session_state['results'] = results
                        st.session_state['df'] = df
                        st.session_state['ticker'] = ticker

        if 'results' in st.session_state:
            results = st.session_state['results']

            st.success("Huấn luyện hoàn tất!")

            # Hiển thị đặc trưng đã sử dụng
            st.info(f"Đặc trưng sử dụng: {', '.join(results['features'])}")

            # Metrics
            st.subheader("Kết quả đánh giá")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("RMSE", f"{results['rmse']:.4f}", help="Root Mean Square Error")
            with col2:
                st.metric("MAPE", f"{results['mape']:.2f}%", help="Mean Absolute Percentage Error")
            with col3:
                st.metric("MAE", f"{results['mae']:.4f}", help="Mean Absolute Error")
            with col4:
                st.metric("Độ chính xác xu hướng", f"{results['directional_acc']:.2f}%", help="Directional Accuracy")

            # Biểu đồ chính - So sánh giá
            st.subheader("Biểu đồ So sánh Giá Thực tế vs Dự báo")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=results['dates_test'],
                y=results['actual_prices'],
                mode='lines',
                name='Giá Thực tế',
                line=dict(color='#1f77b4', width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=results['dates_test'],
                y=results['predicted_prices'],
                mode='lines',
                name='Giá Dự báo',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            fig1.update_layout(
                title=f'Dự báo Giá Cổ Phiếu {st.session_state.get("ticker", ticker)} - Random Forest (MAPE: {results["mape"]:.2f}%)',
                xaxis_title='Thời gian',
                yaxis_title='Giá ($)',
                template='plotly_white',
                height=500,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Biểu đồ sai số
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Biểu đồ Sai số")
                errors = results['actual_prices'] - results['predicted_prices']
                colors = ['green' if e > 0 else 'red' for e in errors]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=results['dates_test'],
                    y=errors,
                    marker_color=colors,
                    name='Sai số'
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                fig2.update_layout(
                    title='Sai số dự báo (Thực tế - Dự báo)',
                    xaxis_title='Thời gian',
                    yaxis_title='Chênh lệch giá ($)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                st.subheader("Phân phối Sai số")
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=50,
                    marker_color='#1f77b4',
                    name='Sai số'
                ))
                fig3.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
                fig3.update_layout(
                    title='Phân phối Sai số',
                    xaxis_title='Sai số ($)',
                    yaxis_title='Tần suất',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Scatter plot
            st.subheader("Actual vs Predicted")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=results['actual_prices'],
                y=results['predicted_prices'],
                mode='markers',
                marker=dict(color='#1f77b4', size=5, opacity=0.6),
                name='Dữ liệu'
            ))
            min_val = min(results['actual_prices'].min(), results['predicted_prices'].min())
            max_val = max(results['actual_prices'].max(), results['predicted_prices'].max())
            fig4.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ))
            fig4.update_layout(
                title='Giá Thực tế vs Giá Dự báo',
                xaxis_title='Giá Thực tế ($)',
                yaxis_title='Giá Dự báo ($)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Bảng so sánh
            st.subheader("Bảng So sánh Chi tiết")
            comparison_df = pd.DataFrame({
                'Ngày': results['dates_test'],
                'Giá Thực tế ($)': results['actual_prices'],
                'Giá Dự báo ($)': results['predicted_prices'],
                'Sai số ($)': results['actual_prices'] - results['predicted_prices'],
                'Sai số (%)': ((results['actual_prices'] - results['predicted_prices']) / results['actual_prices']) * 100
            })
            comparison_df = comparison_df.set_index('Ngày')
            st.dataframe(comparison_df.tail(30).style.format({
                'Giá Thực tế ($)': '${:.2f}',
                'Giá Dự báo ($)': '${:.2f}',
                'Sai số ($)': '${:.2f}',
                'Sai số (%)': '{:.2f}%'
            }), use_container_width=True)
    else:
        st.info("Vui lòng cấu hình và nhấn **Huấn luyện Mô hình** ở sidebar để bắt đầu")

# Tab 3: Phân tích Kỹ thuật
with tab3:
    st.subheader(f"Phân tích Kỹ thuật - {ticker}")

    with st.spinner("Đang tải dữ liệu..."):
        df, error = load_and_prepare_data(ticker, str(start_date), str(end_date))

    if error:
        st.error(error)
    elif df is not None:
        # Chọn chỉ báo - nằm trong trang
        st.markdown("#### Tùy chọn hiển thị")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            show_ma = st.checkbox("Moving Averages (MA)", value=True)
        with col_opt2:
            show_bb = st.checkbox("Bollinger Bands", value=True)
        with col_opt3:
            show_volume = st.checkbox("Volume", value=True)

        # Biểu đồ giá với các chỉ báo
        fig_main = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'Giá {ticker} với Chỉ báo Kỹ thuật', 'RSI (14)', 'MACD', 'Volume')
        )

        # Candlestick
        fig_main.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Moving Averages
        if show_ma:
            colors_ma = {'ma_20': '#FFA500', 'ma_50': '#00CED1', 'ma_200': '#FF1493'}
            for ma, color in colors_ma.items():
                if ma in df.columns:
                    fig_main.add_trace(go.Scatter(
                        x=df.index, y=df[ma],
                        mode='lines',
                        name=ma.upper(),
                        line=dict(color=color, width=1)
                    ), row=1, col=1)

        # Bollinger Bands
        if show_bb:
            fig_main.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(173, 216, 230, 0.8)', width=1)
            ), row=1, col=1)
            fig_main.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(173, 216, 230, 0.8)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.2)'
            ), row=1, col=1)

        # RSI
        fig_main.add_trace(go.Scatter(
            x=df.index, y=df['rsi_14'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1)
        ), row=2, col=1)
        fig_main.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig_main.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig_main.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.1)", line_width=0, row=2, col=1)

        # MACD
        colors_macd = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
        fig_main.add_trace(go.Bar(
            x=df.index, y=df['macd_hist'],
            name='MACD Hist',
            marker_color=colors_macd
        ), row=3, col=1)
        fig_main.add_trace(go.Scatter(
            x=df.index, y=df['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=1)
        ), row=3, col=1)
        fig_main.add_trace(go.Scatter(
            x=df.index, y=df['macd_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='orange', width=1)
        ), row=3, col=1)

        # Volume
        if show_volume:
            colors_vol = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
            fig_main.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                name='Volume',
                marker_color=colors_vol,
                opacity=0.7
            ), row=4, col=1)

        fig_main.update_layout(
            height=900,
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )
        fig_main.update_xaxes(title_text="Thời gian", row=4, col=1)
        fig_main.update_yaxes(title_text="Giá ($)", row=1, col=1)
        fig_main.update_yaxes(title_text="RSI", row=2, col=1)
        fig_main.update_yaxes(title_text="MACD", row=3, col=1)
        fig_main.update_yaxes(title_text="Volume", row=4, col=1)

        st.plotly_chart(fig_main, use_container_width=True)

        # Bảng tóm tắt chỉ báo
        st.subheader("Tóm tắt Chỉ báo Kỹ thuật")

        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi_status = "Quá mua" if latest['rsi_14'] > 70 else ("Quá bán" if latest['rsi_14'] < 30 else "Trung tính")
            st.metric("RSI (14)", f"{latest['rsi_14']:.2f}", rsi_status)
        with col2:
            macd_status = "Bullish" if latest['macd'] > latest['macd_signal'] else "Bearish"
            st.metric("MACD", f"{latest['macd']:.4f}", macd_status)
        with col3:
            stoch_status = "Quá mua" if latest['stoch_k'] > 80 else ("Quá bán" if latest['stoch_k'] < 20 else "Trung tính")
            st.metric("Stochastic %K", f"{latest['stoch_k']:.2f}", stoch_status)
        with col4:
            adx_status = "Xu hướng mạnh" if latest['adx'] > 25 else "Xu hướng yếu"
            st.metric("ADX", f"{latest['adx']:.2f}", adx_status)

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("ATR (14)", f"${latest['atr_14']:.2f}", "Biến động")
        with col6:
            cci_status = "Quá mua" if latest['cci'] > 100 else ("Quá bán" if latest['cci'] < -100 else "Trung tính")
            st.metric("CCI", f"{latest['cci']:.2f}", cci_status)
        with col7:
            bb_pos = (latest['Close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100
            bb_status = "Gần BB Upper" if bb_pos > 80 else ("Gần BB Lower" if bb_pos < 20 else "Giữa BB")
            st.metric("BB Position", f"{bb_pos:.1f}%", bb_status)
        with col8:
            williams_status = "Quá mua" if latest['williams_r'] > -20 else ("Quá bán" if latest['williams_r'] < -80 else "Trung tính")
            st.metric("Williams %R", f"{latest['williams_r']:.2f}", williams_status)

# Footer
st.markdown("---")

