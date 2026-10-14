
import warnings
warnings.filterwarnings("ignore")

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


# =====================================================
# 1. CẤU HÌNH WEBSITE
# =====================================================

st.set_page_config(
    page_title="Dự báo giá cổ phiếu",
    page_icon="📈",
    layout="wide"
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.markdown("""
<style>
.main {
    background-color: #f7f8fc;
}
.hero {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(
        120deg, #21134f, #4b2e83, #e5b83d
    );
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
}
.hero p {
    color: #f4efff;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e7e8ef;
    padding: 15px;
    border-radius: 12px;
}
.notice {
    padding: 14px;
    border-radius: 8px;
    background: #fff7df;
    border-left: 5px solid #e5b83d;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📈 DỰ BÁO GIÁ CỔ PHIẾU</h1>
    <p>
        Phân tích dữ liệu lịch sử và dự báo giá cổ phiếu
        trong tối đa 10 phiên giao dịch tiếp theo bằng LSTM.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
<b>Lưu ý:</b> Đây là kết quả ước tính từ mô hình học máy,
không phải lời khuyên đầu tư hoặc cam kết lợi nhuận.
Giá thực tế có thể khác đáng kể so với dự báo.
</div>
""", unsafe_allow_html=True)

st.write("")


# =====================================================
# 2. TẢI DỮ LIỆU YAHOO FINANCE
# =====================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_stock_data(ticker, start_date, end_date):

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Xử lý cả cột thông thường và MultiIndex của yfinance
    if isinstance(df.columns, pd.MultiIndex):

        close_columns = [
            col for col in df.columns
            if any(str(level).lower() == "close" for level in col)
        ]

        if not close_columns:
            return pd.DataFrame()

        close = df[close_columns[0]]

    else:

        if "Close" not in df.columns:
            return pd.DataFrame()

        close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    result = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    })

    result = result.dropna()
    result = result[result["Close"] > 0]
    result.index = pd.to_datetime(result.index)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    return result


# =====================================================
# 3. CHUYỂN DỮ LIỆU THÀNH CHUỖI 60 PHIÊN
# =====================================================

def create_sequences(data, window_size=60):

    X, y = [], []

    for i in range(window_size, len(data)):

        X.append(data[i - window_size:i])
        y.append(data[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return X.reshape(-1, window_size, 1), y


# =====================================================
# 4. HUẤN LUYỆN LSTM VÀ DỰ BÁO
# =====================================================

def forecast_stock(prices, forecast_days, epochs):

    values = prices[["Close"]].values.astype(np.float32)

    if len(values) < 100:
        raise ValueError(
            "Mã cổ phiếu có quá ít dữ liệu. "
            "Ứng dụng cần ít nhất khoảng 100 phiên giao dịch."
        )

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)

    # Tạo dữ liệu đầu vào với 60 phiên
    X_train, y_train = create_sequences(
        scaled_data,
        window_size=60
    )

    if len(X_train) < 20:
        raise ValueError("Không đủ dữ liệu để huấn luyện mô hình.")

    # Chia phần cuối dữ liệu làm tập kiểm tra quá trình huấn luyện
    val_size = max(1, int(len(X_train) * 0.1))

    X_fit = X_train[:-val_size]
    y_fit = y_train[:-val_size]

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    if len(X_fit) == 0:
        raise ValueError("Không đủ dữ liệu huấn luyện.")

    # Xây dựng mô hình LSTM hai lớp theo source ban đầu
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Dự báo nối tiếp từng phiên
    last_sequence = scaled_data[-60:].reshape(1, 60, 1)

    predictions_scaled = []

    for _ in range(forecast_days):

        next_prediction = model.predict(
            last_sequence,
            verbose=0
        )

        next_value = float(next_prediction[0, 0])
        predictions_scaled.append(next_value)

        # Bỏ phiên cũ nhất và thêm giá dự báo mới
        last_sequence = np.concatenate(
            [
                last_sequence[:, 1:, :],
                np.array(
                    next_value,
                    dtype=np.float32
                ).reshape(1, 1, 1)
            ],
            axis=1
        )

    # Chuyển kết quả về đơn vị giá ban đầu
    predictions = scaler.inverse_transform(
        np.array(
            predictions_scaled,
            dtype=np.float32
        ).reshape(-1, 1)
    ).flatten()

    predictions = np.maximum(predictions, 0)

    return predictions


# =====================================================
# 5. TẠO NGÀY DỰ BÁO
# =====================================================

def get_future_dates(last_date, number_of_days):

    dates = []
    current_date = pd.Timestamp(last_date).normalize()

    while len(dates) < number_of_days:

        current_date += pd.Timedelta(days=1)

        # Bỏ thứ Bảy và Chủ nhật
        if current_date.weekday() < 5:
            dates.append(current_date)

    return dates


# =====================================================
# 6. THANH ĐIỀU KHIỂN
# =====================================================

with st.sidebar:

    st.header("⚙️ Cài đặt dự báo")

    preset = st.selectbox(
        "Chọn mã chứng khoán gợi ý",
        [
            "Nhập mã khác",
            "VNM.VN",
            "FPT.VN",
            "HPG.VN",
            "VCB.VN",
            "VIC.VN",
            "MWG.VN",
            "SSI.VN",
            "ACB.VN",
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "GOOGL"
        ]
    )

    default_ticker = (
        "" if preset == "Nhập mã khác"
        else preset
    )

    ticker_input = st.text_input(
        "Mã chứng khoán",
        value=default_ticker,
        placeholder="Ví dụ: FPT.VN hoặc AAPL",
        help=(
            "Nhập mã theo Yahoo Finance. "
            "Cổ phiếu Việt Nam thường dùng hậu tố .VN."
        )
    )

    ticker = ticker_input.strip().upper()

    history_years = st.selectbox(
        "Thời gian dữ liệu lịch sử",
        options=[3, 5, 8, 10],
        index=1,
        format_func=lambda x: f"{x} năm"
    )

    forecast_days = st.slider(
        "Muốn dự báo bao nhiêu phiên?",
        min_value=1,
        max_value=10,
        value=10,
        step=1
    )

    epochs = st.slider(
        "Số epoch huấn luyện",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    run_button = st.button(
        "🚀 BẮT ĐẦU DỰ BÁO",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Ngày dự báo bỏ thứ Bảy và Chủ nhật. "
        "Lịch nghỉ lễ của từng sàn chưa được xử lý đầy đủ."
    )


# =====================================================
# 7. CHẠY DỰ BÁO
# =====================================================

if run_button:

    if not ticker:
        st.error("Vui lòng nhập mã chứng khoán.")
        st.stop()

    end_date = datetime.now().date() + timedelta(days=1)

    start_date = (
        end_date - timedelta(days=history_years * 365)
    )

    try:

        with st.spinner(
            f"Đang tải dữ liệu của {ticker}..."
        ):

            stock_df = download_stock_data(
                ticker,
                start_date.isoformat(),
                end_date.isoformat()
            )

        if stock_df.empty:
            st.error(
                f"Không tìm thấy dữ liệu cho mã {ticker}. "
                "Hãy kiểm tra lại mã hoặc thử mã khác."
            )
            st.stop()

        if len(stock_df) < 100:
            st.error(
                "Dữ liệu lịch sử quá ngắn. "
                "Hãy chọn mã có ít nhất khoảng 100 phiên giao dịch."
            )
            st.stop()

        with st.spinner(
            "Mô hình LSTM đang học dữ liệu và dự báo..."
        ):

            predicted_prices = forecast_stock(
                stock_df,
                forecast_days,
                epochs
            )

    except Exception as error:

        st.error(f"Không thể dự báo: {error}")

        st.info(
            "Hãy kiểm tra kết nối Internet, mã chứng khoán "
            "và log lỗi nếu chạy trên Streamlit Cloud."
        )

        st.stop()

    # =================================================
    # 8. TÍNH PHẦN TRĂM TĂNG / GIẢM
    # =================================================

    last_date = stock_df.index[-1]
    current_price = float(stock_df["Close"].iloc[-1])

    final_predicted_price = float(predicted_prices[-1])

    total_change_pct = (
        (final_predicted_price / current_price) - 1
    ) * 100

    future_dates = get_future_dates(
        last_date,
        forecast_days
    )

    rows = []
    previous_price = current_price

    for i, predicted_price in enumerate(predicted_prices):

        predicted_price = float(predicted_price)

        # Thay đổi so với phiên trước
        daily_change_pct = (
            (predicted_price / previous_price) - 1
        ) * 100

        # Thay đổi so với giá hiện tại
        cumulative_change_pct = (
            (predicted_price / current_price) - 1
        ) * 100

        rows.append({
            "Phiên": i + 1,
            "Ngày dự báo": future_dates[i].strftime("%d/%m/%Y"),
            "Giá dự báo": predicted_price,
            "Thay đổi phiên (%)": daily_change_pct,
            "Tăng/giảm từ hiện tại (%)": cumulative_change_pct
        })

        previous_price = predicted_price

    forecast_df = pd.DataFrame(rows)

    # =================================================
    # 9. HIỂN THỊ KẾT QUẢ
    # =================================================

    st.subheader(f"📊 Kết quả dự báo: {ticker}")

    st.caption(
        f"Giá đóng cửa gần nhất: "
        f"{pd.Timestamp(last_date).strftime('%d/%m/%Y')} "
        f"| Số phiên lịch sử: {len(stock_df):,}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Giá đóng cửa gần nhất",
        f"{current_price:,.2f}"
    )

    col2.metric(
        f"Giá dự báo sau {forecast_days} phiên",
        f"{final_predicted_price:,.2f}"
    )

    col3.metric(
        "Tỷ lệ tăng/giảm dự báo",
        f"{total_change_pct:+.2f}%",
        delta=f"{total_change_pct:+.2f}%",
        delta_color="normal"
    )

    if total_change_pct > 0:

        st.success(
            f"📈 Mô hình dự báo giá có thể tăng "
            f"{total_change_pct:.2f}% sau {forecast_days} phiên."
        )

    elif total_change_pct < 0:

        st.warning(
            f"📉 Mô hình dự báo giá có thể giảm "
            f"{abs(total_change_pct):.2f}% sau {forecast_days} phiên."
        )

    else:

        st.info(
            "Mô hình dự báo giá gần như không thay đổi."
        )

    # =================================================
    # 10. BẢNG DỰ BÁO CHI TIẾT
    # =================================================

    st.markdown("### 📋 Bảng dự báo từng phiên")

    display_df = forecast_df.copy()

    display_df["Giá dự báo"] = display_df[
        "Giá dự báo"
    ].map(lambda x: f"{x:,.2f}")

    display_df["Thay đổi phiên (%)"] = display_df[
        "Thay đổi phiên (%)"
    ].map(lambda x: f"{x:+.2f}%")

    display_df["Tăng/giảm từ hiện tại (%)"] = display_df[
        "Tăng/giảm từ hiện tại (%)"
    ].map(lambda x: f"{x:+.2f}%")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # =================================================
    # 11. BIỂU ĐỒ GIÁ LỊCH SỬ VÀ DỰ BÁO
    # =================================================

    st.markdown("### 📈 Biểu đồ giá lịch sử và dự báo")

    history = stock_df.tail(120)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Giá đóng cửa lịch sử"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[last_date] + future_dates,
            y=[current_price] + list(predicted_prices),
            mode="lines+markers",
            name="Giá dự báo",
            line=dict(dash="dash", width=3)
        )
    )

    fig.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Giá cổ phiếu",
        hovermode="x unified",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # 12. TẢI KẾT QUẢ CSV
    # =================================================

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Tải bảng dự báo xuống CSV",
        data=csv_data,
        file_name=(
            f"du_bao_{ticker.replace('.', '_')}_"
            f"{forecast_days}_phien.csv"
        ),
        mime="text/csv"
    )

    # =================================================
    # 13. GIẢI THÍCH CÔNG THỨC
    # =================================================

    with st.expander("ℹ️ Cách tính tỷ lệ tăng/giảm"):

        st.write(
            "Tỷ lệ thay đổi (%) = "
            "(Giá dự báo / Giá đóng cửa gần nhất - 1) × 100."
        )

        st.write(
            "Kết quả dương biểu thị giá dự báo cao hơn giá hiện tại; "
            "kết quả âm biểu thị giá dự báo thấp hơn giá hiện tại."
        )

        st.write(
            "Đây là thay đổi giá ước tính, chưa trừ phí giao dịch, "
            "thuế, chênh lệch giá mua/bán hoặc các chi phí đầu tư khác."
        )

else:

    st.subheader("👋 Bắt đầu dự báo cổ phiếu")

    st.write("""
    1. Chọn hoặc nhập mã chứng khoán ở thanh bên trái.
    2. Chọn thời gian dữ liệu lịch sử.
    3. Chọn số phiên muốn dự báo từ 1 đến 10.
    4. Nhấn **BẮT ĐẦU DỰ BÁO**.
    """)

    st.markdown("#### Một số mã có thể thử")

    st.write(
        "**Việt Nam:** `VNM.VN`, `FPT.VN`, `HPG.VN`, "
        "`VCB.VN`, `MWG.VN`"
    )

    st.write(
        "**Quốc tế:** `AAPL`, `MSFT`, `NVDA`, `TSLA`, `GOOGL`"
    )

    st.caption(
        "Không phải mọi mã đều có dữ liệu trên Yahoo Finance. "
        "Dự báo có thể sai và không đảm bảo lợi nhuận."
    )
