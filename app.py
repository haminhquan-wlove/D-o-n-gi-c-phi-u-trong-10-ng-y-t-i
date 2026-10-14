from __future__ import annotations

import os
import random
import re
import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Dự báo cổ phiếu 10 phiên",
    page_icon="📈",
    layout="wide",
)

# Keep TensorFlow logs quieter on Streamlit Community Cloud.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# The source notebook uses an LSTM with two layers, 60 prior trading days,
# MinMax scaling, and recursive forecasting for 10 future observations.
WINDOW_SIZE = 60
FORECAST_DAYS = 10
DATA_PERIOD = "10y"


@st.cache_data(ttl=3600, show_spinner=False)
def download_history(symbol: str) -> pd.DataFrame:
    """Download daily close-price history from Yahoo Finance."""
    frame = yf.download(
        tickers=symbol,
        period=DATA_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise ValueError(
            "Không lấy được dữ liệu. Hãy kiểm tra mã chứng khoán hoặc thử lại sau."
        )

    # yfinance versions differ: columns may be a normal Index or a MultiIndex.
    close_column = None
    for column in frame.columns:
        parts = column if isinstance(column, tuple) else (column,)
        if any(str(part).strip().lower() == "close" for part in parts):
            close_column = column
            break
    if close_column is None:
        raise ValueError("Nguồn dữ liệu không có cột giá đóng cửa (Close).")

    close = frame[close_column]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = close.rename("Close").to_frame()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].dropna()
    out = out[out["Close"] > 0]
    return out.astype(float)


def make_sequences(array: np.ndarray, window_size: int = WINDOW_SIZE):
    x_values, y_values = [], []
    for i in range(window_size, len(array)):
        x_values.append(array[i - window_size : i])
        y_values.append(array[i])
    return np.asarray(x_values), np.asarray(y_values)


def build_lstm(tf):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


@st.cache_data(ttl=3600, show_spinner=False)
def generate_forecast(symbol: str, cache_key: str):
    """Train/evaluate an LSTM and produce a cached 10-session forecast."""
    # Imported only when the user requests a forecast to reduce initial app load.
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import MinMaxScaler

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    history = download_history(symbol)
    if len(history) < 180:
        raise ValueError(
            f"Mã này chỉ có {len(history)} phiên dữ liệu. Cần tối thiểu 180 phiên "
            "để xây dựng và kiểm tra mô hình; hãy thử mã có lịch sử dài hơn."
        )

    prices = history["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)
    split = int(len(prices) * 0.8)
    train_prices, test_prices = prices[:split], prices[split:]
    if len(train_prices) <= WINDOW_SIZE + 20 or len(test_prices) < 5:
        raise ValueError("Dữ liệu chưa đủ để chia tập huấn luyện và kiểm tra.")

    # Stage 1: chronological holdout evaluation, matching the user's source code.
    eval_scaler = MinMaxScaler()
    train_scaled = eval_scaler.fit_transform(train_prices)
    test_scaled = eval_scaler.transform(test_prices)
    x_train_all, y_train_all = make_sequences(train_scaled)

    val_size = max(1, int(0.1 * len(x_train_all)))
    if len(x_train_all) - val_size < 10:
        raise ValueError("Không đủ chuỗi dữ liệu để huấn luyện mô hình.")
    x_train = x_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    x_val = x_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    tf.keras.backend.clear_session()
    eval_model = build_lstm(tf)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    eval_fit_history = eval_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

    combined = np.vstack((train_scaled[-WINDOW_SIZE:], test_scaled))
    x_test, y_test = make_sequences(combined)
    test_prediction_scaled = eval_model.predict(x_test, verbose=0)
    test_prediction = eval_scaler.inverse_transform(test_prediction_scaled).ravel()
    test_actual = eval_scaler.inverse_transform(y_test).ravel()
    rmse = float(np.sqrt(mean_squared_error(test_actual, test_prediction)))
    mae = float(mean_absolute_error(test_actual, test_prediction))
    r2 = float(r2_score(test_actual, test_prediction)) if len(test_actual) > 1 else float("nan")

    # Stage 2: refit on all available observations before projecting the future.
    # Use the best validation epoch count chosen in Stage 1, then train on all sequences.
    tf.keras.backend.clear_session()
    final_scaler = MinMaxScaler()
    all_scaled = final_scaler.fit_transform(prices)
    x_all, y_all = make_sequences(all_scaled)
    final_val_size = max(1, int(0.1 * len(x_all)))
    final_model = build_lstm(tf)
    best_epoch_count = int(np.argmin(eval_fit_history.history["val_loss"]) + 1)
    final_model.fit(
        x_all,
        y_all,
        epochs=best_epoch_count,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )

    rolling_window = all_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    forecast_scaled = []
    for _ in range(FORECAST_DAYS):
        next_value = final_model.predict(rolling_window, verbose=0)
        forecast_scaled.append(float(next_value[0, 0]))
        rolling_window = np.concatenate(
            [rolling_window[:, 1:, :], next_value.reshape(1, 1, 1)], axis=1
        )

    forecast_prices = final_scaler.inverse_transform(
        np.asarray(forecast_scaled, dtype=np.float32).reshape(-1, 1)
    ).ravel()
    forecast_prices = np.maximum(forecast_prices, 0)

    latest_price = float(prices[-1, 0])
    prior = np.concatenate([[latest_price], forecast_prices[:-1]])
    daily_change = (forecast_prices / prior - 1) * 100
    cumulative_change = (forecast_prices / latest_price - 1) * 100

    last_date = pd.Timestamp(history.index[-1]).normalize()
    # Weekdays are approximated; exchange-specific public holidays are not removed.
    forecast_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1), periods=FORECAST_DAYS
    )
    forecast_table = pd.DataFrame(
        {
            "Phiên dự báo": np.arange(1, FORECAST_DAYS + 1),
            "Ngày (ước tính)": forecast_dates.strftime("%d/%m/%Y"),
            "Giá dự báo": forecast_prices,
            "% thay đổi so với phiên trước": daily_change,
            "% so với giá hiện tại": cumulative_change,
        }
    )

    return {
        "history": history,
        "latest_price": latest_price,
        "latest_date": last_date.strftime("%d/%m/%Y"),
        "forecast_table": forecast_table,
        "forecast_prices": forecast_prices,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_actual": test_actual,
        "test_prediction": test_prediction,
        "split_date": pd.Timestamp(history.index[split]).strftime("%d/%m/%Y"),
    }


def format_price(value: float) -> str:
    return f"{value:,.2f}"


st.title("📈 Dự báo giá cổ phiếu trong 10 phiên giao dịch")
st.write(
    "Nhập mã chứng khoán để xem dự báo giá bằng mô hình LSTM, mức tăng/giảm "
    "ước tính theo phần trăm và biểu đồ diễn biến dự báo."
)

with st.sidebar:
    st.header("Chọn cổ phiếu")
    symbol_input = st.text_input(
        "Mã chứng khoán Yahoo Finance",
        value="VNM.VN",
        help=(
            "Ví dụ Việt Nam: VNM.VN, FPT.VN, VCB.VN, HPG.VN. "
            "Ví dụ quốc tế: AAPL, MSFT, NVDA, TSLA. Mã phải tồn tại trên Yahoo Finance."
        ),
        max_chars=30,
    )
    st.caption("Dữ liệu ngày từ Yahoo Finance; mô hình dùng giá Close lịch sử.")
    run_forecast = st.button("Dự báo 10 phiên", type="primary", use_container_width=True)
    st.divider()
    st.caption("Cửa sổ dữ liệu: tối đa 10 năm gần nhất. Dữ liệu được lưu đệm 1 giờ.")

symbol = symbol_input.strip().upper()
if not symbol:
    st.info("Nhập mã chứng khoán ở thanh bên trái để bắt đầu.")
    st.stop()
if not re.fullmatch(r"[A-Z0-9.^=_-]{1,30}", symbol):
    st.error("Mã chứng khoán có ký tự không hợp lệ. Hãy nhập lại mã Yahoo Finance.")
    st.stop()

if run_forecast:
    # Avoid accidentally showing the previous ticker's result if this request fails.
    st.session_state.pop("forecast_result", None)
    st.session_state.pop("forecast_symbol", None)
    try:
        with st.spinner(f"Đang tải dữ liệu và huấn luyện LSTM cho {symbol}. Lần đầu có thể mất vài phút..."):
            raw_history = download_history(symbol)
            if raw_history.empty:
                raise ValueError("Không tìm thấy dữ liệu.")
            cache_key = f"{symbol}_{raw_history.index[-1]}_{len(raw_history)}"
            result = generate_forecast(symbol, cache_key)
        st.session_state["forecast_result"] = result
        st.session_state["forecast_symbol"] = symbol
    except Exception as exc:
        st.error(f"Không thể tạo dự báo cho {symbol}: {exc}")
        st.info(
            "Kiểm tra mã chứng khoán, kết nối mạng và thử lại. Một số mã không có "
            "dữ liệu trên Yahoo Finance hoặc có lịch sử quá ngắn."
        )

if "forecast_result" in st.session_state:
    result = st.session_state["forecast_result"]
    shown_symbol = st.session_state.get("forecast_symbol", symbol)
    start_price = result["latest_price"]
    end_price = float(result["forecast_prices"][-1])
    total_pct = (end_price / start_price - 1) * 100 if start_price else 0.0

    st.subheader(f"Kết quả dự báo: {shown_symbol}")
    st.caption(
        f"Giá đóng cửa gần nhất trong dữ liệu: {result['latest_date']}. "
        "Dự báo tính từ phiên tiếp theo có dữ liệu."
    )
    a, b, c = st.columns(3)
    a.metric("Giá đóng cửa gần nhất", format_price(start_price))
    b.metric("Giá dự báo phiên thứ 10", format_price(end_price))
    c.metric(
        "Lãi/lỗ ước tính sau 10 phiên",
        f"{total_pct:+.2f}%",
        delta=f"{total_pct:+.2f}%",
        delta_color="normal",
    )
    if total_pct > 0:
        st.success(f"Theo mô hình, giá phiên thứ 10 cao hơn khoảng {total_pct:.2f}% so với giá đóng cửa gần nhất.")
    elif total_pct < 0:
        st.warning(f"Theo mô hình, giá phiên thứ 10 thấp hơn khoảng {abs(total_pct):.2f}% so với giá đóng cửa gần nhất.")
    else:
        st.info("Theo mô hình, giá phiên thứ 10 gần như không đổi so với giá đóng cửa gần nhất.")

    st.subheader("Bảng dự báo 10 phiên")
    display_table = result["forecast_table"].copy()
    display_table["Giá dự báo"] = display_table["Giá dự báo"].map(format_price)
    display_table["% thay đổi so với phiên trước"] = display_table["% thay đổi so với phiên trước"].map(lambda x: f"{x:+.2f}%")
    display_table["% so với giá hiện tại"] = display_table["% so với giá hiện tại"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.subheader("Biểu đồ lịch sử và dự báo")
    historical = result["history"].tail(90)["Close"]
    chart_hist = pd.DataFrame({"Ngày": historical.index, "Giá đóng cửa": historical.to_numpy()}).set_index("Ngày")
    forecast_raw = result["forecast_table"]
    forecast_dates = pd.to_datetime(forecast_raw["Ngày (ước tính)"], format="%d/%m/%Y")
    chart_forecast = pd.DataFrame(
        {"Ngày": forecast_dates, "Giá dự báo": result["forecast_prices"]}
    ).set_index("Ngày")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(chart_hist.index, chart_hist["Giá đóng cửa"], label="Giá đóng cửa lịch sử", linewidth=2)
    ax.plot(chart_forecast.index, chart_forecast["Giá dự báo"], label="LSTM dự báo 10 phiên", marker="o", linewidth=2)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá (theo đơn vị dữ liệu Yahoo Finance)")
    ax.set_title(f"{shown_symbol}: lịch sử gần nhất và giá dự báo")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Độ sai số trên tập kiểm tra lịch sử"):
        st.write(
            "Các chỉ số này đo kết quả trên phần dữ liệu lịch sử được giữ lại để kiểm tra, "
            "không phải xác suất dự báo đúng trong tương lai."
        )
        x, y, z = st.columns(3)
        x.metric("RMSE", format_price(result["rmse"]))
        y.metric("MAE", format_price(result["mae"]))
        z.metric("R²", f"{result['r2']:.3f}" if np.isfinite(result["r2"]) else "Không xác định")

    st.caption(
        "Ngày dự báo là ngày làm việc ước tính, chỉ loại trừ thứ Bảy và Chủ nhật; "
        "chưa hiệu chỉnh lịch nghỉ lễ riêng của từng sàn. Giá dự báo là kết quả mô hình "
        "LSTM dùng giá đóng cửa quá khứ, không đảm bảo giá thực tế sẽ diễn biến như vậy."
    )
else:
    st.info("Nhập hoặc giữ mã ở thanh bên trái, sau đó nhấn **Dự báo 10 phiên**.")

st.divider()
st.caption(
    "Lưu ý: Đây là công cụ tham khảo/học tập, không phải khuyến nghị mua bán chứng khoán. "
    "Giá cổ phiếu có thể biến động mạnh; hãy tự đánh giá rủi ro trước khi ra quyết định."
)
