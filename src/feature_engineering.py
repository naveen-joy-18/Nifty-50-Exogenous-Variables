import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

def compute_technical_indicators(df_nifty):
    """
    Computes comprehensive endogenous technical indicators from OHLCV data.
    """
    df = df_nifty.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']
    vol = df['Volume']
    
    # Returns
    df['ret_1d'] = close.pct_change()
    df['ret_5d'] = close.pct_change(5)
    df['log_ret'] = np.log(close / close.shift(1))
    
    # Moving averages
    df['sma_5'] = close.rolling(5).mean() / close - 1.0
    df['sma_20'] = close.rolling(20).mean() / close - 1.0
    df['ema_9'] = close.ewm(span=9, adjust=False).mean() / close - 1.0
    df['ema_21'] = close.ewm(span=21, adjust=False).mean() / close - 1.0
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + (2 * std20)
    df['bb_lower'] = sma20 - (2 * std20)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma20 + 1e-8)
    df['bb_pct_b'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean() / close
    
    # Candlestick Geometry
    candle_range = (high - low) + 1e-8
    df['upper_wick_ratio'] = (high - np.maximum(open_p, close)) / candle_range
    df['lower_wick_ratio'] = (np.minimum(open_p, close) - low) / candle_range
    df['real_body_ratio'] = (close - open_p).abs() / candle_range
    df['body_direction'] = np.sign(close - open_p)
    
    # Volatility
    df['hist_vol_10'] = df['ret_1d'].rolling(10).std() * np.sqrt(252)
    df['hist_vol_20'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    
    # Volume momentum
    df['vol_sma_20'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    
    return df


def compute_exogenous_features(market_data):
    """
    Computes macroeconomic, geopolitical, and cross-market exogenous features:
    - Brent Crude Oil returns & rolling momentum
    - USD/INR exchange rate dynamics & volatility
    - US 10-Year Treasury Yield changes & slope
    - Gold returns, safe-haven flight indicator, Gold/Oil ratio
    - S&P 500 overnight transmission return
    - India VIX fear index level & spike metric
    - Macro Stress Composite Index
    - Geopolitical Risk Proxy
    """
    exo_df = pd.DataFrame(index=market_data['NIFTY'].index)
    
    crude = market_data['CRUDE_OIL']['Close']
    usdinr = market_data['USD_INR']['Close']
    us10y = market_data['US_10Y']['Close']
    gold = market_data['GOLD']['Close']
    sp500 = market_data['SP500']['Close']
    vix = market_data['INDIA_VIX']['Close']
    nifty = market_data['NIFTY']['Close']
    
    # 1. Crude Oil Features
    exo_df['crude_ret_1d'] = crude.pct_change()
    exo_df['crude_ret_5d'] = crude.pct_change(5)
    exo_df['crude_vol_10'] = exo_df['crude_ret_1d'].rolling(10).std()
    exo_df['crude_sma20_ratio'] = crude / (crude.rolling(20).mean() + 1e-8) - 1.0
    
    # 2. USD/INR Currency Features
    exo_df['usdinr_ret_1d'] = usdinr.pct_change()
    exo_df['usdinr_ret_5d'] = usdinr.pct_change(5)
    exo_df['usdinr_vol_10'] = exo_df['usdinr_ret_1d'].rolling(10).std()
    exo_df['usdinr_sma20_ratio'] = usdinr / (usdinr.rolling(20).mean() + 1e-8) - 1.0
    
    # 3. US 10Y Yield Features
    exo_df['us10y_diff_1d'] = us10y.diff()
    exo_df['us10y_diff_5d'] = us10y.diff(5)
    exo_df['us10y_level'] = us10y / 10.0 # scale
    
    # 4. Gold & Geopolitical Safe Haven
    exo_df['gold_ret_1d'] = gold.pct_change()
    exo_df['gold_ret_5d'] = gold.pct_change(5)
    exo_df['gold_crude_ratio'] = (gold / (crude + 1e-8)).pct_change(5)
    
    # 5. Global S&P 500 Lead-Lag
    exo_df['sp500_ret_1d_lag1'] = sp500.pct_change().shift(1) # Overnight US lead
    exo_df['sp500_ret_5d'] = sp500.pct_change(5).shift(1)
    
    # 6. India VIX Sentiment
    exo_df['vix_level'] = vix / 100.0
    exo_df['vix_ret_1d'] = vix.pct_change()
    exo_df['vix_spike_5d'] = vix / (vix.rolling(20).mean() + 1e-8) - 1.0
    
    # 7. Composite Macro Stress Index
    # Scaled sum of currency pressure, oil volatility, yield surge, and VIX
    z_usdinr = (exo_df['usdinr_ret_1d'].abs() - exo_df['usdinr_ret_1d'].abs().rolling(20).mean()) / (exo_df['usdinr_ret_1d'].abs().rolling(20).std() + 1e-8)
    z_crude = (exo_df['crude_ret_1d'].abs() - exo_df['crude_ret_1d'].abs().rolling(20).mean()) / (exo_df['crude_ret_1d'].abs().rolling(20).std() + 1e-8)
    z_vix = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)
    exo_df['macro_stress_index'] = (0.3 * z_usdinr + 0.3 * z_crude + 0.4 * z_vix).clip(-3.0, 3.0)
    
    # 8. Geopolitical Risk Proxy (Co-movement of gold surge + oil surge during high VIX)
    exo_df['geopolitical_risk_proxy'] = np.where(
        (exo_df['gold_ret_5d'] > 0.01) & (exo_df['crude_ret_5d'] > 0.02) & (vix > 15), 1.0, 0.0
    )
    
    return exo_df


def build_aligned_multimodal_dataset(image_dict, market_data, seq_len=10):
    """
    Constructs the end-to-end synchronized dataset across:
    1. Candlestick Image paths
    2. Endogenous Technical Sequence (L x d_endo)
    3. Exogenous Macroeconomic Feature Vector (d_exo)
    4. Forecasting Targets: Next-Day Return & Direction
    """
    # 1. Compute Endogenous Features
    df_nifty = market_data['NIFTY'].copy()
    df_nifty_tech = compute_technical_indicators(df_nifty)
    
    # 2. Compute Exogenous Features
    df_exo = compute_exogenous_features(market_data)
    
    # 3. Define feature columns
    endo_cols = [
        'ret_1d', 'ret_5d', 'log_ret', 'sma_5', 'sma_20', 'ema_9', 'ema_21',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_width', 'bb_pct_b',
        'atr_14', 'upper_wick_ratio', 'lower_wick_ratio', 'real_body_ratio',
        'body_direction', 'hist_vol_10', 'hist_vol_20', 'vol_sma_20'
    ]
    
    exo_cols = [
        'crude_ret_1d', 'crude_ret_5d', 'crude_vol_10', 'crude_sma20_ratio',
        'usdinr_ret_1d', 'usdinr_ret_5d', 'usdinr_vol_10', 'usdinr_sma20_ratio',
        'us10y_diff_1d', 'us10y_diff_5d', 'us10y_level',
        'gold_ret_1d', 'gold_ret_5d', 'gold_crude_ratio',
        'sp500_ret_1d_lag1', 'sp500_ret_5d',
        'vix_level', 'vix_ret_1d', 'vix_spike_5d',
        'macro_stress_index', 'geopolitical_risk_proxy'
    ]
    
    # Target: Next-day return (%) and Direction (1/0)
    df_nifty_tech['target_next_ret'] = df_nifty_tech['Close'].pct_change().shift(-1) * 100.0
    df_nifty_tech['target_next_close'] = df_nifty_tech['Close'].shift(-1)
    df_nifty_tech['target_direction'] = (df_nifty_tech['target_next_ret'] > 0).astype(float)
    
    # Combine dataframes
    full_df = pd.concat([df_nifty_tech, df_exo], axis=1).dropna()
    
    # Align with images
    aligned_dates = []
    aligned_images = []
    aligned_endo_seqs = []
    aligned_exo_vecs = []
    aligned_targets = []
    aligned_target_prices = []
    aligned_current_prices = []
    aligned_directions = []
    
    # Scale features
    scaler_endo = StandardScaler()
    scaler_exo = StandardScaler()
    
    endo_matrix = scaler_endo.fit_transform(full_df[endo_cols].values)
    exo_matrix = scaler_exo.fit_transform(full_df[exo_cols].values)
    
    dates_list = [d.strftime('%Y-%m-%d') for d in full_df.index]
    
    for i in range(seq_len, len(dates_list)):
        dt_str = dates_list[i]
        if dt_str in image_dict:
            img_path = image_dict[dt_str]
            # Endogenous sequence over window [i-seq_len : i]
            endo_seq = endo_matrix[i - seq_len + 1 : i + 1] # shape (seq_len, d_endo)
            exo_vec = exo_matrix[i] # current exogenous snapshot (d_exo)
            
            target_ret = full_df['target_next_ret'].iloc[i]
            target_close = full_df['target_next_close'].iloc[i]
            curr_close = full_df['Close'].iloc[i]
            direction = full_df['target_direction'].iloc[i]
            
            if not np.isnan(target_ret) and not np.isnan(target_close):
                aligned_dates.append(dt_str)
                aligned_images.append(img_path)
                aligned_endo_seqs.append(endo_seq)
                aligned_exo_vecs.append(exo_vec)
                aligned_targets.append(target_ret)
                aligned_target_prices.append(target_close)
                aligned_current_prices.append(curr_close)
                aligned_directions.append(direction)
                
    aligned_data = {
        'dates': aligned_dates,
        'images': aligned_images,
        'endo_seqs': np.array(aligned_endo_seqs),
        'exo_vecs': np.array(aligned_exo_vecs),
        'targets': np.array(aligned_targets),
        'target_prices': np.array(aligned_target_prices),
        'current_prices': np.array(aligned_current_prices),
        'directions': np.array(aligned_directions),
        'endo_cols': endo_cols,
        'exo_cols': exo_cols,
        'scaler_endo': scaler_endo,
        'scaler_exo': scaler_exo,
        'full_df': full_df
    }
    
    print(f"[Dataset Alignment] Successfully synchronized {len(aligned_dates)} multimodal samples.")
    print(f"  - Endogenous sequence shape: {aligned_data['endo_seqs'].shape}")
    print(f"  - Exogenous vector shape: {aligned_data['exo_vecs'].shape}")
    print(f"  - Target return shape: {aligned_data['targets'].shape}")
    
    return aligned_data
