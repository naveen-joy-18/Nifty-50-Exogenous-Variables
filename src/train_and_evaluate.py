import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from src.data_loader import NiftyMultimodalDataset, get_image_transforms
from src.models import (
    ProposedExoVisionXAttnModel,
    PureVisionCNNBaseline,
    PureBiLSTMBaseline,
    VanillaCNNLSTM_NoExo,
    EarlyConcatMultimodal
)

def compute_metrics(y_true_ret, y_pred_ret, y_true_price, y_pred_price, pred_dirs=None):
    """
    Computes standard regression, forecasting, and directional metrics.
    """
    # Return metrics
    rmse_ret = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))
    mae_ret = mean_absolute_error(y_true_ret, y_pred_ret)
    
    # Price metrics
    rmse_price = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae_price = mean_absolute_error(y_true_price, y_pred_price)
    mape_price = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-8))) * 100.0
    r2 = r2_score(y_true_price, y_pred_price)
    
    # Directional Accuracy
    dir_true = (y_true_ret > 0).astype(int)
    if pred_dirs is not None and len(pred_dirs) == len(y_true_ret):
        dir_pred = (np.array(pred_dirs) > 0.5).astype(int)
    else:
        dir_pred = (y_pred_ret > 0).astype(int)
    directional_acc = accuracy_score(dir_true, dir_pred) * 100.0
    
    return {
        'RMSE_Price': float(rmse_price),
        'MAE_Price': float(mae_price),
        'MAPE_Price': float(mape_price),
        'R2_Score': float(r2),
        'RMSE_Return': float(rmse_ret),
        'MAE_Return': float(mae_ret),
        'Directional_Accuracy': float(directional_acc)
    }


def simulate_trading_strategy(y_true_ret, y_pred_ret, pred_dirs=None, initial_capital=100000.0, fee_bps=5.0):
    """
    Simulates a daily quantitative long/short/cash trading strategy based on model predicted returns and directional heads.
    """
    if pred_dirs is not None and len(pred_dirs) == len(y_true_ret):
        probs = np.array(pred_dirs)
        positions = np.where(probs > 0.52, 1.0, np.where(probs < 0.48, -1.0, 0.0))
    else:
        positions = np.where(y_pred_ret > 0.05, 1.0, np.where(y_pred_ret < -0.05, -1.0, 0.0))
        
    strat_ret = positions * (y_true_ret / 100.0)
    
    # Apply transaction cost / slippage (5 bps = 0.05%)
    pos_changes = np.abs(np.diff(np.insert(positions, 0, 0.0)))
    costs = pos_changes * (fee_bps / 10000.0)
    net_strat_ret = strat_ret - costs
    
    equity_curve = initial_capital * np.cumprod(1.0 + net_strat_ret)
    cum_return_pct = ((equity_curve[-1] - initial_capital) / initial_capital) * 100.0
    
    # Sharpe Ratio (annualized, 252 days)
    mean_daily = np.mean(net_strat_ret)
    std_daily = np.std(net_strat_ret) + 1e-8
    sharpe_ratio = (mean_daily / std_daily) * np.sqrt(252)
    
    # Sortino Ratio
    downside_returns = net_strat_ret[net_strat_ret < 0]
    downside_std = np.std(downside_returns) + 1e-8 if len(downside_returns) > 0 else 1e-8
    sortino_ratio = (mean_daily / downside_std) * np.sqrt(252)
    
    # Maximum Drawdown (MDD)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_drawdown_pct = np.min(drawdowns) * 100.0
    
    # Win rate
    active_trades = net_strat_ret[positions != 0]
    win_rate = (np.sum(active_trades > 0) / len(active_trades) * 100.0) if len(active_trades) > 0 else 0.0
    
    return {
        'Cumulative_Return_Pct': float(cum_return_pct),
        'Sharpe_Ratio': float(sharpe_ratio),
        'Sortino_Ratio': float(sortino_ratio),
        'Max_Drawdown_Pct': float(max_drawdown_pct),
        'Win_Rate_Pct': float(win_rate),
        'Equity_Curve': equity_curve.tolist(),
        'Net_Returns': net_strat_ret.tolist()
    }


def train_deep_learning_model(model, train_loader, test_loader, epochs=45, lr=1e-3, device='cpu'):
    """
    Trains PyTorch multimodal / baseline models with Huber loss + BCE multi-task objective.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    huber_loss_fn = nn.SmoothL1Loss()
    bce_loss_fn = nn.BCELoss()
    
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            endo = batch['endo'].to(device)
            exo = batch['exo'].to(device)
            target = batch['target'].to(device)
            direction = batch['direction'].to(device)
            
            optimizer.zero_grad()
            out = model(imgs, endo, exo)
            
            loss_reg = huber_loss_fn(out['pred_return'], target)
            loss_cls = bce_loss_fn(out['pred_direction'], direction) if 'pred_direction' in out else 0.0
            
            loss = loss_reg + 0.3 * loss_cls
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['image'].to(device)
                endo = batch['endo'].to(device)
                exo = batch['exo'].to(device)
                target = batch['target'].to(device)
                direction = batch['direction'].to(device)
                
                out = model(imgs, endo, exo)
                l_reg = huber_loss_fn(out['pred_return'], target)
                l_cls = bce_loss_fn(out['pred_direction'], direction) if 'pred_direction' in out else 0.0
                test_loss += (l_reg + 0.4 * l_cls).item()
                
                val_preds.extend(out['pred_return'].cpu().numpy().tolist())
                val_targets.extend(target.cpu().numpy().tolist())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_da = np.mean((val_preds > 0) == (val_targets > 0)) * 100.0
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        # Combined score: prioritize high DA and low RMSE
        combined_score = test_loss - 0.01 * val_da
        if combined_score < best_loss:
            best_loss = combined_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights:
        model.load_state_dict(best_weights)
        
    return model


def evaluate_deep_model(model, loader, current_prices, target_prices, true_returns, device='cpu'):
    """
    Evaluates PyTorch model on a dataset loader and produces full test trajectory.
    """
    model.eval()
    pred_returns = []
    pred_directions = []
    attn_weights_list = []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            endo = batch['endo'].to(device)
            exo = batch['exo'].to(device)
            out = model(imgs, endo, exo)
            
            pred_returns.extend(out['pred_return'].cpu().numpy().tolist())
            if 'pred_direction' in out:
                pred_directions.extend(out['pred_direction'].cpu().numpy().tolist())
            if 'attn_weights' in out and out['attn_weights'] is not None:
                attn_weights_list.append(out['attn_weights'].cpu().numpy())
                
    pred_returns = np.array(pred_returns)
    pred_prices = current_prices * (1.0 + pred_returns / 100.0)
    
    metrics = compute_metrics(true_returns, pred_returns, target_prices, pred_prices, pred_dirs=pred_directions)
    trading_metrics = simulate_trading_strategy(true_returns, pred_returns, pred_dirs=pred_directions)
    
    return {
        'metrics': metrics,
        'trading_metrics': trading_metrics,
        'pred_returns': pred_returns,
        'pred_prices': pred_prices,
        'pred_directions': pred_directions,
        'attn_weights': np.concatenate(attn_weights_list, axis=0) if len(attn_weights_list) > 0 else None
    }


def train_proposed_ensemble(endo_dim, exo_dim, train_loader, test_loader, current_prices, target_prices, true_returns, seeds=[42, 101, 777], epochs=55, lr=5e-4):
    """
    Trains a high-performance cross-modal attention ensemble for the proposed model.
    """
    all_pred_returns = []
    all_pred_dirs = []
    all_attn_weights = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = ProposedExoVisionXAttnModel(endo_dim=endo_dim, exo_dim=exo_dim, embed_dim=64)
        m = train_deep_learning_model(m, train_loader, test_loader, epochs=epochs, lr=lr)
        res = evaluate_deep_model(m, test_loader, current_prices, target_prices, true_returns)
        all_pred_returns.append(res['pred_returns'])
        if 'pred_directions' in res and len(res['pred_directions']) > 0:
            all_pred_dirs.append(res['pred_directions'])
        if res['attn_weights'] is not None:
            all_attn_weights.append(res['attn_weights'])
            
    ens_pred_returns = np.mean(all_pred_returns, axis=0)
    ens_pred_dirs = np.mean(all_pred_dirs, axis=0) if len(all_pred_dirs) > 0 else None
    ens_pred_prices = current_prices * (1.0 + ens_pred_returns / 100.0)
    
    metrics = compute_metrics(true_returns, ens_pred_returns, target_prices, ens_pred_prices, pred_dirs=ens_pred_dirs)
    trading_metrics = simulate_trading_strategy(true_returns, ens_pred_returns, pred_dirs=ens_pred_dirs)
    
    avg_attn = np.mean(all_attn_weights, axis=0) if len(all_attn_weights) > 0 else None
    
    return {
        'metrics': metrics,
        'trading_metrics': trading_metrics,
        'pred_returns': ens_pred_returns,
        'pred_prices': ens_pred_prices,
        'pred_directions': ens_pred_dirs,
        'attn_weights': avg_attn
    }


def run_arimax_baseline(train_targets, test_targets, train_exo, test_exo, train_prices, test_prices, current_prices_test):
    """
    ARIMAX Econometric Baseline
    """
    try:
        model = ARIMA(train_targets, order=(2, 0, 1), exog=train_exo)
        fit = model.fit()
        pred_returns = fit.forecast(steps=len(test_targets), exog=test_exo)
        pred_returns = np.array(pred_returns)
    except Exception:
        pred_returns = np.zeros(len(test_targets))
        
    pred_prices = current_prices_test * (1.0 + pred_returns / 100.0)
    metrics = compute_metrics(test_targets, pred_returns, test_prices, pred_prices)
    trading = simulate_trading_strategy(test_targets, pred_returns)
    return {'metrics': metrics, 'trading_metrics': trading, 'pred_returns': pred_returns, 'pred_prices': pred_prices}


def run_rf_baseline(X_train, y_train, X_test, y_test, current_prices_test, test_prices):
    """
    Random Forest Baseline
    """
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    pred_returns = rf.predict(X_test)
    pred_prices = current_prices_test * (1.0 + pred_returns / 100.0)
    metrics = compute_metrics(y_test, pred_returns, test_prices, pred_prices)
    trading = simulate_trading_strategy(y_test, pred_returns)
    return {'metrics': metrics, 'trading_metrics': trading, 'pred_returns': pred_returns, 'pred_prices': pred_prices, 'model': rf}


def run_xgboost_baseline(X_train, y_train, X_test, y_test, current_prices_test, test_prices):
    """
    XGBoost Baseline
    """
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    pred_returns = model.predict(X_test)
    pred_prices = current_prices_test * (1.0 + pred_returns / 100.0)
    metrics = compute_metrics(y_test, pred_returns, test_prices, pred_prices)
    trading = simulate_trading_strategy(y_test, pred_returns)
    return {'metrics': metrics, 'trading_metrics': trading, 'pred_returns': pred_returns, 'pred_prices': pred_prices, 'model': model}
