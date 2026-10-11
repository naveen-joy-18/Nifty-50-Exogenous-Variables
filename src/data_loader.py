import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import yfinance as yf

class NiftyMultimodalDataset(Dataset):
    """
    Multimodal Dataset combining:
    1. Daily Candlestick Chart Images (Preloaded in Memory for Ultra-Fast Training)
    2. Endogenous Technical Indicators & OHLCV Time-Series (Endogenous Modality)
    3. Macroeconomic & Geopolitical Exogenous Variables (Exogenous Modality)
    """
    def __init__(self, image_paths, endo_features, exo_features, targets, directions, transform=None):
        self.image_paths = image_paths
        self.endo_features = torch.tensor(endo_features, dtype=torch.float32)
        self.exo_features = torch.tensor(exo_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.directions = torch.tensor(directions, dtype=torch.float32)
        
        # Pre-load and transform all images into RAM tensor cache
        cached_imgs = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            if transform:
                img_t = transform(img)
            else:
                img_t = transforms.ToTensor()(img)
            cached_imgs.append(img_t)
        self.images_tensor = torch.stack(cached_imgs, dim=0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            'image': self.images_tensor[idx],
            'endo': self.endo_features[idx],
            'exo': self.exo_features[idx],
            'target': self.targets[idx],
            'direction': self.directions[idx],
            'path': self.image_paths[idx]
        }


def parse_filename_date(filename):
    """
    Parses dates in formats like '1.8.25.png', '09.10.25.jpeg', '01.07.26.png'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('.')
    if len(parts) == 3:
        try:
            d = int(parts[0])
            m = int(parts[1])
            y = int(parts[2])
            if y < 100:
                y += 2000
            return f"{y:04d}-{m:02d}-{d:02d}"
        except Exception:
            return None
    return None


def fetch_market_and_exogenous_data(start_date="2025-05-01", end_date="2026-08-20"):
    """
    Fetches Nifty 50 and all relevant exogenous macroeconomic/geopolitical drivers:
    - Nifty 50 (^NSEI): Endogenous target & OHLCV
    - Brent Crude Oil (BZ=F): Key macroeconomic energy import driver for India
    - USD/INR (INR=X): Foreign exchange rate & currency risk
    - US 10-Year Treasury Yield (^TNX): Global sovereign discount rate & FII liquidity
    - Gold (GC=F): Geopolitical safe haven & inflation hedge
    - S&P 500 (^GSPC): Global equity sentiment & overnight contagion
    - India VIX (^INDIAVIX): Domestic market fear & volatility gauge
    """
    tickers = {
        'NIFTY': '^NSEI',
        'CRUDE_OIL': 'BZ=F',
        'USD_INR': 'INR=X',
        'US_10Y': '^TNX',
        'GOLD': 'GC=F',
        'SP500': '^GSPC',
        'INDIA_VIX': '^INDIAVIX'
    }
    
    data_dict = {}
    print("[Data Fetcher] Downloading financial and macroeconomic series from Yahoo Finance...")
    for key, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            # Forward fill and backward fill missing days (e.g. cross-market holiday differences)
            df = df.reindex(pd.date_range(start_date, end_date, freq='D')).ffill().bfill()
            data_dict[key] = df
            print(f"  -> Fetched {key} ({ticker}): {len(df)} calendar days.")
        except Exception as e:
            print(f"  -> Error fetching {key}: {e}")
            
    return data_dict


def scan_candlestick_images(workspace_dir):
    """
    Scans the directory tree for all candlestick chart images and maps them by date string.
    """
    image_dict = {}
    for root, dirs, files in os.walk(workspace_dir):
        for f in sorted(files):
            if f.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                dt_str = parse_filename_date(f)
                if dt_str:
                    image_dict[dt_str] = os.path.abspath(os.path.join(root, f))
    print(f"[Image Scanner] Found {len(image_dict)} candlestick chart images mapped to valid dates.")
    return image_dict


def get_image_transforms(img_size=(128, 128)):
    """
    Returns image transformation pipelines for training and testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.90, 0.90, 0.90], std=[0.20, 0.20, 0.20])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.90, 0.90, 0.90], std=[0.20, 0.20, 0.20])
    ])
    
    return train_transform, test_transform
