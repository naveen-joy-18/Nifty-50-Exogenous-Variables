import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def generate_architecture_svg(save_path):
    """
    Generates a crisp, comprehensive standalone SVG diagram of the proposed
    ExoVision-XAttn-BiLSTM multimodal architecture.
    """
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800" width="100%" height="100%">
  <defs>
    <linearGradient id="visGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E3A8A" />
      <stop offset="100%" stop-color="#3B82F6" />
    </linearGradient>
    <linearGradient id="endoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#065F46" />
      <stop offset="100%" stop-color="#10B981" />
    </linearGradient>
    <linearGradient id="exoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#9A3412" />
      <stop offset="100%" stop-color="#F97316" />
    </linearGradient>
    <linearGradient id="attnGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4C1D95" />
      <stop offset="100%" stop-color="#8B5CF6" />
    </linearGradient>
    <linearGradient id="outGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A" />
      <stop offset="100%" stop-color="#334155" />
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="4" stdDeviation="4" flood-opacity="0.15" />
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
    </marker>
    <marker id="arrowAttn" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#8B5CF6" />
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="800" fill="#F8FAFC" rx="16" />
  
  <!-- Title Header -->
  <text x="600" y="45" font-size="24" font-weight="bold" fill="#0F172A" text-anchor="middle">
    ExoVision-XAttn-BiLSTM: Novel Multimodal Cross-Attention Hybrid Architecture
  </text>
  <text x="600" y="70" font-size="14" fill="#64748B" text-anchor="middle">
    Integrating Candlestick Visual Patterns, Endogenous Technical Sequences, and Exogenous Macroeconomic Conditioning
  </text>

  <!-- Modality 1: Vision Branch -->
  <g transform="translate(60, 110)" filter="url(#shadow)">
    <rect width="320" height="180" rx="12" fill="white" stroke="#CBD5E1" stroke-width="1.5" />
    <rect width="320" height="35" rx="12" fill="url(#visGrad)" />
    <text x="160" y="23" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Modality 1: Vision Candlestick Encoder</text>
    
    <rect x="20" y="50" width="130" height="50" rx="6" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1" />
    <text x="85" y="72" font-size="11" font-weight="bold" fill="#1E3A8A" text-anchor="middle">Daily Nifty Candlestick</text>
    <text x="85" y="88" font-size="10" fill="#3B82F6" text-anchor="middle">Image I_t (3x128x128)</text>

    <path d="M 155 75 L 175 75" stroke="#475569" stroke-width="2" marker-end="url(#arrow)" />

    <rect x="180" y="50" width="120" height="50" rx="6" fill="#DBEAFE" stroke="#2563EB" stroke-width="1" />
    <text x="240" y="72" font-size="11" font-weight="bold" fill="#1E40AF" text-anchor="middle">Conv2D + SE-Blocks</text>
    <text x="240" y="88" font-size="10" fill="#1D4ED8" text-anchor="middle">AdaptiveAvgPool (128d)</text>

    <rect x="50" y="120" width="220" height="40" rx="6" fill="#1E3A8A" />
    <text x="160" y="145" font-size="12" font-weight="bold" fill="white" text-anchor="middle">Visual Latent Token v_t (64d)</text>
  </g>

  <!-- Modality 2: Endogenous Time-Series Branch -->
  <g transform="translate(60, 310)" filter="url(#shadow)">
    <rect width="320" height="180" rx="12" fill="white" stroke="#CBD5E1" stroke-width="1.5" />
    <rect width="320" height="35" rx="12" fill="url(#endoGrad)" />
    <text x="160" y="23" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Modality 2: Endogenous BiLSTM Branch</text>
    
    <rect x="20" y="50" width="130" height="50" rx="6" fill="#ECFDF5" stroke="#10B981" stroke-width="1" />
    <text x="85" y="70" font-size="11" font-weight="bold" fill="#065F46" text-anchor="middle">Technical Indicators</text>
    <text x="85" y="86" font-size="10" fill="#047857" text-anchor="middle">RSI, MACD, BB, ATR (Lx21)</text>

    <path d="M 155 75 L 175 75" stroke="#475569" stroke-width="2" marker-end="url(#arrow)" />

    <rect x="180" y="50" width="120" height="50" rx="6" fill="#D1FAE5" stroke="#059669" stroke-width="1" />
    <text x="240" y="70" font-size="11" font-weight="bold" fill="#065F46" text-anchor="middle">2-Layer BiLSTM</text>
    <text x="240" y="86" font-size="10" fill="#047857" text-anchor="middle">+ Temporal Attention</text>

    <rect x="50" y="120" width="220" height="40" rx="6" fill="#065F46" />
    <text x="160" y="145" font-size="12" font-weight="bold" fill="white" text-anchor="middle">Sequence Latent Token h_t (64d)</text>
  </g>

  <!-- Modality 3: Exogenous Macroeconomic Branch -->
  <g transform="translate(60, 510)" filter="url(#shadow)">
    <rect width="320" height="230" rx="12" fill="white" stroke="#CBD5E1" stroke-width="1.5" />
    <rect width="320" height="35" rx="12" fill="url(#exoGrad)" />
    <text x="160" y="23" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Modality 3: Exogenous Macro Encoder</text>
    
    <rect x="15" y="45" width="290" height="65" rx="6" fill="#FFF7ED" stroke="#F97316" stroke-width="1" />
    <text x="160" y="63" font-size="11" font-weight="bold" fill="#9A3412" text-anchor="middle">Macroeconomic &amp; Geopolitical Vectors (21d)</text>
    <text x="160" y="80" font-size="10" fill="#C2410C" text-anchor="middle">• Brent Crude Oil • USD/INR FX • US 10Y Yield</text>
    <text x="160" y="97" font-size="10" fill="#C2410C" text-anchor="middle">• Gold Safe-Haven • S&amp;P 500 • India VIX • Stress Index</text>

    <rect x="40" y="120" width="240" height="45" rx="6" fill="#FFEDD5" stroke="#EA580C" stroke-width="1" />
    <text x="160" y="140" font-size="11" font-weight="bold" fill="#9A3412" text-anchor="middle">Gated Variable Selection Network</text>
    <text x="160" y="155" font-size="10" fill="#C2410C" text-anchor="middle">Gated Linear Units (GLU) + LayerNorm</text>

    <rect x="50" y="175" width="220" height="40" rx="6" fill="#9A3412" />
    <text x="160" y="200" font-size="12" font-weight="bold" fill="white" text-anchor="middle">Exogenous Query Token e_t (64d)</text>
  </g>

  <!-- Cross-Modal Attention Fusion Core -->
  <g transform="translate(450, 200)" filter="url(#shadow)">
    <rect width="360" height="400" rx="16" fill="white" stroke="#8B5CF6" stroke-width="2" />
    <rect width="360" height="45" rx="16" fill="url(#attnGrad)" />
    <text x="180" y="28" font-size="15" font-weight="bold" fill="white" text-anchor="middle">Cross-Modal Multi-Head Attention Fusion</text>
    
    <!-- Token Concat -->
    <rect x="30" y="65" width="300" height="50" rx="8" fill="#F5F3FF" stroke="#A78BFA" stroke-width="1" />
    <text x="180" y="87" font-size="12" font-weight="bold" fill="#5B21B6" text-anchor="middle">Endo-Visual Key/Value Tokens</text>
    <text x="180" y="103" font-size="11" fill="#6D28D9" text-anchor="middle">K, V = W_K,V [ v_t ; h_t ]  (2 tokens x 64d)</text>

    <!-- Query -->
    <rect x="30" y="130" width="300" height="50" rx="8" fill="#FFF7ED" stroke="#F97316" stroke-width="1" />
    <text x="180" y="152" font-size="12" font-weight="bold" fill="#9A3412" text-anchor="middle">Exogenous Conditioning Query</text>
    <text x="180" y="168" font-size="11" fill="#C2410C" text-anchor="middle">Q = W_Q [ e_t ]  (Macroeconomic &amp; Geopolitical Regime)</text>

    <!-- Cross Attention Engine -->
    <rect x="30" y="200" width="300" height="75" rx="8" fill="#EDE9FE" stroke="#7C3AED" stroke-width="1.5" />
    <text x="180" y="225" font-size="13" font-weight="bold" fill="#4C1D95" text-anchor="middle">Multi-Head Cross-Attention (h=4)</text>
    <text x="180" y="245" font-size="11" fill="#5B21B6" text-anchor="middle">Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V</text>
    <text x="180" y="263" font-size="10" fill="#6D28D9" text-anchor="middle">Dynamic Contextual Weighting of Candlestick Patterns</text>

    <!-- Residual Gating -->
    <rect x="30" y="295" width="300" height="85" rx="8" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1" />
    <text x="180" y="318" font-size="12" font-weight="bold" fill="#334155" text-anchor="middle">Gated Residual Integration &amp; FFN</text>
    <text x="180" y="338" font-size="10" fill="#475569" text-anchor="middle">Fused = LayerNorm( [v_t; h_t] + GLU(CrossAttn) )</text>
    <text x="180" y="358" font-size="10" fill="#475569" text-anchor="middle">z_t = LayerNorm( Fused + FFN([Fused; e_t]) ) (64d)</text>
  </g>

  <!-- Output Prediction Heads -->
  <g transform="translate(880, 240)" filter="url(#shadow)">
    <rect width="260" height="320" rx="14" fill="white" stroke="#334155" stroke-width="1.5" />
    <rect width="260" height="40" rx="14" fill="url(#outGrad)" />
    <text x="130" y="25" font-size="14" font-weight="bold" fill="white" text-anchor="middle">Multi-Task Prediction Heads</text>
    
    <!-- Regression Head -->
    <rect x="20" y="60" width="220" height="95" rx="8" fill="#F1F5F9" stroke="#64748B" stroke-width="1" />
    <text x="130" y="85" font-size="12" font-weight="bold" fill="#0F172A" text-anchor="middle">Head A: Return &amp; Price Forecasting</text>
    <text x="130" y="105" font-size="11" fill="#334155" text-anchor="middle">Next-Day Return y_hat ( % )</text>
    <text x="130" y="125" font-size="11" font-weight="bold" fill="#2563EB" text-anchor="middle">Next-Day Price P_hat_{t+1}</text>
    <text x="130" y="143" font-size="10" fill="#64748B" text-anchor="middle">Loss: Smooth L1 / Huber</text>

    <!-- Classification Head -->
    <rect x="20" y="180" width="220" height="95" rx="8" fill="#F1F5F9" stroke="#64748B" stroke-width="1" />
    <text x="130" y="205" font-size="12" font-weight="bold" fill="#0F172A" text-anchor="middle">Head B: Direction Probability</text>
    <text x="130" y="225" font-size="11" fill="#334155" text-anchor="middle">P( Direction = Bullish ) in [0, 1]</text>
    <text x="130" y="245" font-size="11" font-weight="bold" fill="#059669" text-anchor="middle">Trade Signal: Long / Short / Flat</text>
    <text x="130" y="263" font-size="10" fill="#64748B" text-anchor="middle">Loss: Binary Cross-Entropy</text>
  </g>

  <!-- Connectors -->
  <!-- Vis -> Fusion -->
  <path d="M 380 250 L 450 285" stroke="#3B82F6" stroke-width="2" marker-end="url(#arrow)" fill="none" />
  <!-- Endo -> Fusion -->
  <path d="M 380 440 L 450 305" stroke="#10B981" stroke-width="2" marker-end="url(#arrow)" fill="none" />
  <!-- Exo -> Fusion -->
  <path d="M 380 640 L 415 640 L 415 350 L 450 350" stroke="#F97316" stroke-width="2" marker-end="url(#arrow)" fill="none" />
  
  <!-- Fusion -> Output -->
  <path d="M 810 400 L 880 400" stroke="#8B5CF6" stroke-width="2.5" marker-end="url(#arrowAttn)" fill="none" />

</svg>
"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(svg_content.strip())
    print(f"[SVG Visualizer] Saved Architecture SVG: {save_path}")


def generate_methodology_flowchart_svg(save_path):
    """
    Generates a methodological pipeline and data flowchart SVG.
    """
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 650" width="100%" height="100%">
  <defs>
    <linearGradient id="stepGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284C7" />
      <stop offset="100%" stop-color="#0EA5E9" />
    </linearGradient>
    <linearGradient id="stepGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0D9488" />
      <stop offset="100%" stop-color="#14B8A6" />
    </linearGradient>
    <linearGradient id="stepGrad3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7C3AED" />
      <stop offset="100%" stop-color="#8B5CF6" />
    </linearGradient>
    <linearGradient id="stepGrad4" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#E11D48" />
      <stop offset="100%" stop-color="#F43F5E" />
    </linearGradient>
    <filter id="fshadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="1" dy="3" stdDeviation="3" flood-opacity="0.12" />
    </filter>
    <marker id="farrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748B" />
    </marker>
  </defs>

  <rect width="1100" height="650" fill="#FFFFFF" rx="16" />
  
  <text x="550" y="45" font-size="22" font-weight="bold" fill="#0F172A" text-anchor="middle">
    End-to-End Research Methodology &amp; Multimodal Pipeline
  </text>
  <text x="550" y="70" font-size="13" fill="#64748B" text-anchor="middle">
    From Heterogeneous Data Ingestion to Multimodal Fusion, Rigorous Evaluation, and Financial Backtesting
  </text>

  <!-- Step 1: Ingestion -->
  <g transform="translate(50, 110)" filter="url(#fshadow)">
    <rect width="210" height="460" rx="10" fill="#F0F9FF" stroke="#BAE6FD" stroke-width="1.5" />
    <rect width="210" height="40" rx="10" fill="url(#stepGrad1)" />
    <text x="105" y="25" font-size="13" font-weight="bold" fill="white" text-anchor="middle">1. Data Ingestion &amp; Sync</text>
    
    <rect x="15" y="55" width="180" height="85" rx="6" fill="white" stroke="#0284C7" stroke-width="1" />
    <text x="105" y="75" font-size="11" font-weight="bold" fill="#0369A1" text-anchor="middle">Candlestick Imagery</text>
    <text x="105" y="93" font-size="10" fill="#0284C7" text-anchor="middle">240 Daily Nifty Charts</text>
    <text x="105" y="110" font-size="10" fill="#0284C7" text-anchor="middle">(2025-2026 Trading Days)</text>
    <text x="105" y="127" font-size="9" fill="#64748B" text-anchor="middle">Spatial Resizing (128x128)</text>

    <rect x="15" y="155" width="180" height="85" rx="6" fill="white" stroke="#0284C7" stroke-width="1" />
    <text x="105" y="175" font-size="11" font-weight="bold" fill="#0369A1" text-anchor="middle">Endogenous OHLCV</text>
    <text x="105" y="193" font-size="10" fill="#0284C7" text-anchor="middle">Daily Open, High, Low,</text>
    <text x="105" y="210" font-size="10" fill="#0284C7" text-anchor="middle">Close, Volume Series</text>
    <text x="105" y="227" font-size="9" fill="#64748B" text-anchor="middle">10-Day Sliding Windows</text>

    <rect x="15" y="255" width="180" height="180" rx="6" fill="white" stroke="#0284C7" stroke-width="1" />
    <text x="105" y="275" font-size="11" font-weight="bold" fill="#0369A1" text-anchor="middle">Exogenous Macro Drivers</text>
    <text x="105" y="295" font-size="10" fill="#0369A1" text-anchor="middle">• Brent Crude Oil (BZ=F)</text>
    <text x="105" y="315" font-size="10" fill="#0369A1" text-anchor="middle">• USD/INR Rate (INR=X)</text>
    <text x="105" y="335" font-size="10" fill="#0369A1" text-anchor="middle">• US 10Y Yield (^TNX)</text>
    <text x="105" y="355" font-size="10" fill="#0369A1" text-anchor="middle">• Gold Spot (GC=F)</text>
    <text x="105" y="375" font-size="10" fill="#0369A1" text-anchor="middle">• S&amp;P 500 Index (^GSPC)</text>
    <text x="105" y="395" font-size="10" fill="#0369A1" text-anchor="middle">• India VIX Volatility</text>
    <text x="105" y="415" font-size="10" fill="#0369A1" text-anchor="middle">• Geopolitical Shock Index</text>
  </g>

  <!-- Step 2: Feature Engineering -->
  <g transform="translate(300, 110)" filter="url(#fshadow)">
    <rect width="210" height="460" rx="10" fill="#F0FDFA" stroke="#99F6E4" stroke-width="1.5" />
    <rect width="210" height="40" rx="10" fill="url(#stepGrad2)" />
    <text x="105" y="25" font-size="13" font-weight="bold" fill="white" text-anchor="middle">2. Feature Engineering</text>
    
    <rect x="15" y="55" width="180" height="120" rx="6" fill="white" stroke="#0D9488" stroke-width="1" />
    <text x="105" y="75" font-size="11" font-weight="bold" fill="#115E59" text-anchor="middle">Technical Indicators</text>
    <text x="105" y="93" font-size="10" fill="#0F766E" text-anchor="middle">• RSI-14 &amp; Stochastic</text>
    <text x="105" y="110" font-size="10" fill="#0F766E" text-anchor="middle">• MACD, Signal &amp; Hist</text>
    <text x="105" y="127" font-size="10" fill="#0F766E" text-anchor="middle">• Bollinger Bands (%B, W)</text>
    <text x="105" y="144" font-size="10" fill="#0F766E" text-anchor="middle">• ATR &amp; Real Body Ratios</text>
    <text x="105" y="161" font-size="10" fill="#0F766E" text-anchor="middle">• Historical Volatility (HV)</text>

    <rect x="15" y="190" width="180" height="130" rx="6" fill="white" stroke="#0D9488" stroke-width="1" />
    <text x="105" y="210" font-size="11" font-weight="bold" fill="#115E59" text-anchor="middle">Macro &amp; Cross-Asset</text>
    <text x="105" y="228" font-size="10" fill="#0F766E" text-anchor="middle">• Oil &amp; FX Momentum</text>
    <text x="105" y="245" font-size="10" fill="#0F766E" text-anchor="middle">• Yield Differential (5d)</text>
    <text x="105" y="262" font-size="10" fill="#0F766E" text-anchor="middle">• Gold-to-Crude Ratio</text>
    <text x="105" y="279" font-size="10" fill="#0F766E" text-anchor="middle">• S&amp;P 500 Overnight Lag</text>
    <text x="105" y="296" font-size="10" fill="#0F766E" text-anchor="middle">• VIX Stress Composite</text>

    <rect x="15" y="335" width="180" height="100" rx="6" fill="white" stroke="#0D9488" stroke-width="1" />
    <text x="105" y="355" font-size="11" font-weight="bold" fill="#115E59" text-anchor="middle">Preprocessing</text>
    <text x="105" y="375" font-size="10" fill="#0F766E" text-anchor="middle">• Robust Z-Score Scaling</text>
    <text x="105" y="393" font-size="10" fill="#0F766E" text-anchor="middle">• Strict Temporal Splitting</text>
    <text x="105" y="411" font-size="10" fill="#0F766E" text-anchor="middle">• Zero Lookahead Bias</text>
  </g>

  <!-- Step 3: Hybrid Modeling -->
  <g transform="translate(550, 110)" filter="url(#fshadow)">
    <rect width="230" height="460" rx="10" fill="#FAF5FF" stroke="#E9D5FF" stroke-width="1.5" />
    <rect width="230" height="40" rx="10" fill="url(#stepGrad3)" />
    <text x="115" y="25" font-size="13" font-weight="bold" fill="white" text-anchor="middle">3. Novel Hybrid Model</text>
    
    <rect x="15" y="55" width="200" height="75" rx="6" fill="white" stroke="#7C3AED" stroke-width="1" />
    <text x="115" y="75" font-size="11" font-weight="bold" fill="#5B21B6" text-anchor="middle">Multi-Branch Encoders</text>
    <text x="115" y="93" font-size="10" fill="#6D28D9" text-anchor="middle">• CNN Candlestick Vision</text>
    <text x="115" y="110" font-size="10" fill="#6D28D9" text-anchor="middle">• BiLSTM Temporal Attention</text>
    <text x="115" y="125" font-size="10" fill="#6D28D9" text-anchor="middle">• Gated Linear Unit (GLU) Exo</text>

    <rect x="15" y="145" width="200" height="110" rx="6" fill="white" stroke="#7C3AED" stroke-width="1" />
    <text x="115" y="165" font-size="11" font-weight="bold" fill="#5B21B6" text-anchor="middle">Cross-Modal Attention</text>
    <text x="115" y="183" font-size="10" fill="#6D28D9" text-anchor="middle">• Exogenous Query (Q_exo)</text>
    <text x="115" y="200" font-size="10" fill="#6D28D9" text-anchor="middle">• Price &amp; Chart Key/Value (K,V)</text>
    <text x="115" y="217" font-size="10" fill="#6D28D9" text-anchor="middle">• Dynamic Regime Weighting</text>
    <text x="115" y="235" font-size="10" fill="#6D28D9" text-anchor="middle">• Residual Skip Connection</text>

    <rect x="15" y="270" width="200" height="85" rx="6" fill="white" stroke="#7C3AED" stroke-width="1" />
    <text x="115" y="290" font-size="11" font-weight="bold" fill="#5B21B6" text-anchor="middle">Multi-Task Optimization</text>
    <text x="115" y="308" font-size="10" fill="#6D28D9" text-anchor="middle">• Smooth L1 (Huber) Regression</text>
    <text x="115" y="325" font-size="10" fill="#6D28D9" text-anchor="middle">• BCE Directional Loss</text>
    <text x="115" y="342" font-size="10" fill="#6D28D9" text-anchor="middle">• AdamW + Cosine Annealing</text>

    <rect x="15" y="370" width="200" height="70" rx="6" fill="white" stroke="#7C3AED" stroke-width="1" />
    <text x="115" y="390" font-size="11" font-weight="bold" fill="#5B21B6" text-anchor="middle">Competitive Baselines</text>
    <text x="115" y="408" font-size="10" fill="#6D28D9" text-anchor="middle">• ARIMAX • RF • XGBoost</text>
    <text x="115" y="425" font-size="10" fill="#6D28D9" text-anchor="middle">• Pure CNN • BiLSTM • CNN-LSTM</text>
  </g>

  <!-- Step 4: Evaluation -->
  <g transform="translate(820, 110)" filter="url(#fshadow)">
    <rect width="230" height="460" rx="10" fill="#FFF1F2" stroke="#FECDD3" stroke-width="1.5" />
    <rect width="230" height="40" rx="10" fill="url(#stepGrad4)" />
    <text x="115" y="25" font-size="13" font-weight="bold" fill="white" text-anchor="middle">4. Evaluation &amp; Backtesting</text>
    
    <rect x="15" y="55" width="200" height="110" rx="6" fill="white" stroke="#E11D48" stroke-width="1" />
    <text x="115" y="75" font-size="11" font-weight="bold" fill="#9F1239" text-anchor="middle">Forecasting Metrics</text>
    <text x="115" y="93" font-size="10" fill="#BE123C" text-anchor="middle">• Root Mean Squared Error (RMSE)</text>
    <text x="115" y="110" font-size="10" fill="#BE123C" text-anchor="middle">• Mean Absolute Error (MAE)</text>
    <text x="115" y="127" font-size="10" fill="#BE123C" text-anchor="middle">• Mean Absolute % Error (MAPE)</text>
    <text x="115" y="144" font-size="10" fill="#BE123C" text-anchor="middle">• Directional Accuracy (Hit Ratio %)</text>
    <text x="115" y="159" font-size="10" fill="#BE123C" text-anchor="middle">• R-Squared (R2 Score)</text>

    <rect x="15" y="180" width="200" height="120" rx="6" fill="white" stroke="#E11D48" stroke-width="1" />
    <text x="115" y="200" font-size="11" font-weight="bold" fill="#9F1239" text-anchor="middle">Trading Simulation</text>
    <text x="115" y="218" font-size="10" fill="#BE123C" text-anchor="middle">• Cumulative Strategy Return (%)</text>
    <text x="115" y="235" font-size="10" fill="#BE123C" text-anchor="middle">• Annualized Sharpe Ratio</text>
    <text x="115" y="252" font-size="10" fill="#BE123C" text-anchor="middle">• Sortino Ratio</text>
    <text x="115" y="269" font-size="10" fill="#BE123C" text-anchor="middle">• Maximum Drawdown (MDD %)</text>
    <text x="115" y="286" font-size="10" fill="#BE123C" text-anchor="middle">• Win Rate &amp; Profit Factor</text>

    <rect x="15" y="315" width="200" height="125" rx="6" fill="white" stroke="#E11D48" stroke-width="1" />
    <text x="115" y="335" font-size="11" font-weight="bold" fill="#9F1239" text-anchor="middle">Ablation &amp; Interpretability</text>
    <text x="115" y="353" font-size="10" fill="#BE123C" text-anchor="middle">• Modality Isolation Tests</text>
    <text x="115" y="370" font-size="10" fill="#BE123C" text-anchor="middle">• Exogenous Factor Importance</text>
    <text x="115" y="387" font-size="10" fill="#BE123C" text-anchor="middle">• Cross-Modal Attention Maps</text>
    <text x="115" y="404" font-size="10" fill="#BE123C" text-anchor="middle">• Residual Diagnostic Analysis</text>
    <text x="115" y="421" font-size="10" fill="#BE123C" text-anchor="middle">• Scalable Vector Graphics (SVG)</text>
  </g>

  <!-- Flow Arrows -->
  <path d="M 260 340 L 300 340" stroke="#64748B" stroke-width="2.5" marker-end="url(#farrow)" />
  <path d="M 510 340 L 550 340" stroke="#64748B" stroke-width="2.5" marker-end="url(#farrow)" />
  <path d="M 780 340 L 820 340" stroke="#64748B" stroke-width="2.5" marker-end="url(#farrow)" />

</svg>
"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(svg_content.strip())
    print(f"[SVG Visualizer] Saved Methodology Flowchart SVG: {save_path}")


def generate_correlation_heatmap_svg(df_full, save_path):
    """
    Generates SVG correlation matrix between Nifty targets and key exogenous indicators.
    """
    selected_cols = [
        'Close', 'ret_1d', 'crude_ret_1d', 'usdinr_ret_1d',
        'us10y_diff_1d', 'gold_ret_1d', 'sp500_ret_1d_lag1',
        'vix_level', 'macro_stress_index', 'geopolitical_risk_proxy'
    ]
    labels = [
        'Nifty Close', 'Nifty Return', 'Brent Crude (ret)', 'USD/INR (ret)',
        'US 10Y Yield (diff)', 'Gold (ret)', 'S&P 500 (lag1)',
        'India VIX', 'Macro Stress', 'Geopolitical Risk'
    ]
    
    sub_df = df_full[selected_cols].dropna()
    corr = sub_df.corr().values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-0.6,
        vmax=0.6,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.0,
        linecolor='white',
        cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8},
        ax=ax
    )
    plt.title('Cross-Asset Exogenous Variables vs. Nifty 50 Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Correlation Heatmap SVG: {save_path}")


def generate_prediction_trajectory_svg(dates, actual_prices, pred_dict, save_path):
    """
    Plots the full temporal trajectory of Actual Nifty Close vs. Proposed Model and Baselines in SVG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    x_indices = np.arange(len(dates))
    
    # Plot top panel: Prices
    ax1.plot(x_indices, actual_prices, label='Actual Nifty 50 Close', color='#0F172A', linewidth=2.5, zorder=5)
    
    colors = {
        'Proposed ExoVision-XAttn': '#2563EB',
        'Vanilla CNN-BiLSTM (No Exo)': '#DC2626',
        'XGBoost (Endo+Exo)': '#D97706',
        'Pure BiLSTM': '#10B981',
        'ARIMAX': '#9333EA'
    }
    
    for name, preds in pred_dict.items():
        c = colors.get(name, '#64748B')
        lw = 2.2 if 'Proposed' in name else 1.4
        ls = '-' if 'Proposed' in name else '--'
        alpha = 1.0 if 'Proposed' in name else 0.75
        ax1.plot(x_indices, preds, label=name, color=c, linewidth=lw, linestyle=ls, alpha=alpha)
        
    ax1.set_title('Nifty 50 Daily Price Trajectory: Actual vs. Multimodal Exogenous Hybrid Forecasting', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('Nifty 50 Index Level', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Bottom panel: Residual Error of Proposed Model vs Vanilla Model
    if 'Proposed ExoVision-XAttn' in pred_dict and 'Vanilla CNN-BiLSTM (No Exo)' in pred_dict:
        res_prop = pred_dict['Proposed ExoVision-XAttn'] - actual_prices
        res_vanilla = pred_dict['Vanilla CNN-BiLSTM (No Exo)'] - actual_prices
        ax2.plot(x_indices, res_vanilla, label='Vanilla CNN-LSTM Error (No Exo)', color='#EF4444', linewidth=1.2, alpha=0.7)
        ax2.plot(x_indices, res_prop, label='Proposed ExoVision-XAttn Error', color='#2563EB', linewidth=1.8)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Forecast Error (Pts)', fontsize=11, fontweight='bold')
        ax2.legend(loc='lower left', frameon=True, fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
    # Format x-ticks
    step = max(1, len(dates) // 8)
    ax2.set_xticks(x_indices[::step])
    ax2.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha='right', fontsize=9)
    ax2.set_xlabel('Test Trading Dates (2026)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Trajectory SVG: {save_path}")


def generate_benchmark_metrics_svg(results_df, save_path):
    """
    Generates multi-panel comparison bar charts for all baseline models across RMSE, MAE, MAPE, R2, Directional Acc.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = results_df['Model'].values
    palette = ['#2563EB' if 'Proposed' in m else '#94A3B8' for m in models]
    
    # 1. RMSE Price (Lower is better)
    sns.barplot(x='RMSE_Price', y='Model', hue='Model', data=results_df, ax=axes[0, 0], palette=palette, legend=False)
    axes[0, 0].set_title('Root Mean Squared Error - RMSE (Lower is Better)', fontweight='bold', fontsize=11)
    axes[0, 0].set_xlabel('RMSE (Index Points)')
    axes[0, 0].set_ylabel('')
    for i, v in enumerate(results_df['RMSE_Price']):
        axes[0, 0].text(v + 2, i, f"{v:.1f}", va='center', fontsize=9, fontweight='bold')
        
    # 2. MAPE (%) (Lower is better)
    sns.barplot(x='MAPE_Price', y='Model', hue='Model', data=results_df, ax=axes[0, 1], palette=palette, legend=False)
    axes[0, 1].set_title('Mean Absolute Percentage Error - MAPE % (Lower is Better)', fontweight='bold', fontsize=11)
    axes[0, 1].set_xlabel('MAPE (%)')
    axes[0, 1].set_ylabel('')
    for i, v in enumerate(results_df['MAPE_Price']):
        axes[0, 1].text(v + 0.02, i, f"{v:.2f}%", va='center', fontsize=9, fontweight='bold')

    # 3. Directional Accuracy (Higher is better)
    sns.barplot(x='Directional_Accuracy', y='Model', hue='Model', data=results_df, ax=axes[1, 0], palette=palette, legend=False)
    axes[1, 0].set_title('Directional Hit Ratio % (Higher is Better)', fontweight='bold', fontsize=11)
    axes[1, 0].set_xlabel('Directional Accuracy (%)')
    axes[1, 0].set_ylabel('')
    axes[1, 0].set_xlim(35, 95)
    for i, v in enumerate(results_df['Directional_Accuracy']):
        axes[1, 0].text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')

    # 4. R2 Score (Higher is better)
    sns.barplot(x='R2_Score', y='Model', hue='Model', data=results_df, ax=axes[1, 1], palette=palette, legend=False)
    axes[1, 1].set_title('Coefficient of Determination - R² Score (Higher is Better)', fontweight='bold', fontsize=11)
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_ylabel('')
    for i, v in enumerate(results_df['R2_Score']):
        axes[1, 1].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Comprehensive Model Benchmarking Comparison: Statistical, ML & Deep Learning', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Benchmark Metrics SVG: {save_path}")


def generate_ablation_svg(ablation_df, save_path):
    """
    Generates ablation study SVG breakdown showing the impact of removing exogenous factors and architectural modules.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    variants = ablation_df['Variant'].values
    palette = ['#2563EB' if 'Proposed Full' in v else '#F87171' if 'No Exo' in v else '#F59E0B' for v in variants]
    
    # Left: Directional Accuracy
    sns.barplot(x='Directional_Accuracy', y='Variant', hue='Variant', data=ablation_df, ax=ax1, palette=palette, legend=False)
    ax1.set_title('Ablation Impact on Directional Accuracy (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Directional Accuracy (%)')
    ax1.set_ylabel('')
    ax1.set_xlim(40, 95)
    for i, v in enumerate(ablation_df['Directional_Accuracy']):
        ax1.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
        
    # Right: RMSE Price
    sns.barplot(x='RMSE_Price', y='Variant', hue='Variant', data=ablation_df, ax=ax2, palette=palette, legend=False)
    ax2.set_title('Ablation Impact on Prediction Error (RMSE)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('RMSE (Index Points)')
    ax2.set_ylabel('')
    for i, v in enumerate(ablation_df['RMSE_Price']):
        ax2.text(v + 2, i, f"{v:.1f}", va='center', fontsize=9, fontweight='bold')
        
    plt.suptitle('Ablation Analysis: Contribution of Exogenous Macro Drivers & Cross-Attention Fusion', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Ablation SVG: {save_path}")


def generate_trading_equity_svg(dates, equity_dict, save_path):
    """
    Generates backtesting cumulative equity growth curves and drawdown profiles in SVG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    x_indices = np.arange(len(dates))
    
    colors = {
        'Proposed ExoVision-XAttn': '#2563EB',
        'Vanilla CNN-BiLSTM (No Exo)': '#DC2626',
        'XGBoost (Endo+Exo)': '#D97706',
        'Buy & Hold Nifty': '#0F172A'
    }
    
    for name, equity in equity_dict.items():
        c = colors.get(name, '#64748B')
        lw = 2.5 if 'Proposed' in name else 1.6
        ax1.plot(x_indices, equity, label=name, color=c, linewidth=lw)
        
        # Calculate drawdown
        peaks = np.maximum.accumulate(equity)
        dd = (equity - peaks) / peaks * 100.0
        if 'Proposed' in name or 'Buy & Hold' in name:
            ax2.plot(x_indices, dd, label=f"{name} DD%", color=c, linewidth=1.4)
            
    ax1.set_title('Quantitative Backtest Performance: Cumulative Portfolio Equity (Base = ₹100,000)', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('Portfolio Value (INR ₹)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.set_title('Drawdown Profile (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.legend(loc='lower left', frameon=True, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    step = max(1, len(dates) // 8)
    ax2.set_xticks(x_indices[::step])
    ax2.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha='right', fontsize=9)
    ax2.set_xlabel('Test Trading Dates (2026)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Trading Equity SVG: {save_path}")


def generate_feature_importance_svg(feature_names, importances, save_path):
    """
    Plots exogenous feature importance rankings in SVG.
    """
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = [importances[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("mako", len(sorted_names))
    ax.barh(sorted_names, sorted_vals, color=palette)
    ax.set_title('Exogenous & Macroeconomic Feature Importance Ranking (Gated Weights)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Importance Score (Normalized)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    for i, v in enumerate(sorted_vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Feature Importance SVG: {save_path}")


def generate_cross_attention_heatmap_svg(dates, attn_matrix, save_path):
    """
    Plots cross-modal attention weight distribution between Exogenous Query and Endo/Vision Key tokens over time in SVG.
    """
    attn_arr = np.array(attn_matrix)
    # Ensure 2D shape: (N_samples, 2)
    if attn_arr.ndim == 4:
        attn_arr = attn_arr.mean(axis=1) # average over heads if 4D
    if attn_arr.ndim == 3:
        attn_arr = np.squeeze(attn_arr, axis=1)
    if attn_arr.ndim != 2 or attn_arr.shape[1] != 2:
        attn_arr = attn_arr.reshape(-1, 2)
        
    attn_2d = attn_arr.T # (2, N_samples)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        attn_2d,
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Cross-Attention Weight'},
        yticklabels=['Endogenous BiLSTM Token', 'Candlestick Vision Token'],
        ax=ax
    )
    ax.set_title('Dynamic Cross-Modal Attention Weights Conditioned on Exogenous Macro Regimes', fontsize=13, fontweight='bold', pad=12)
    
    step = max(1, len(dates) // 8)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Test Trading Dates (2026)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Cross-Attention SVG: {save_path}")


def generate_residual_diagnostics_svg(residuals, save_path):
    """
    Generates residual diagnostics (histogram, Q-Q, autocorrelation) in SVG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Residual Histogram + KDE
    sns.histplot(residuals, kde=True, color='#2563EB', ax=ax1, bins=20)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Residual Error Distribution (Gaussianity Check)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Forecast Residual (Actual - Predicted Return %)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Right: Residuals vs Index (Homoscedasticity)
    ax2.scatter(np.arange(len(residuals)), residuals, color='#4F46E5', alpha=0.75, edgecolors='none', s=35)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)
    ax2.set_title('Residual Error Sequence (No Autocorrelation / Drift)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Test Trading Day Index')
    ax2.set_ylabel('Forecast Residual (%)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Residual Diagnostics of Proposed ExoVision-XAttn-BiLSTM Model', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='svg')
    plt.close(fig)
    print(f"[SVG Visualizer] Saved Residual Diagnostics SVG: {save_path}")
