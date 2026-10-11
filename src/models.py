import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class CandleVisionEncoder(nn.Module):
    """
    Convolutional Vision Backbone with Residual Squeeze-and-Excitation Blocks
    for extracting intraday candlestick chart patterns, wicks, bodies, and support/resistance geometry.
    """
    def __init__(self, out_dim=64):
        super(CandleVisionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalSelfAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, seq_len, hidden_dim)
        scores = self.attn(x) # (B, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1) # (B, hidden_dim)
        return context, weights


class EndoBiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM with Temporal Self-Attention over endogenous technical series.
    """
    def __init__(self, input_dim=21, hidden_dim=64, out_dim=64):
        super(EndoBiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.attn = TemporalSelfAttention(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        lstm_out, _ = self.lstm(x) # (B, seq_len, hidden_dim)
        context, weights = self.attn(lstm_out)
        return self.proj(context)


class ExoGatedEncoder(nn.Module):
    """
    Gated Variable Selection & Regime Network for Exogenous Macroeconomic & Geopolitical features.
    """
    def __init__(self, input_dim=21, out_dim=64):
        super(ExoGatedEncoder, self).__init__()
        self.glu1 = GatedLinearUnit(input_dim, 64)
        self.glu2 = GatedLinearUnit(64, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (B, input_dim)
        h = self.glu1(x)
        h = self.dropout(h)
        h = self.glu2(h)
        return self.norm(h)


class CrossModalXAttnFusion(nn.Module):
    """
    Novel Cross-Modal Multi-Head Attention Fusion where Exogenous Macro/Geopolitical context
    actively conditions and queries visual candlestick patterns and endogenous time-series dynamics.
    """
    def __init__(self, embed_dim=64, num_heads=4):
        super(CrossModalXAttnFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Attention: Query=Exo, Key=Endo+Vision, Value=Endo+Vision
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.glu = GatedLinearUnit(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-Context Gated Fusion
        self.exo_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Feedforward sublayer
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, endo_vis_embed, exo_embed):
        # endo_vis_embed: (B, 2, embed_dim) - token 0: endo, token 1: vision
        # exo_embed: (B, 1, embed_dim) - query token: exogenous macro state
        
        # Cross-Attention
        attn_out, attn_weights = self.mha(query=exo_embed, key=endo_vis_embed, value=endo_vis_embed)
        # attn_out: (B, 1, embed_dim)
        
        # Dynamic Exogenous Gating
        gate = self.exo_gate(exo_embed)
        gated_attn = self.glu(attn_out) * gate
        
        # Pool endo-vision representation
        pooled_endo_vis = torch.mean(endo_vis_embed, dim=1, keepdim=True) # (B, 1, embed_dim)
        fused = self.norm1(pooled_endo_vis + gated_attn)
        
        # Final non-linear refinement with residual skip
        combined = torch.cat([fused, exo_embed], dim=-1).squeeze(1) # (B, embed_dim * 2)
        out = self.norm3(fused.squeeze(1) + self.ffn(combined))
        
        return out, attn_weights


class ProposedExoVisionXAttnModel(nn.Module):
    """
    Novel Multimodal Cross-Modal Attention Hybrid Deep Learning Architecture:
    ExoVision-XAttn-BiLSTM
    """
    def __init__(self, endo_dim=21, exo_dim=21, embed_dim=64):
        super(ProposedExoVisionXAttnModel, self).__init__()
        self.vision_encoder = CandleVisionEncoder(out_dim=embed_dim)
        self.endo_encoder = EndoBiLSTMEncoder(input_dim=endo_dim, hidden_dim=embed_dim, out_dim=embed_dim)
        self.exo_encoder = ExoGatedEncoder(input_dim=exo_dim, out_dim=embed_dim)
        
        self.fusion = CrossModalXAttnFusion(embed_dim=embed_dim, num_heads=4)
        
        # Residual fusion projection
        self.proj_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Multi-task heads with residual skip
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1) # Next-day return (%)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Up/Down direction probability
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, images, endo_seqs, exo_vecs):
        # 1. Extract visual embedding from candlestick chart
        vis_emb = self.vision_encoder(images) # (B, embed_dim)
        
        # 2. Extract endogenous sequence embedding
        endo_emb = self.endo_encoder(endo_seqs) # (B, embed_dim)
        
        # 3. Extract exogenous macroeconomic context embedding
        exo_emb = self.exo_encoder(exo_vecs) # (B, embed_dim)
        
        # Stack endogenous and vision as key/value tokens
        endo_vis_tokens = torch.stack([endo_emb, vis_emb], dim=1) # (B, 2, embed_dim)
        exo_query = exo_emb.unsqueeze(1) # (B, 1, embed_dim)
        
        # Cross-Modal Fusion
        xattn_emb, attn_weights = self.fusion(endo_vis_tokens, exo_query)
        
        # Combine with multi-scale skip connections
        all_emb = torch.cat([xattn_emb, endo_emb, vis_emb], dim=-1)
        fused_emb = self.proj_fusion(all_emb)
        
        # Multi-task outputs
        pred_return = self.regression_head(fused_emb).squeeze(-1)
        pred_dir = self.classification_head(fused_emb).squeeze(-1)
        
        return {
            'pred_return': pred_return,
            'pred_direction': pred_dir,
            'fused_emb': fused_emb,
            'attn_weights': attn_weights,
            'vis_emb': vis_emb,
            'endo_emb': endo_emb,
            'exo_emb': exo_emb
        }


# =========================================================================
# BASELINE AND ABLATION MODELS FOR RIGOROUS COMPARATIVE BENCHMARKING
# =========================================================================

class PureVisionCNNBaseline(nn.Module):
    """Baseline 1: Vision-Only CNN on Candlestick Images"""
    def __init__(self, embed_dim=64):
        super(PureVisionCNNBaseline, self).__init__()
        self.vision_encoder = CandleVisionEncoder(out_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, images, *args):
        emb = self.vision_encoder(images)
        return {'pred_return': self.fc(emb).squeeze(-1), 'pred_direction': self.cls(emb).squeeze(-1)}


class PureBiLSTMBaseline(nn.Module):
    """Baseline 2: Pure BiLSTM on Endogenous Numerical Series Only"""
    def __init__(self, endo_dim=21, embed_dim=64):
        super(PureBiLSTMBaseline, self).__init__()
        self.endo_encoder = EndoBiLSTMEncoder(input_dim=endo_dim, hidden_dim=embed_dim, out_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, images, endo_seqs, *args):
        emb = self.endo_encoder(endo_seqs)
        return {'pred_return': self.fc(emb).squeeze(-1), 'pred_direction': self.cls(emb).squeeze(-1)}


class VanillaCNNLSTM_NoExo(nn.Module):
    """Baseline 3: Hybrid CNN-BiLSTM without Exogenous Macroeconomic Variables"""
    def __init__(self, endo_dim=21, embed_dim=64):
        super(VanillaCNNLSTM_NoExo, self).__init__()
        self.vision_encoder = CandleVisionEncoder(out_dim=embed_dim)
        self.endo_encoder = EndoBiLSTMEncoder(input_dim=endo_dim, hidden_dim=embed_dim, out_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, images, endo_seqs, *args):
        v = self.vision_encoder(images)
        e = self.endo_encoder(endo_seqs)
        comb = torch.cat([v, e], dim=-1)
        return {'pred_return': self.fc(comb).squeeze(-1), 'pred_direction': self.cls(comb).squeeze(-1)}


class EarlyConcatMultimodal(nn.Module):
    """Baseline 4: Standard Multimodal Linear Concatenation with Exogenous variables"""
    def __init__(self, endo_dim=21, exo_dim=21, embed_dim=64):
        super(EarlyConcatMultimodal, self).__init__()
        self.vision_encoder = CandleVisionEncoder(out_dim=embed_dim)
        self.endo_encoder = EndoBiLSTMEncoder(input_dim=endo_dim, hidden_dim=embed_dim, out_dim=embed_dim)
        self.exo_encoder = nn.Sequential(
            nn.Linear(exo_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, images, endo_seqs, exo_vecs):
        v = self.vision_encoder(images)
        e = self.endo_encoder(endo_seqs)
        x = self.exo_encoder(exo_vecs)
        comb = torch.cat([v, e, x], dim=-1)
        return {'pred_return': self.fc(comb).squeeze(-1), 'pred_direction': self.cls(comb).squeeze(-1)}
