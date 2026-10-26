# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:40:00 2026

@author: umroot
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


    
    
    
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    
    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
        

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding)) #normally it means adding no zeros on left and padding zeros on right

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1] #transpose was done before calling patchembedding, so it's fine x:bs,F,T
        x = self.padding_patch_layer(x) #bs,F,T+padding
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) #bs,F,num_patches,patch_len
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #bs*F,num_patches,patch_len
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) #bs*F,num_patches,d_model
        return self.dropout(x), n_vars


    
class PatchTSTEncoder(nn.Module):

    def __init__(self, seq_len, d_model, d_ff, n_heads, e_layers, dropout=0.1, factor=5, activation="relu", patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )
        
        # Prediction Head
        self.num_patches = int((seq_len - patch_len) / stride + 2)
        self.fc = nn.Linear(
            self.num_patches * d_model,
            seq_len
        )
        self.flatten = nn.Flatten(start_dim=2)
        # self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
        #                         head_dropout=configs.dropout)
        # self.fc = nn.Linear(
        #     seq_len * d_model,
        #     seq_len * n_features
        # )

    def forward(self, x_enc):
         # do patching and embedding
        #x_enc: bs,T,F
        x_enc = x_enc.permute(0, 2, 1) #bs,F,T
        
        enc_out, n_vars = self.patch_embedding(x_enc)  #[bs * nvars x patch_num x d_model]

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        #for fair comparison with baselines, after encoding we apply directly an fc layer to transform into shape (bs,T*F)
        enc_out = self.flatten(enc_out) #bs x nvars x d_model * patch_num
        enc_out = self.fc(enc_out) #bs x nvars x seq_len
        #to uncomment this line
        enc_out = enc_out.flatten(1)

        return enc_out
    
    
class ResBlock(nn.Module):
    def __init__(self, seq_len, d_model, dropout, n_feat):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(n_feat, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_feat),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x
    
class TSMixerEncoder(nn.Module):
    def __init__(self, layer, seq_len, d_model, dropout, n_feat):
        super(TSMixerEncoder, self).__init__()
        self.layer = layer
        self.model = nn.ModuleList([ResBlock(seq_len, d_model, dropout, n_feat)
                                    for _ in range(layer)])


    def forward(self, x):

        # x: [B, T, F]
        for i in range(self.layer):
            x_enc = self.model[i](x)

        x_enc = x_enc.flatten(1)  # (B,T*F)


        return x_enc


            
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        #x:bs,T,F
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x) #bs,T,d_model
    
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        #self.seq_len = configs.seq_len
        self.pred_len = 0 #because timesblock is used as an encoder and to ensure fair comparison with baselines
        #self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels))
        self.k=k
        self.seq_len=seq_len

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
    
class TimesNetEncoder(nn.Module):
    def __init__(self, seq_len, k, d_model, d_ff, num_kernels, n_features, layer):
        super(TimesNetEncoder, self).__init__()
        #self.configs = configs
        #self.task_name = configs.task_name
        #self.seq_len = configs.seq_len
        #self.label_len = configs.label_len
        self.pred_len = 0
        self.model = nn.ModuleList([TimesBlock(seq_len, k, d_model, d_ff, num_kernels)
                                    for _ in range(layer)])
        self.enc_embedding = DataEmbedding(n_features, d_model)
        #self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer=layer
        # Convert encoder output back to T*F
        self.fc = nn.Linear(
            seq_len * d_model,
            seq_len * n_features
        )
    def forward(self, x_enc, x_mark_enc=None):
        #x_enc:B,T,F
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,d_model]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out)) 
        #enc_out:B,T,d_model
        x = enc_out.flatten(1)  # (B,T*d_model)
        x = self.fc(x)                         # (B,T*F)
        return x
        #example of instanciation with 1 single timeblock: tne=TimesNetEncoder(168, 2, 128, 128, 6, 8, 1)


def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2) *
        (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)   # (1,T,d_model)


class TransformerEncoder(nn.Module): #vanilla tf encoder
    def __init__(
        self,
        feature_dim,
        d_model,
        seq_len,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.register_buffer(
                                "pos_embedding",
                                positional_encoding(seq_len, d_model)
                            )
        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(seq_len * d_model, seq_len * feature_dim)

    def forward(self, x): #, pos_encoding
        # x: (B,T,F)
        # pos_encoding: (1,T,d_model)

        x = self.input_proj(x)      # (B,T,d)
        x = x + self.pos_embedding        # (B,T,d) #+pos_encoding
        x = self.encoder(x)         # (B,T,d)
        x = x.flatten(1)            # (B,T*d)
        x = self.fc(x)              # (B,T*F)

        return x

class TCNEncoder(nn.Module):
    def __init__(self, feature_dim, channels, seq_len,
                 kernel_size=3, num_layers=3):
        super().__init__()

        layers = []
        in_channels = feature_dim

        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            layers.extend([
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation
                ),
                nn.ReLU(),
            ])

            in_channels = channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels * seq_len, seq_len * feature_dim)

        self.seq_len = seq_len

    def forward(self, x):
        # x: (B, T, F)

        x = x.transpose(1, 2)          # (B, F, T)
        x = self.tcn(x) #B, C, T1, T1>T because of causal padding

        # Remove extra timesteps introduced by causal padding
        x = x[..., :self.seq_len]      # (B, C, T)

        x = x.flatten(1)               # (B, C*T)
        x = self.fc(x)                 # (B, T*F)

        return x
    


class LSTMEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, seq_len, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(#GRU(#
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, seq_len * feature_dim)
        

    def forward(self, x):
        # x: (B, T, F)

        _, (h_n, _) = self.lstm(x)  #(n_layers, B, H) lstm
        #_, h_n = self.lstm(x) #gru
        # Last layer hidden state
        h = h_n[-1]     # #hidden state of the last layer:  (B, H)

        out = self.fc(h)     # (B, T*F)

        return out

