import torch
import torch.nn as nn


def build_encoder(input_channels, encoder_channels, kernel_sizes, strides):
    """인코더 네트워크 생성 함수"""
    encoder_layers = []
    in_channels = input_channels
    for out_ch, k, s in zip(encoder_channels, kernel_sizes, strides):
        encoder_layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_ch,
            kernel_size=k,
            stride=s
        ))
        encoder_layers.append(nn.ReLU())
        in_channels = out_ch  # match: prev-out -> next-in
    return nn.Sequential(*encoder_layers)


def build_embedding(input_size, embed_dim, use_layer_norm=True):
    """임베딩 네트워크 생성 함수"""
    norm_layer = nn.LayerNorm(embed_dim) if use_layer_norm else nn.BatchNorm1d(embed_dim)
    return nn.Sequential(
        nn.Linear(input_size, embed_dim),
        norm_layer,
        nn.Tanh()
    )


def build_mlp(input_dim, hidden_sizes, output_dim):
    """MLP 네트워크 생성 함수"""
    layers = []
    in_size = input_dim
    for out_size in hidden_sizes:
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
        in_size = out_size
    layers.append(nn.Linear(in_size, output_dim))
    return nn.Sequential(*layers)
