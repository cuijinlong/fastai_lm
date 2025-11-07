import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import math
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ImageTransformerClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, d_model=128, nhead=8, num_layers=3, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 图像到序列的投影 (28x28 = 784)
        self.patch_embed = nn.Linear(784, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 展平图像 [batch, 784]
        x = x.view(batch_size, -1)
        
        # 添加序列维度并投影 [batch, 1, d_model]
        x = x.unsqueeze(1)
        x = self.patch_embed(x)
        
        # 位置编码 [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        
        # Transformer [seq_len, batch, d_model]
        x = self.transformer(x)
        
        # 全局平均池化并分类 [batch, num_classes]
        x = x.mean(dim=0)
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer