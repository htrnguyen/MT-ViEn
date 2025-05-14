#!/usr/bin/env python
# coding: utf-8

"""
models.py: Định nghĩa 4 mô hình dịch máy: MarianMT, TransformerScratch, GPTScratch, GPT2FineTuned.
Tích hợp layer normalization, residual connections, scaled dot-product attention.
Tối ưu cho GPU P100, hỗ trợ kiểm tra cấu trúc mô hình.
"""

import torch
import torch.nn as nn
import math
from transformers import MarianMTModel, GPT2LMHeadModel
from .utils import print_device_info

# Thiết lập thiết bị (mặc định GPU P100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    Lớp mã hóa vị trí cho các mô hình Transformer/GPT, thêm thông tin thứ tự token.
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MarianMTWrapper(nn.Module):
    """
    Wrapper cho mô hình MarianMT từ Hugging Face, hỗ trợ tích hợp với pipeline.
    """
    def __init__(self, model_name):
        super().__init__()
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        self.config = self.model.config
        
        print(f"Khởi tạo MarianMTWrapper:")
        print(f"- Model: {model_name}")
        print(f"- Vocab size: {self.config.vocab_size}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **generation_params):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_params
        )

class TransformerScratchModel(nn.Module):
    """
    Mô hình Transformer từ đầu với layer normalization, residual connections,
    và scaled dot-product attention.
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.2,
        max_seq_len=50
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # Layer normalization
        self.src_norm = nn.LayerNorm(d_model)
        self.tgt_norm = nn.LayerNorm(d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Khởi tạo trọng số
        self._init_weights()
        
        print(f"Khởi tạo TransformerScratchModel:")
        print(f"- Kích thước từ vựng: src={src_vocab_size}, tgt={tgt_vocab_size}")
        print(f"- d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}/{num_decoder_layers}")

    def _init_weights(self):
        """Khởi tạo trọng số cho embedding và linear layers."""
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Tạo mask nhân quả cho decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, src, tgt_decoder_input, src_key_padding_mask, tgt_key_padding_mask):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.src_norm(src_emb)
        
        tgt_emb = self.tgt_embedding(tgt_decoder_input) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.tgt_norm(tgt_emb)
        
        tgt_mask = self._generate_square_subsequent_mask(tgt_decoder_input.size(1))
        
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc_out(output)

    def generate(self, src, src_key_padding_mask, max_len, start_token_id, end_token_id, pad_token_id):
        self.eval()
        batch_size = src.size(0)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.src_norm(src_emb)
        
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        tgt_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(start_token_id)
        
        for _ in range(max_len - 1):
            tgt_emb = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
            tgt_emb = self.positional_encoding(tgt_emb)
            tgt_emb = self.tgt_norm(tgt_emb)
            
            tgt_mask = self._generate_square_subsequent_mask(tgt_tokens.size(1))
            
            output = self.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            pred_logits = self.fc_out(output[:, -1, :])
            next_token = pred_logits.argmax(1).unsqueeze(1)
            tgt_tokens = torch.cat((tgt_tokens, next_token), dim=1)
            
            if (next_token == end_token_id).all():
                break
        
        return tgt_tokens

class ScratchGPTModel(nn.Module):
    """
    Mô hình GPT từ đầu với layer normalization, residual connections,
    và causal self-attention.
    """
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.2,
        max_seq_len=50
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # Layer normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Khởi tạo trọng số
        self._init_weights()
        
        print(f"Khởi tạo ScratchGPTModel:")
        print(f"- Kích thước từ vựng: {vocab_size}")
        print(f"- d_model={d_model}, nhead={nhead}, layers={num_decoder_layers}")

    def _init_weights(self):
        """Khởi tạo trọng số cho embedding và linear layers."""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Tạo mask nhân quả cho self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, tgt_input, tgt_key_padding_mask=None):
        tgt_emb = self.token_embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.input_norm(tgt_emb)
        
        causal_mask = self._generate_square_subsequent_mask(tgt_input.size(1))
        
        output = self.transformer_decoder(
            tgt_emb,
            mask=causal_mask,
            src_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)

    def generate(self, prompt_ids, max_len, end_token_id, pad_token_id):
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_len - prompt_ids.size(1)):
            padding_mask = (generated == pad_token_id)
            
            tgt_emb = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.positional_encoding(tgt_emb)
            tgt_emb = self.input_norm(tgt_emb)
            
            causal_mask = self._generate_square_subsequent_mask(generated.size(1))
            
            output = self.transformer_decoder(
                tgt_emb,
                mask=causal_mask,
                src_key_padding_mask=padding_mask
            )
            
            next_logits = self.fc_out(output[:, -1, :])
            next_token = next_logits.argmax(1).unsqueeze(1)
            generated = torch.cat((generated, next_token), dim=1)
            
            if (next_token == end_token_id).all():
                break
        
        return generated

class GPT2FineTunedWrapper(nn.Module):
    """
    Wrapper cho mô hình GPT-2 từ Hugging Face, hỗ trợ fine-tuning cho dịch máy.
    """
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.config = self.model.config
        
        print(f"Khởi tạo GPT2FineTunedWrapper:")
        print(f"- Model: {model_name}")
        print(f"- Vocab size: {self.config.vocab_size}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **generation_params):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_params
        )

if __name__ == "__main__":
    # Kiểm tra thiết bị
    print_device_info()
    
    # Kiểm tra khởi tạo mô hình
    vocab_size = 30000
    
    # MarianMT
    marian = MarianMTWrapper("Helsinki-NLP/opus-mt-en-vi").to(device)
    
    # TransformerScratch
    transformer = TransformerScratchModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size
    ).to(device)
    
    # GPTScratch
    gpt_scratch = ScratchGPTModel(vocab_size=vocab_size).to(device)
    
    # GPT2FineTuned
    gpt2 = GPT2FineTunedWrapper("gpt2").to(device)
    
    # Kiểm tra forward pass với dữ liệu giả
    batch_size, seq_len = 2, 10
    src = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    print("\nKiểm tra forward pass MarianMTWrapper:")
    marian_out = marian(src, attention_mask=attention_mask)
    print(f"- Kích thước output logits: {marian_out.logits.shape}")
    
    print("\nKiểm tra forward pass TransformerScratchModel:")
    transformer_out = transformer(src, tgt, padding_mask, padding_mask)
    print(f"- Kích thước output: {transformer_out.shape}")
    
    print("\nKiểm tra forward pass ScratchGPTModel:")
    gpt_scratch_out = gpt_scratch(tgt, padding_mask)
    print(f"- Kích thước output: {gpt_scratch_out.shape}")
    
    print("\nKiểm tra forward pass GPT2FineTunedWrapper:")
    gpt2_out = gpt2(src, attention_mask=attention_mask)
    print(f"- Kích thước output logits: {gpt2_out.logits.shape}")