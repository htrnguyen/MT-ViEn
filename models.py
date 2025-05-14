#!/usr/bin/env python
# coding: utf-8

"""
models.py: Định nghĩa 4 mô hình dịch máy: MarianMT, TransformerScratch, GPTScratch, GPT2FineTuned.
Tích hợp layer normalization, residual connections, scaled dot-product attention.
Tối ưu cho GPU P100, hỗ trợ kiểm tra cấu trúc mô hình.
"""

import math

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, MarianMTModel

from utils import print_device_info

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
        x = x + self.pe[:, : x.size(1)]
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
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask=None, **generation_params):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generation_params
        )


class TransformerScratchModel(nn.Module):
    """
    Mô hình Transformer từ đầu, tái hiện từ mt-vien_v1.py, với beam search và sửa lỗi batch processing.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=8,
        num_decoder_layers=8,
        dim_feedforward=512,
        dropout=0.2,
        max_seq_len=50,
    ):
        super().__init__()
        self.d_model = d_model

        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, max_len=max_seq_len + 10
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Khởi tạo trọng số
        self._init_weights()

        print(f"Khởi tạo TransformerScratchModel:")
        print(f"- Kích thước từ vựng: src={src_vocab_size}, tgt={tgt_vocab_size}")
        print(
            f"- d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}/{num_decoder_layers}"
        )

    def _init_weights(self):
        """Khởi tạo trọng số với Xavier uniform và gain điều chỉnh."""
        nn.init.xavier_uniform_(
            self.src_embedding.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_uniform_(
            self.tgt_embedding.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_uniform_(
            self.fc_out.weight, gain=nn.init.calculate_gain("linear")
        )
        nn.init.zeros_(self.fc_out.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Tạo mask nhân quả cho decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self, src, tgt_decoder_input, src_key_padding_mask, tgt_key_padding_mask
    ):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)

        tgt_emb = self.tgt_embedding(tgt_decoder_input) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)

        tgt_mask = self._generate_square_subsequent_mask(tgt_decoder_input.size(1))

        memory = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(output)

    def generate(
        self,
        src,
        src_key_padding_mask,
        max_len,
        start_token_id,
        end_token_id,
        pad_token_id,
        num_beams=5,
    ):
        self.eval()
        batch_size = src.size(0)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        memory = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # Beam search cho từng mẫu trong batch
        final_beams = []
        final_preds = []

        for b in range(batch_size):
            beams = [
                (torch.tensor([[start_token_id]], dtype=torch.long, device=device), 0.0)
            ]
            completed = []

            for _ in range(max_len - 1):
                new_beams = []
                for beam, score in beams:
                    tgt_emb = self.tgt_embedding(beam) * math.sqrt(self.d_model)
                    tgt_emb = self.positional_encoding(tgt_emb)
                    tgt_mask = self._generate_square_subsequent_mask(beam.size(1))

                    output = self.transformer_decoder(
                        tgt_emb,
                        memory[b : b + 1],
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=src_key_padding_mask[b : b + 1],
                    )

                    pred_logits = self.fc_out(output[:, -1, :])
                    probs = torch.softmax(pred_logits, dim=-1)
                    top_probs, top_idx = probs.topk(num_beams, dim=-1)

                    for i in range(num_beams):
                        next_token = top_idx[0, i].unsqueeze(0).unsqueeze(0)
                        next_score = score - torch.log(top_probs[0, i]).item()
                        new_beam = torch.cat((beam, next_token), dim=1)
                        new_beams.append((new_beam, next_score))

                new_beams = sorted(new_beams, key=lambda x: x[1])[:num_beams]
                beams = []
                for beam, score in new_beams:
                    if beam[0, -1].item() == end_token_id:
                        completed.append((beam, score))
                    else:
                        beams.append((beam, score))

                if len(completed) >= num_beams or not beams:
                    break

            if not completed:
                completed = beams

            best_beam, _ = min(completed, key=lambda x: x[1])
            ids_list = best_beam[0].cpu().tolist()
            end_idx = (
                ids_list.index(end_token_id)
                if end_token_id in ids_list
                else len(ids_list)
            )
            final_beams.append(best_beam)
            final_preds.append(
                tokenizer.decode(ids_list[1:end_idx], skip_special_tokens=True)
            )

        best_beams = torch.cat(final_beams, dim=0).view(batch_size, -1)
        return best_beams, final_preds


class ScratchGPTModel(nn.Module):
    """
    Mô hình GPT từ đầu với beam search và scheduled sampling, sửa lỗi mask shape.
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_decoder_layers=8,
        dim_feedforward=512,
        dropout=0.2,
        max_seq_len=50,
    ):
        super().__init__()
        self.d_model = d_model

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, max_len=max_seq_len + 10
        )

        # Layer normalization
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_decoder_layers
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Khởi tạo trọng số
        self._init_weights()

        print(f"Khởi tạo ScratchGPTModel:")
        print(f"- Kích thước từ vựng: {vocab_size}")
        print(f"- d_model={d_model}, nhead={nhead}, layers={num_decoder_layers}")

    def _init_weights(self):
        """Khởi tạo trọng số với Xavier uniform và gain điều chỉnh."""
        nn.init.xavier_uniform_(
            self.token_embedding.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_uniform_(
            self.fc_out.weight, gain=nn.init.calculate_gain("linear")
        )
        nn.init.zeros_(self.fc_out.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Tạo mask nhân quả cho self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, tgt_input, tgt_key_padding_mask=None, teacher_forcing_ratio=0.5):
        # Đồng bộ mask với input khi teacher forcing
        if torch.rand(1).item() < teacher_forcing_ratio and self.training:
            tgt_emb = self.token_embedding(tgt_input) * math.sqrt(self.d_model)
            seq_len = tgt_input.size(1)
            causal_mask = self._generate_square_subsequent_mask(seq_len)
            if (
                tgt_key_padding_mask is not None
                and seq_len < tgt_key_padding_mask.size(1)
            ):
                tgt_key_padding_mask = tgt_key_padding_mask[:, :seq_len]
        else:
            with torch.no_grad():
                output = self.forward(tgt_input[:, :-1], tgt_key_padding_mask)
                next_tokens = output.argmax(-1)[:, -1:]
                tgt_input = torch.cat((tgt_input[:, :-1], next_tokens), dim=1)
            tgt_emb = self.token_embedding(tgt_input) * math.sqrt(self.d_model)
            seq_len = tgt_input.size(1)
            causal_mask = self._generate_square_subsequent_mask(seq_len)
            if (
                tgt_key_padding_mask is not None
                and seq_len < tgt_key_padding_mask.size(1)
            ):
                tgt_key_padding_mask = tgt_key_padding_mask[:, :seq_len]

        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.input_norm(tgt_emb)

        output = self.transformer_decoder(
            tgt_emb, mask=causal_mask, src_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)

    def generate(self, prompt_ids, max_len, end_token_id, pad_token_id, num_beams=8):
        self.eval()
        batch_size = prompt_ids.size(0)
        beams = [(prompt_ids.clone(), 0.0)]
        completed = []

        for _ in range(max_len - prompt_ids.size(1)):
            new_beams = []
            for beam, score in beams:
                padding_mask = beam == pad_token_id

                tgt_emb = self.token_embedding(beam) * math.sqrt(self.d_model)
                tgt_emb = self.positional_encoding(tgt_emb)
                tgt_emb = self.input_norm(tgt_emb)

                causal_mask = self._generate_square_subsequent_mask(beam.size(1))

                output = self.transformer_decoder(
                    tgt_emb, mask=causal_mask, src_key_padding_mask=padding_mask
                )

                next_logits = self.fc_out(output[:, -1, :])
                probs = torch.softmax(next_logits, dim=-1)
                top_probs, top_idx = probs.topk(num_beams, dim=-1)

                for i in range(num_beams):
                    next_token = top_idx[:, i].unsqueeze(1)
                    next_score = (
                        score - torch.log(top_probs[:, i]).mean().item()
                    )  # Trung bình log-prob cho batch
                    new_beam = torch.cat((beam, next_token), dim=1)
                    new_beams.append((new_beam, next_score))

            new_beams = sorted(new_beams, key=lambda x: x[1])[:num_beams]
            beams = []
            for beam, score in new_beams:
                if beam[:, -1].item() == end_token_id:
                    completed.append((beam, score))
                else:
                    beams.append((beam, score))

            if len(completed) >= num_beams or not beams:
                break

        if not completed:
            completed = beams

        best_beam, _ = min(completed, key=lambda x: x[1])

        preds = []
        for ids in best_beam:
            ids_list = ids.cpu().tolist()
            end_idx = (
                ids_list.index(end_token_id)
                if end_token_id in ids_list
                else len(ids_list)
            )
            preds.append(
                tokenizer.decode(
                    ids_list[prompt_ids.size(1) : end_idx], skip_special_tokens=True
                )
            )

        return best_beam, preds


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
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask=None, **generation_params):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generation_params
        )


if __name__ == "__main__":
    print_device_info()

    vocab_size = 30000
    marian = MarianMTWrapper("Helsinki-NLP/opus-mt-en-vi").to(device)
    transformer = TransformerScratchModel(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size
    ).to(device)
    gpt_scratch = ScratchGPTModel(vocab_size=vocab_size).to(device)
    gpt2 = GPT2FineTunedWrapper("gpt2").to(device)

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
