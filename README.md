```bash
/
├── data/
│ └── eng-vie.csv
├── src/
│ ├── data_utils.py # Xử lý dữ liệu, tokenizer, split train/val/test
│ ├── bpe_tokenizer.py # Training, load/save BPE tokenizer
│ ├── model_marianmt.py # Pipeline MarianMT: train, eval, save/load
│ ├── model_gpt2.py # Pipeline GPT-2: train, eval, save/load
│ ├── model_transformer.py # Pipeline Transformer scratch: train, eval, save/load
│ ├── model_gpt_scratch.py # Pipeline GPT scratch: train, eval, save/load
│ ├── eval_utils.py # Hàm tính BLEU, ROUGE, test inference, visualize
│ └── infer.py # Hàm infer/translate từng mô hình từ câu input
├── output/
│ └── ... # Lưu mô hình, tokenizer, checkpoint, metrics
├── main.ipynb # Notebook chính chạy các bước theo thứ tự, trực quan hóa
└── requirements.txt # Thư viện cần cài (dùng cho local hoặc Colab/Kaggle)
```
