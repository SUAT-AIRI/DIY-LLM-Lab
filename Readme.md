# DIY-LLM-Lab

**DIY-LLM-Lab** is an experimental project aimed at collecting and organizing practical utility scripts related to Large Language Models (LLMs).

## 🧰 Project Structure

The project follows a modular structure for better maintenance and scalability:

```
DIY-LLM-Lab/
├── data_utils/        # Data processing and analysis tools
│   └── count_tokens_stats.py  # Script for counting text token statistics
├── inference/         # Inference-related tools
│   ├── inference_single.py    # Single-GPU inference with progress bar
│   └── inference_multi.py     # Multi-GPU inference via Accelerate
├── evaluation/        # Model evaluation tools (coming soon)
├── README.md          # Project documentation
└── requirements.txt   # Project dependencies
```

## 🚀 Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/SUAT-AIRI/DIY-LLM-Lab.git
   cd DIY-LLM-Lab
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Example - Run the token counting script:

   ```bash
   # Count tokens using GPT‑4o tokenizer and export CSV report
   python data_utils/count_tokens_stats.py r1_interactive_filter.json --model gpt-4o --csv
   ```

   Supported tokenizers include both `tiktoken` and Hugging Face `transformers`.

4. Example - Run single-GPU inference:

   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   python inference/inference_single.py \
     --model /path/to/model \
     --input /path/to/input.json \
     --output /path/to/output.json \
     --prompt_max_len 4096 \
     --max_new_tokens 512 \
     --bf16 \
     --batch_size 1
   ```

5. Example - Run multi-GPU inference with Accelerate:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   accelerate launch inference/inference_multi.py \
     --model /path/to/model \
     --input /path/to/input.json \
     --output /path/to/output.json \
     --prompt_max_len 4096 \
     --max_new_tokens 512 \
     --batch_size 4 \
     --bf16
   ```

## 📌 Roadmap

* Add more utility scripts related to LLM workflows.
* Support a wider range of tokenizers and models.
* Improve the performance and usability of existing tools.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to help improve this project.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
