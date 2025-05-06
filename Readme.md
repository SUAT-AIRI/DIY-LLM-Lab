# DIY-LLM-Lab

**DIY-LLM-Lab** is an experimental project aimed at collecting and organizing practical utility scripts related to Large Language Models (LLMs).

## 🧰 Project Structure

The project follows a modular structure for better maintenance and scalability:

```
DIY-LLM-Lab/
├── data_utils/        # Data processing and analysis tools
│   └── count_tokens_stats.py  # Script for counting text token statistics
├── inference/         # Inference-related tools (coming soon)
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

## 📌 Roadmap

* Add more utility scripts related to LLM workflows.
* Support a wider range of tokenizers and models.
* Improve the performance and usability of existing tools.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to help improve this project.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
