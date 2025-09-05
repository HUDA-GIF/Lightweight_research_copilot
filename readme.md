
# 📚 Lightweight Research Copilot
📚 Lightweight Research Copilot (Optimized)

An interactive Streamlit web app that helps researchers, students, and knowledge workers summarize text and generate quizzes.
It supports both fast extractive summarization (using NLTK + TextRank) and advanced abstractive summarization (using transformer models via Hugging Face).

✨ Features

Summarization Methods

⚡ Fast (Extractive): Uses NLTK + NetworkX TextRank for quick results.

🤖 Advanced (Transformer): Uses facebook/bart-large-cnn for summarization and deepset/minilm-uncased-squad2 for Q&A.

Quiz Generation

Automatically generates comprehension questions from the summarized text.

Supports both generic question sets and QA-model-generated questions.

Performance Monitoring

Displays estimated processing time, memory usage, and model status.

Fallback Handling

Works even if transformers is not installed or system memory is limited.

Automatically downloads required NLTK resources (punkt, stopwords).

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/your-username/lightweight-research-copilot.git
cd lightweight-research-copilot

2. Install Dependencies

Create a virtual environment (recommended) and install dependencies:

pip install -r requirements.txt


requirements.txt

streamlit
nltk
numpy
networkx
psutil
transformers

3. Run the App
streamlit run app.py

🛠 Usage

Paste your text into the input box.

Choose a summarization method in the sidebar:

Fast (Extractive)

Advanced (Transformer) (if available)

Adjust settings:

Number of sentences / summary length.

Enable or disable quiz generation.

Click Analyze Text.

View:

✅ Summary

📌 Quiz questions

⚡ Performance metrics
