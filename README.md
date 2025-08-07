# 🧠 Notebook2UI – Instantly Convert Jupyter Notebooks to Streamlit Apps

**Notebook2UI** is a Streamlit app that lets you upload a `.ipynb` (Jupyter Notebook) file and instantly receive:
- A concise summary of your notebook.
- Streamlit code that mirrors your notebook’s logic and visualizations.
- Downloadable `.py` script and `requirements.txt` for easy deployment.

Powered by **OpenAI's GPT-4.1-mini** for fast, accurate code and summary generation.

---

## 🚀 Features

- 📤 Upload Jupyter Notebooks
- 🧠 Automatic notebook summarization
- 💻 Streamlit UI code generation
- 🧾 Downloadable `.py` and `requirements.txt`
- 🧪 In-browser app UI preview
- 🛠️ Robust error handling for invalid or unsupported notebooks

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Python 3.10+](https://www.python.org/)
- [OpenAI GPT-4.1-mini](https://platform.openai.com/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [nbformat](https://pypi.org/project/nbformat/)

---

## 📦 Requirements

Install dependencies before running the app:

```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/notebook2ui.git
    cd notebook2ui
    ```

2. **Set up your environment:**
    - Create a `.env` file in the root directory with your OpenAI API key:
      ```
      OPENAI_API_KEY=your_openai_api_key_here
      ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## 🖼️ App Images

_Add screenshots or GIFs of the app here._

---

## 🧠 Powered By

- OpenAI GPT-4.1-mini
- Streamlit

---

## 📄 License

MIT License © 2025 

---

_Let me know if you’d like to see the project structure or need deployment instructions for Streamlit Cloud, Hugging Face Spaces, or other platforms._
