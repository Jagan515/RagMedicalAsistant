# RAG Medical Chatbot 

> 🌈 **Welcome!** This README gives a clear, colorful, copy-paste-ready guide to run your Flask + RAG app (`app.py`), create a Python venv, build the Docker image, and deploy to **Render**.
> It includes the `Dockerfile`, venv commands, environment variables, quick troubleshooting, and useful links.

---

## 📌 Project overview

A Flask app that uses:

* Hugging Face embeddings (downloaded via helper)
* Pinecone vector store (`PineconeVectorStore`)
* A conversational LLM (`ChatOpenAI` / OpenRouter endpoint)
* A Runnable RAG chain for retrieval → prompt → LLM → parsed output

Files of interest:

* `app.py` — Flask server (main)
* `src/helper.py` — embedding + helper utilities
* `src/prompt.py` — `prompt_template`
* `requirements.txt` — Python deps
* `Dockerfile` — container image for deployment

---

# 🔧 `README.md` 

````markdown
# RAG Medical Chatbot

A Flask-based Retrieval-Augmented Generation (RAG) chatbot that:
- retrieves similar docs from Pinecone,
- builds a prompt from retrieved docs,
- queries an LLM (via OpenRouter / ChatOpenAI),
- returns JSON responses to a client UI.

---

## 🚀 Quick start (local, with virtualenv)

> These commands assume a Unix-like shell (macOS / Linux). Windows commands are shown after.

1. Clone repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
````

2. Create & activate virtual environment

```bash
# Python 3.13+ recommended
python3 -m venv env
source env/bin/activate        # macOS / Linux
# OR on Windows PowerShell:
# python -m venv env
# .\env\Scripts\Activate.ps1   # PowerShell
# OR on Windows cmd:
# .\env\Scripts\activate.bat
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Export environment variables (create a `.env` in repo root)

```
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
# Optionally
PINECONE_ENVIRONMENT=your_pinecone_env
PORT=8080
```

5. Run the app (development)

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=8080
# OR
python app.py
```

Open `http://localhost:8080` and use the chat UI.

---

## 🐳 Docker (build & run locally)

**Dockerfile**

```dockerfile
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose Render-assigned port
EXPOSE $PORT

# Start app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "3"]
```

Build & run:

```bash
docker build -t rag-medical-chatbot:latest .
# run (set PORT, API keys as env)
docker run -e PORT=8080 -e PINECONE_API_KEY=$PINECONE_API_KEY -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY -p 8080:8080 rag-medical-chatbot:latest
```

---

## ☁️ Deploy on Render (Web Service)

1. Create a new **Web Service** on Render and connect your Git repository.
2. Set the build & start commands (Render auto-detects Python, but set explicitly):

   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 3`
3. Add environment variables in Render dashboard:

   * `PINECONE_API_KEY`
   * `OPENROUTER_API_KEY`
   * (optional) `PINECONE_ENVIRONMENT`, `TOKENIZERS_PARALLELISM=false`
4. Select instance plan (free hobby is fine for testing).
5. Deploy — Render’s build + deploy pipeline runs and your app becomes reachable at the assigned URL.

> ⚠️ Note: Render's deployment pipeline times vary; builds that install large models or heavy dependencies may take longer. Ensure required env vars are set before deploying.

---

## 🧩 `app.py` (reference / main file)

> The application uses your `src` helpers and `prompt_template`. Add any missing helper implementations into `src/helper.py` and `src/prompt.py`.

```python
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, docs_to_prompt_input
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from src.prompt import prompt_template
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

# Env vars
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY", "")
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Embeddings + Pinecone vectorstore
embeddings = download_hugging_face_embeddings()

index_name = "rag-medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM (ChatOpenAI)
chat = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

parser = StrOutputParser()

docs_to_prompt_runnable = RunnableLambda(
    lambda inputs: docs_to_prompt_input(inputs["docs"], inputs["question"])
)

question_answer_chain = RunnableLambda(
    lambda inputs, _chat=chat: _chat.invoke(prompt.format(**inputs))
)

initial_doc_runnable = RunnableLambda(
    lambda inputs: {
        "docs": docsearch.similarity_search(inputs["question"], k=3),
        "question": inputs["question"],
    }
)

rag_chain = (
    initial_doc_runnable
    | docs_to_prompt_runnable
    | question_answer_chain
    | parser
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat_route():
    try:
        msg_text = request.form.get("msg") or request.args.get("msg") or ""
        logger.info("User message: %s", msg_text)

        result = rag_chain.invoke({"question": msg_text})
        logger.info("RAG result: %s", result)

        return jsonify({"response": str(result)})

    except Exception as e:
        logger.exception("Error while processing request")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
```

---

## ✅ Environment / Required Keys (summary)

Add these to your `.env` or Render environment settings:

* `PINECONE_API_KEY` — Pinecone API key
* `PINECONE_ENVIRONMENT` — Pinecone environment (if required)
* `OPENROUTER_API_KEY` — OpenRouter / LLM key
* `PORT` — port number (Render sets this automatically)

---

## 🧰 Useful links (languages / tools used)

* Python — [https://www.python.org/](https://www.python.org/)
* Flask — [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
* Gunicorn — [https://gunicorn.org/](https://gunicorn.org/)
* Docker — [https://www.docker.com/](https://www.docker.com/)
* Render — [https://render.com/](https://render.com/)
* Pinecone — [https://www.pinecone.io/](https://www.pinecone.io/)
* Hugging Face — [https://huggingface.co/](https://huggingface.co/)
* LangChain (core / runnables) — [https://python.langchain.com/](https://python.langchain.com/)
* OpenRouter — [https://openrouter.ai/](https://openrouter.ai/)

---

## 🛠 Troubleshooting & tips

* **Model errors / timeouts**: Make sure LLM API key and `base_url` are correct and the model name is available for your account.
* **Pinecone index missing**: Ensure index `rag-medical-chatbot` exists and contains vectors.
* **Tokenizers parallelism**: If you see warnings about tokenizers, ensure `TOKENIZERS_PARALLELISM=false` is set.
* **Large dependency installs on Render**: Use wheels or smaller base images, cache builds via Docker if build times are large.

---

## 🧪 Testing locally

* Test retrieval separately: call `docsearch.similarity_search("test question", k=3)` in a small script to validate vectorstore.
* Test prompt formatting: call `prompt.format(context="...", question="...")` to verify input variables.

---

## ❤️ A few best practices

* Use a `.env` for local development; never commit real keys to Git.
* Freeze dependencies: `pip freeze > requirements.txt` before pushing.
* Keep model & index names configurable via env vars (avoid hard-coded names).
* Add basic healthcheck endpoint `/health` returning `200 OK`.

---


Happy hacking! 🧑‍💻🚀


