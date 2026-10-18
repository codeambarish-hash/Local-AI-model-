# Local-AI-model-
Qwen2.5-0.5B AI chatbot


# Qwen AI Chatbot 🤖

A lightweight AI chatbot powered by **Qwen2.5-0.5B-Instruct** using Python and Hugging Face Transformers. Run the model locally on your computer, interact through CMD, or launch a Flask-based web interface.

## ✨ Features

- Local AI text generation
- CMD-based chatbot
- Flask web interface
- PDF, TXT, and DOCX document processing (if enabled in the web app)
- Runs on CPU without requiring an NVIDIA GPU
- Model weights downloaded automatically from Hugging Face on the first run

## 🛠️ Tech Stack

- Python
- Qwen2.5-0.5B-Instruct
- Hugging Face Transformers
- PyTorch
- Flask
- pypdf
- python-docx

## 📋 Prerequisites

Install the following before starting:

- Python 3.11–3.13
- Git
- Visual Studio Code (optional)

Check your Python and Git installations in Windows CMD:

```cmd
python --version
git --version
```

If a command is not recognized, install the missing software and reopen CMD.

## 🚀 Step 1: Create the project folder

Open CMD and run:

```cmd
cd /d "%USERPROFILE%\Desktop"
mkdir Qwen-Chatbot
cd Qwen-Chatbot
```

If the folder already exists, do not create it again. Navigate to your existing project folder instead.

## 🐍 Step 2: Create a virtual environment

```cmd
python -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your CMD prompt.

## 📦 Step 3: Install the required packages

Upgrade pip:

```cmd
python -m pip install --upgrade pip
```

Install the dependencies:

```cmd
pip install torch transformers flask pypdf python-docx
```

Check that the main libraries are installed:

```cmd
python -c "import torch, transformers, flask; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Flask:', flask.__version__)"
```

**Note:** PyTorch and Transformers may take time to install. The first model run also requires an internet connection to download the model files.

## 🧠 Step 4: Download and test the Qwen model

Create a Python file named `test_qwen.py` and add:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading AI model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

messages = [
    {
        "role": "user",
        "content": "Hello! Introduce yourself as an AI assistant."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True
)

print("\nAI Response:")
print(response)
```

Run the model:

```cmd
python test_qwen.py
```

The model will download automatically the first time. Once loading finishes, the chatbot should print an AI-generated response.

The initial download and model loading may take several minutes depending on your internet speed and computer.

## 💬 Step 5: Run the CMD chatbot

If your project contains `chatbot.py`, execute:

```cmd
python chatbot.py
```

Type a message and press Enter to interact with the chatbot.

To exit, use the exit command supported by your script, or press `Ctrl+C`.

## 🌐 Step 6: Run the Flask web application

If your project contains `app.py` and `templates/index.html`, run:

```cmd
python app.py
```

Open the local URL printed in CMD. For a standard Flask configuration, it is often:

```text
http://127.0.0.1:5000
```

Keep CMD running while you use the website. Press `Ctrl+C` in CMD to stop the server.

## 📁 Step 7: Recommended project structure

```text
Qwen-Chatbot/
├── app.py
├── chatbot.py
├── test_qwen.py
├── requirements.txt
├── .gitignore
├── README.md
├── templates/
│   └── index.html
└── uploads/
```

Your actual project may contain additional files. Keep only the files your implementation uses.

## 📄 Step 8: Create requirements.txt

With your virtual environment activated, run:

```cmd
python -m pip freeze > requirements.txt
```

This records your installed Python package versions for reference.

## 🔒 Step 9: Create .gitignore

Create a file named `.gitignore` in your project root:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environment
venv/
.venv/

# Environment variables and secrets
.env
.env.*

# Uploaded user documents
uploads/*
!uploads/.gitkeep

# Local model/cache files
models/
.cache/
huggingface_cache/

# Editor and OS files
.vscode/
.DS_Store
Thumbs.db
```

Do not commit virtual environments, private documents, API keys, or downloaded model-weight files. Hugging Face normally stores downloaded model files in its cache outside your repository.

If your app uses a different upload folder or model cache path, update `.gitignore` accordingly.

## 🐙 Step 10: Upload the project to GitHub

First, create a new **empty repository** on GitHub. For example:

`Qwen-Chatbot`

Do not add a README on GitHub if you have already created one locally.

Open CMD in your project folder:

```cmd
cd /d "%USERPROFILE%\Desktop\Qwen-Chatbot"
```

Initialize Git:

```cmd
git init
git add .
git status
```

Review the listed files. Ensure that no virtual environment, private documents, model weights, or secrets are included.

Commit the project:

```cmd
git commit -m "Initial commit: Qwen AI chatbot"
```

Set the main branch:

```cmd
git branch -M main
```

Connect your GitHub repository (replace the example URL with your actual repository URL):

```cmd
git remote add origin https://github.com/YOUR_USERNAME/Qwen-Chatbot.git
```

Push the project:

```cmd
git push -u origin main
```

Follow GitHub's authentication instructions if prompted.

### Updating the repository later

Whenever you make changes:

```cmd
git add .
git commit -m "Update chatbot"
git push
```

## ⚠️ Troubleshooting

**Python is not recognized**

Reinstall Python and enable the option to add Python to PATH, then reopen CMD.

**`No module named transformers`**

Activate the virtual environment and install the packages:

```cmd
venv\Scripts\activate
pip install torch transformers flask pypdf python-docx
```

**Model download fails**

Check your internet connection, Hugging Face access, available disk space, and the error message. Retry after resolving the issue.

**Out of memory**

Close unnecessary applications and reduce `max_new_tokens`. The model may still require substantial RAM depending on the execution environment.

**Port 5000 is already in use**

Stop the other application using that port or configure your Flask application to use another available port.

**Git push is rejected**

Confirm the repository URL, authentication, and whether the remote repository already contains commits. Do not force-push unless you understand the consequences.

## 🔐 Security and deployment

This project is intended for local development unless you have separately secured and deployed the web application.

- Do not expose Flask's debug server to the public internet.
- Validate uploaded file types and file sizes.
- Use unique, safely generated filenames for uploads.
- Keep uploaded documents private and delete temporary files when appropriate.
- Use per-user sessions and storage for multi-user deployments.
- Do not commit credentials or private documents to GitHub.

A public GitHub repository contains your source code; it does **not** automatically host the AI model or run the chatbot.

For a public website, deploy the frontend and backend to suitable hosting services and configure a hosted inference provider or a securely hosted model API. Your local Qwen model will not run on Vercel simply because its code is uploaded to GitHub.

## 📜 License

Check the licenses of Qwen and all dependencies before redistributing or deploying the model. Add a license file to your repository if you intend to license your own code.

---

**Built with Python, PyTorch, Hugging Face Transformers, and Qwen2.5-0.5B-Instruct.**
