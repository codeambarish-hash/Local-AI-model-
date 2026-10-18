from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader
from docx import Document
import torch
import os

app = Flask(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model = model.to("cpu")
model.eval()

print("Qwen loaded successfully!")

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Answer clearly and accurately."
        )
    }
]

document_text = ""


def read_txt(filepath):

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        return file.read()


def read_pdf(filepath):

    reader = PdfReader(filepath)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_docx(filepath):

    document = Document(filepath)

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text


def extract_document(filepath):

    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".txt":
        return read_txt(filepath)

    elif extension == ".pdf":
        return read_pdf(filepath)

    elif extension == ".docx":
        return read_docx(filepath)

    return ""


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    global document_text

    if "file" not in request.files:

        return jsonify({
            "error": "No file selected."
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "error": "No file selected."
        }), 400

    allowed_extensions = {
        ".txt",
        ".pdf",
        ".docx"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:

        return jsonify({
            "error": "Only TXT, PDF and DOCX files are supported."
        }), 400

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    try:

        document_text = extract_document(filepath)

    except Exception as error:

        return jsonify({
            "error": f"Could not read document: {error}"
        }), 500

    if not document_text.strip():

        return jsonify({
            "error": "No readable text found in the document."
        }), 400

    # Limit text for the current basic version
    document_text = document_text[:12000]

    return jsonify({
        "message": "Document uploaded successfully.",
        "filename": file.filename,
        "characters": len(document_text)
    })


@app.route("/chat", methods=["POST"])
def chat():

    global messages
    global document_text

    data = request.get_json()

    user_message = data.get(
        "message",
        ""
    ).strip()

    if not user_message:

        return jsonify({
            "error": "Message cannot be empty."
        }), 400

    if document_text:

        context = (
            "The user uploaded a document. "
            "Use the document content below to answer "
            "questions when relevant.\n\n"
            "DOCUMENT CONTENT:\n"
            + document_text
            + "\n\n"
        )

        messages_for_model = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Answer clearly and accurately. "
                    "When the user asks about the uploaded "
                    "document, use the provided document content."
                )
            },
            {
                "role": "system",
                "content": context
            }
        ]

        messages_for_model.extend(messages[1:])

    else:

        messages_for_model = messages

    messages_for_model.append({
        "role": "user",
        "content": user_message
    })

    prompt = tokenizer.apply_chat_template(
        messages_for_model,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05
        )

    new_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    messages.append({
        "role": "user",
        "content": user_message
    })

    messages.append({
        "role": "assistant",
        "content": response
    })

    return jsonify({
        "response": response
    })


@app.route("/clear", methods=["POST"])
def clear():

    global messages
    global document_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Answer clearly and accurately."
            )
        }
    ]

    document_text = ""

    return jsonify({
        "message": "Conversation and document context cleared."
    })


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("QWEN LOCAL DOCUMENT AI")
    print("=" * 60)
    print("Open:")
    print("http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )