# Language Model API

This project uses opensource LLM models from huggingface with langchain to create a simple chatbot.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
uvicorn language_model.main:app --reload
```

call the api using curl, change the text to whatever you want to ask the chatbot

```bash
curl -L 'http://localhost:8000/chat' -H 'Content-Type: application/json' -d '{
    "text":"what is my favourite food?"
}'
```