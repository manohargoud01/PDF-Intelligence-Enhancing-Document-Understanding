

# TCIT Chatbot

This Python script implements a chatbot application, specifically designed to answer questions based on PDF documents. It utilizes Gradio for building the user interface and leverages the LangChain library for natural language processing tasks.

## Requirements

- Python 3.x
- Gradio
- LangChain
- Other necessary libraries (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository or download the script files.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Set up the required environment variables:

```bash
export HUGGINGFACEHUB_API_TOKEN='--'
```

4. Run the `tcit_chatbot.py` script:

```bash
python tcit_chatbot.py
```

## Usage

- Step 1: Process Document
    - Adjust advanced options such as chunk size and chunk overlap for document processing.
    - Generate the vector database for the PDF document.

- Step 2: Chatbot
    - Interact with the chatbot by typing messages and submitting them.
    - Clear the conversation or reset the chatbot state.

## Functionality

- The chatbot utilizes a pre-trained language model (LLM) to answer questions based on the processed PDF document.
- It maintains a conversation history and provides references to specific pages within the document.

