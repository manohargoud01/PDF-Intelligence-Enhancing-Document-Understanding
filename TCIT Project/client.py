import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from pathlib import Path
import chromadb
from unidecode import unidecode
from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate
import re

# Define the default LLM model and parameters
default_llm = "mistralai/Mistral-7B-Instruct-v0.2"
default_temperature = 0.7
default_max_tokens = 1024
default_top_k = 3

def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(str(x)) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name
    )
    return vectordb

def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding)
    return vectordb

def initialize_llmchain(vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    progress(0.5, desc="Initializing HF Hub...")
    llm = HuggingFaceEndpoint(
            repo_id=default_llm, 
            temperature=default_temperature,
            max_new_tokens=default_max_tokens,
            top_k=default_top_k,
        )
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain

def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

def initialize_database(chunk_size, chunk_overlap, progress=gr.Progress()):
    # Path of the PDF file environment
    pdf_file_path = "/workspaces/codespaces-django/TCIT Project/tatastrive-data.pdf"  

    list_file_path = [pdf_file_path]
    collection_name = create_collection_name(list_file_path[0])
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    vector_db = create_db(doc_splits, collection_name)
    return vector_db, collection_name, "Complete!"

def initialize_qa_chain(vector_db):
    qa_chain = initialize_llmchain(vector_db)
    return qa_chain, "Complete!"

def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page

def demo():
    # Define empty states for chatbot and document references
    chatbot = gr.State()
    doc_source1 = gr.State()
    source1_page = gr.State()
    doc_source2 = gr.State()
    source2_page = gr.State()
    doc_source3 = gr.State()
    source3_page = gr.State()

    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        gr.Markdown("<center><h2>TCIT chatbot</center></h2><h3>Ask any questions about your PDF documents</h3>")
        
        with gr.Tab("Step 1 - Process document"):
            with gr.Accordion("Advanced options - Document text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(minimum = 100, maximum = 1000, value=600, step=20, label="Chunk size", info="Chunk size", interactive=True)
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(minimum = 10, maximum = 200, value=40, step=10, label="Chunk overlap", info="Chunk overlap", interactive=True)
            with gr.Row():
                db_progress = gr.Textbox(label="Vector database initialization", value="None")
            with gr.Row():
                db_btn = gr.Button("Generate vector database")
            db_btn.click(initialize_database, \
                inputs=[slider_chunk_size, slider_chunk_overlap], \
                outputs=[vector_db, collection_name, db_progress])

            llm_progress = gr.Textbox(value="None", label="QA chain initialization")
            qachain_btn = gr.Button("Initialize Question Answering chain")
            qachain_btn.click(initialize_qa_chain, \
                inputs=[vector_db], \
                outputs=[qa_chain, llm_progress])

        with gr.Tab("Step 2 - Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                    source2_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                    source3_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type message (e.g. 'What is this document about?')", container=True)
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear conversation")

            msg.submit(conversation, \
                inputs=[qa_chain, msg, chatbot], \
                outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
                queue=False)
            submit_btn.click(conversation, \
                inputs=[qa_chain, msg, chatbot], \
                outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
                queue=False)
            clear_btn.click(lambda:[None,"",0,"",0,"",0], \
                inputs=None, \
                outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
                queue=False)

    demo.queue().launch(debug=True)

if __name__ == "__main__":
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    demo()
