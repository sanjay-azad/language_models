import streamlit as st
import pandas as pd
import openai
from glob import glob
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from message_log import log
from PIL import Image
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key  = st.secrets['OPENAI_API_KEY']

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load the data from vectordb
@st.cache_resource
def knowledge_base():
    persist_directory = 'chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory = persist_directory, embedding_function = embedding)
    # print('document count:', vectordb._collection.count())
    return vectordb

# Initiate chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0.4)
def generate_response(question):
    # question = "what is your take - humans have will power or everything is pre decided?"
    # docs = knowledge_base.similarity_search(question, k = 3)
    
    # Retrieval of relevant documents
    qa_chain = RetrievalQA.from_chain_type(
        chain_type='stuff',
        llm = llm,
        retriever = knowledge_base().as_retriever(search_type="similarity"),
        return_source_documents = True
    )

    # Response generation
    result = qa_chain({"query": question})
    print('Answer:', result["result"])
    print('Documents:\n', result['source_documents'])
    return result["result"]

# Site logo
image = Image.open('logo.jpeg')

# User interface 
with st.sidebar:
    st.image(image, width = 300)
    st.markdown("## [Randomness and Structure](https://sanjay-azad.github.io/home/)")
    st.write("Ask anything about the Existence, Spirituality, and Science")
    st.divider()
    st.caption("Powered by OpenAI & Langchain")

    if st.button("Reset Conversation"):
        st.session_state.clear()
        st.experimental_rerun()


# Dispaly all messages from history of session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Enter your query")
if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating repsonse..."):
            response = generate_response(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    log(prompt, response)
    st.experimental_rerun()

