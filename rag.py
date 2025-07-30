import streamlit as st 
import os
from dotenv import load_dotenv
load_dotenv()

from PyPDF2 import PdfReader
import google.generativeai as genai
key = os.getenv('GOOGLE_API_KEY')

from langchain_huggingface import HuggingFaceEmbeddings # to get embedding model
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter # to split the raw text into chunks
from langchain_community.vectorstores import FAISS


genai.configure(api_key= key)

gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# @st.cache_data(show_spinner='Loading Embedding Model...')

def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
with st.spinner('Loading Embedding Model.....'):
    embedding_model = load_embedding()
st.header('RAG Assistant :blue[Using Embedding & Gemini LLM]')
st.subheader('Your Intelligent Document Assistant')
st.write('Done')
uploaded_file = st.file_uploader('Upload the document here in PDF format',type=['pdf'])

if uploaded_file:
    st.write('Uploaded Successfully')

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    raw_text =''
    
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    st.write('Extracted successfully')
    if raw_text.strip():
        doc = Document(page_content =raw_text,metadata={'source':uploaded_file.name})
        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunk_text = splitter.split_documents([doc])
        text = [i.page_content for i in chunk_text]
        vector_db = FAISS.from_texts(text,embedding_model)
        retrive = vector_db.as_retriever()
        
        st.success('Document Processed Successfully.... Ask question now.')
        query = st.text_input('Enter your query here:')
        if query:
            with st.chat_message('human'):
                with st.spinner('Analyzing the Document'):
                    relevant_docs = retrive.get_relevant_documents(query)
                    content = '\n\n'.join([i.page_content for i in relevant_docs])
                    st.write(relevant_docs)

                    prompt = f'''
                    You are an AI expert. Use the content given to answer the query asked by the user. 
                    If you are unsure you should say 'I am unsure about the question asked'
                    Content:{content}
                    Query:{query}
                    Result:
                    '''
                    response = gemini_model.generate_content(prompt)
                    st.markdown(':[Result]')
                    st.write(response.text)
    else:
        st.warning('Drop the file in proper PDf format')