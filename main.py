__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from openai import ChatCompletion
import streamlit as st
import tempfile
import os

# 제목
st.title("CSLEE's ChatPDF")
st.write("---")

# 파일 업로드
uploaded_files = st.file_uploader("PDF 파일을 선택해 주세요.", type=['pdf'], accept_multiple_files=True)
st.write("---")

def pdfs_to_documents(uploaded_files):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepaths = []
    for uploaded_file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        temp_filepaths.append(temp_filepath)
    return temp_filepaths

# 업로드 되면 동작하는 코드
if uploaded_files:
    temp_filepaths = pdfs_to_documents(uploaded_files)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 100글자 단위로
        chunk_overlap=20,  # 중간에 짤리면 이상할 수 있으므로 오버랩하는 부분은 중복되도록 하는 거다.
        length_function=len,
        is_separator_regex=False,
    )

    pages = []
    for temp_filepath in temp_filepaths:
        loader = PyPDFLoader(temp_filepath)
        pages.extend(loader.load_and_split())

    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("PDF에게 질문해 보세요.")
    question = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        with st.spinner('wait for it...'):
            # 질문 답변하기
            llm = ChatCompletion(model_name="davinci", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])