__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#from dotenv import load_dotenv
#load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
#from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import tempfile
import os

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF파일을 선택해 주세요.",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)


    #Loader
    #loader = PyPDFLoader("addr.pdf")
    #pages = loader.load_and_split()

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300, # 100글자 단위로
        chunk_overlap  = 20, # 중간에 짤리면 이상할 수 있으므로 오버랩하는 부분은 중복되도록 하는 거다.
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)

    #Embedding
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    from langchain.vectorstores import Chroma
    db = Chroma.from_documents(texts, embeddings_model)  # 여기까지가 DB에 저장까지 한거다. 디렉토리를 설정하면 저장해서 쓸수 있겠다.
                                                        # (texts, embeddings_model, persist_directory="/chroma")

    #Question
    st.header("PDF에게 질문해 보세요.")
    question = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        with st.spinner('wait for it...'):
        #질문 답변하기
        # from langchain.chains import RetrievalQA
        # llm = ChatOpenAI(temperature=0)
        # retriever_from_llm = MultiQueryRetriever.from_llm(
        #     retriever=db.as_retriever(), llm=llm
        # )

        #질문과 답변
            from langchain.chains import RetrievalQA
            llm = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])