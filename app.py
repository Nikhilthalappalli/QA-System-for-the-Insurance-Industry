import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


DB_FAISS_PATH = 'vectorstore/db_faiss' #path to the vectorstore

def get_vectorstore(DB_FAISS_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
    return vectorstore

def conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":2}),#adjust the value k based on the computational power
        memory=memory 
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
            
            
def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Question-Answering System for the Insurance Industry")
    if st.button("Start"):
        with st.spinner("Processing"):
            vectorstore = get_vectorstore(DB_FAISS_PATH)
            st.session_state.conversation = conversation_chain(vectorstore)
    
    user_question = st.text_input("Enter the questions",placeholder="")
    if user_question:
        handle_userinput(user_question)
    
if __name__ == '__main__':
    main()