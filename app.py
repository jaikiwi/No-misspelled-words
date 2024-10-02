import streamlit as st
import os
from groq import Groq
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

class MessageHistory:
    def __init__(self, messages):
        self.messages = messages

def convert_chat_history_to_messages(chat_history):
    messages = []
    for message in chat_history:
    
        human_msg = HumanMessage(content=str(message["human"]))
        ai_msg = AIMessage(content=" ".join(message["AI"]) if isinstance(message["AI"], list) else str(message["AI"]))
        messages.extend([human_msg, ai_msg])
    return MessageHistory(messages)

def main():
    st.title("Groq Chat App")
    st.sidebar.title("Select an LLM")

    model = st.sidebar.selectbox(
        "Choose a model",
        ["mixtral-8x7b-32768", "llava-v1.5-7b-4096"]
    )

    conversational_memory_length = st.sidebar.slider("Conversational Memory Length:", 1, 10, value=5)
    memory = ConversationBufferMemory(k=conversational_memory_length)
    user_question = st.text_area("Ask The Question...")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    groq_chat = ChatGroq(
        groq_api_key = groq_api_key,
        model_name=model
    )

    def get_session_history():
        return convert_chat_history_to_messages(st.session_state.chat_history)

    conversation = RunnableWithMessageHistory(
        runnable=groq_chat,
        get_session_history=get_session_history
    )

    if user_question:
        response = conversation.invoke(user_question)

        if isinstance(response, list):
            response = " ".join([str(item) for item in response])

        message = {"human": str(user_question), "AI": str(response)}
        st.session_state.chat_history.append(message)
        st.write("chatbot:", response)

if __name__ == "__main__":
    
    main()
