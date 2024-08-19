import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
def main():
    
    
    # Get Groq API key
    GROQ_API_KEY="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"
    groq_api_key = ''  # Replace 'your_api' with your actual API key

    # The title and greeting message of the Streamlit application
    st.title("Chat with your AI")
    st.write("Hello! I'm your friendly  chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a question:")

   
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input':message['human']},
                {'output':message['AI']}
                )


    
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )


    
    if user_question:

        
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),  

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  
            ]
        )

        
        conversation = LLMChain(
            llm=groq_chat,  
            prompt=prompt,  
            verbose=True,   
            memory=memory,  
        )
        
        
        response = conversation.predict(human_input=user_question)
        message = {'human':user_question,'AI':response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
