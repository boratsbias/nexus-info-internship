#we will use Langchain here which is a framework designed for building applications that are powered by large language models. It provides tools and abstractions to facilitate the development of applications that can use LLm.
import streamlit as st #open source framework used for creating data apps.

import os # a module in python that uses the operating system's functionality like reading and writing files.

from langchain_groq import ChatGroq 
#this is a specialized module within the langchain that helps integrate with ChatGroq which is a class.

from langchain.text_splitter import RecursiveCharacterTextSplitter 
#langchain.text_splitter is a module that provides utilities for splitting text into smaller chunks. Here RecursiveCharacterTextSplitter is a specific text splitter that breaks down text recursively into smaller chunks. It is usefull for processing large texts into manageable chunks for further processing

from langchain.chains.combine_documents import create_stuff_documents_chain 
#langchain.chains.combine_documents is a module that provides utilities for combining multiple documents into a single chain. The create_stuff_documents_chain is a particular function that creates a chain(a sequence of processing steps) for combining and processing documents

from langchain_core.prompts import ChatPromptTemplate 
#langchain_core.prompts is a module that helps in creating and managing prompts for language models. The ChatPromptTemplate is a particular function that creates prompts for chat based interactions with language models.

from langchain.chains import create_retrieval_chain 
# chain creation utilities refer to tools and functionalities that help create and manage sequences of interactions with the language model. the langchain.chains is a module that provides various chain creation utilities. The create_retrieval_chain function fetches relevant documents or information in response to a query.

from langchain_community.vectorstores import FAISS 
#a vector is a numerical representation of data that can capture the essence or features of the data in a structured format. Each number in the vector corresponds to a specific feature or dimension of the data. The length of the vector corresponds to the dimensionality of the representation. Higher-dimensional vectors can capture more complex features. FAISS is a library for efficient similarity search and clustering of dense vectors. It stands for Facebook AI similarity search. Vectorstores are databases or systems designed to handle and search through vectors. 

from langchain_community.document_loaders import PyPDFDirectoryLoader 
#this module includes document loaders from the langchain community. PyPDF is a specific library to load PDF files.

from langchain_google_genai import GoogleGenerativeAIEmbeddings 
#langchain_google_genai module integrates Google's generative AI tools with langchain. GoogleGenerativeAIEmbeddings is a function to create embeddings using Google's generative AI models, which can convert text into numerical representation for various AI application.
from dotenv import load_dotenv # library to load the environment variables.

load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY') #retrieves the groq api key from the environment variable.
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY") #retrieves the google API key from the environment variable and sets it to the environment variable in the current process.

st.title("Admission Enquiry Bot ")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
) #this is used to create a prompt template. Template is a process of defining a well structured format or pattern that will guide how input is provided to a language model. the from_template method creates an instacnce from a given template string. 'Answer the questions based on the provided context only.' tells the model that responses should be based solely on the context provided. 'Please provide the most accurate response based on the question' asks the model to ensure accuracy in its answer. here context refers to the specific information or background that is provided to the language model to help it generate accurate and relevant reponses. the {context} placeholder will get replaced with the actual content when the prompt is used. and input is the question asked by the user.

def vector_embedding():

    if "vectors" not in st.session_state:
        #displaying a spinner with the message "Loading documents Please wait" to inform the user that the document loading process is ongoing.
        with st.spinner('Loading documents Please wait'):
            st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001") #uses GoogleGenerativeAIEmbeddings to create embedding. It selects the model mentioned to do that.
            st.session_state.loader=PyPDFDirectoryLoader("./docs") #sets up the mechanism to load the pdf from the directory mentioned. Does not yet load the file.
            st.session_state.docs=st.session_state.loader.load() #loads the pdf and stores it in the session_state.docs
            st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) #creates an instance or sets up a mechanism for splitting the document according to the information provided. chunk_size is the maximum size of each chunk of text. chunk_overlap means it overlaps the next 200 characters. 
            st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #here it actually splits the document. It splits the first 20 documents and stores it in the variable declared.
            st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #Uses FAISS to search for similary information by taking into account the vectors formed from embedding.
        st.success('Documents loaded successfully')




prompt1=st.text_input("Enter your questions related to the document(s) only") #our question is stored in prompt1.
#check if the embeddings are loaded and ready.
if vector_embedding():
    st.write("chat bot is ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt) #creates a chain based on the llm and prompt template.
    retriever=st.session_state.vectors.as_retriever() #used to search and retrieve relevant pieces of information from a large collection based on a query.
    retrieval_chain=create_retrieval_chain(retriever,document_chain) #creates a retrieval chain by taking the retrieved information then passing through a series of processes to format the response or any other processing required to convert the retrieved documents into a useful output. 
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1}) #executes the retrieval chain based on the user's query as input.
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")