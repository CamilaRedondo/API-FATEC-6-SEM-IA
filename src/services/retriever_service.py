import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from util import dir_management
import classes.chat_handler as chat_handler


load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.2,
    model_kwargs={
        "top_p": 0.85,
    }  # Respostas diretas e focadas, mas menos criativas
)

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

max_batch_size = 3000
vector_db_path = os.path.join(dir_management.get_project_dir(), 'chroma_db')

vectorstore = Chroma(
    embedding_function=hf,
    collection_name='reviews',
    persist_directory=vector_db_path
)

retriever = vectorstore.as_retriever()

prompt_template = """

Responda em português do Brasil.Responda com clareza e objetividade.
Você é um assistente especializado em marketing e feedback de clientes. Responda apenas a perguntas relacionadas a marketing, campanhas, avaliações de clientes e produtos.
Para perguntas sobre produtos, use apenas o feedback dos clientes fornecido no contexto para responder, não invente respostas.
Para perguntas do tipo saudações, responda amigavelmente.  
Se a pergunta estiver fora do escopo, responda: "Essa pergunta está fora do escopo deste chatbot. Por favor, faça perguntas relacionadas a marketing ou reformule a pergunta."


Contexto: {context}
Pergunta: {question}

Resposta (use dados específicos para apoiar sua resposta):

"""


def custom_prompt(context, question):
    return prompt_template.format(context=context, question=question)


def preprocess_question(question):
    synonyms = {
        'celular': 'smartphone',
        'celulares': 'smartphones',
        'telefone': 'smartphone',
        'telefones': 'smartphones',
    }
    for word, synonym in synonyms.items():
        question = question.replace(word, synonym)
    return question.lower()


def run_rag_chain(question):
    question = preprocess_question(question)
    return chat_handler.ask(question=question, retriever=retriever, llm=llm)
