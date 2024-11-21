import os
import re
import spacy
import nltk
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        }#Respostas diretas e focadas, mas menos criativas
)

def load_stopwords():
    try:
        return set(stopwords.words('portuguese'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('portuguese'))


stop_words = load_stopwords()

nlp = spacy.load('pt_core_news_sm')

csv_columns = ['product_name', 'site_category_lv1',
               'site_category_lv2', 'overall_rating', 'review_text']


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_exclamations_and_periods(text):
    text = re.sub(r'[!.,@]', '', text)
    return text


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])


def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def remove_filling_words(text):
    word_list = [
        'de', 'a', 'o', 'do', 'da', 'em', 'para', 'com', 'na', 'por',
        'uma', 'os', 'no', 'se', 'mas', 'as', 'dos', 'pois', 'né'
    ]
    return ' '.join([word for word in text.split() if word not in word_list])


def remove_repetitive_words(text):
    return re.sub(r'\b(\w+)( \1)+\b', '', text)


def remove_repetitive_letters(text):
    return re.sub(r'/(.)\1{3,}/g', '', text)


def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


df = pd.read_csv(os.path.join(
    dir_management.get_project_dir(), 'B2W-Reviews.csv'))
df_reduced = df.drop(
    columns=[col for col in df.columns if col not in csv_columns])

for column in csv_columns:
    df_reduced[column] = df_reduced[column].apply(lambda x: clean_text(str(x)))
    df_reduced[column] = df_reduced[column].apply(
        lambda x: remove_exclamations_and_periods(str(x)))
    df_reduced[column] = df_reduced[column].apply(
        lambda x: remove_accents(str(x)))
    df_reduced[column] = df_reduced[column].apply(
        lambda x: remove_stop_words(str(x)))

df_reduced["review_text"] = df_reduced["review_text"].apply(
    lambda x: remove_filling_words(str(x)))
df_reduced["review_text"] = df_reduced["review_text"].apply(
    lambda x: remove_repetitive_words(str(x)))
df_reduced["review_text"] = df_reduced["review_text"].apply(
    lambda x: remove_repetitive_letters(str(x)))

result_file_name = f'B2W-Reviews-After-PLN.csv'
df_reduced.sort_values('site_category_lv1').to_csv(
    os.path.join(dir_management.get_out_dir(), result_file_name), index= False)

loader = CSVLoader(
    file_path=os.path.join(dir_management.get_out_dir(), result_file_name),
    encoding='utf-8',
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': csv_columns
    }
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

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
if  not os.path.exists(vector_db_path):
    vectorstore = Chroma(
        embedding_function=hf,
        collection_name='reviews',
        persist_directory=vector_db_path
    )

    for batch in batch_documents(splits, max_batch_size):
        vectorstore.add_texts(
            texts=[doc.page_content for doc in batch],
            metadatas=[doc.metadata for doc in batch]
        )

else:
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

    # Recupera documentos e formata o contexto
    # retrieved_docs = retriever.invoke(question)  # aumentar para k=10 para ver o resultado 
    # formatted_context = format_docs(retrieved_docs)

    # # Cria o prompt customizado
    # full_prompt = custom_prompt(formatted_context, question)

    # # Passa o prompt para o modelo de linguagem
    # response = llm.invoke(full_prompt)

    # # Parseia a resposta para o formato correto
    # parsed_response = StrOutputParser().parse(response)

    # return parsed_response.content

    return chat_handler.ask(question=question, retriever=retriever, llm=llm)