# %%
import os
import re
import spacy
import nltk
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

# %%
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatGroq(model="llama3-8b-8192")
nltk.download('stopwords')
nlp = spacy.load('pt_core_news_sm')
stop_words = set(stopwords.words('portuguese'))
csv_columns = ['product_name', 'site_category_lv2',
                    'overall_rating', 'recommend_to_a_friend', 'review_text']

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_exclamations_and_periods(text):
    text = re.sub(r'[!.,]', '', text)
    return text


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])


def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    return text


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %%
rows_number = 500  # Define quantas rows do csv serão utilizadas no RAG
df = pd.read_csv(r'.\B2W-Reviews.csv')
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
new_df = df_reduced.head(rows_number).to_csv(
    rf'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\out\B2W-Reviews-top{rows_number}.csv')

# %%
loader = CSVLoader(file_path=rf'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\out\B2W-Reviews-top{rows_number}.csv',
                   encoding='utf-8',
                   csv_args={
                       'delimiter': ',',
                       'quotechar': '"',
                       'fieldnames': csv_columns
                   })

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=hf)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever |
        format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke(
    "Pode me indicar bons smartphones? Responda em português do Brasil")

# %%
