# %%
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import os
from dotenv import load_dotenv

# %%
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatGroq(model="llama3-8b-8192")

# %%
df = pd.read_csv(
    r'C:\Projetos\b2w-reviews01\B2W-Reviews01.csv', low_memory=False)
columns_to_keep = ['product_name', 'review_title', 'review_text']
df_reduced = df[columns_to_keep]
df_reduced.to_csv(
    r'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\out\B2W-Reviews-top10.csv')

# %%
loader = CSVLoader(file_path=r'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\out\B2W-Reviews-top10.csv',
                   encoding='utf-8',
                   csv_args={
                       'delimiter': ',',
                       'quotechar': '"',
                       'fieldnames': ['Product_name', 'Review_title', 'Review_text']
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

max_batch_size = 3000


def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


vectorstore = Chroma(
    embedding_function=hf,
    collection_name='reviews',
    persist_directory=r'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\chroma_db'
)

for batch in batch_documents(splits, max_batch_size):
    vectorstore.add_texts(
        texts=[doc.page_content for doc in batch],
        metadatas=[doc.metadata for doc in batch]
    )

# %%
