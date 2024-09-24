# %%
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain import hub
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
rows_number = 1000  # Define quantas rows do csv serão utilizadas no RAG
df = pd.read_csv(r'.\B2W-Reviews.csv')
columns_to_keep = ['product_name', 'review_title', 'review_text']
df_reduced = df.drop(
    columns=[col for col in df.columns if col not in columns_to_keep])
new_df = df_reduced.head(rows_number).to_csv(
    rf'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\out\B2W-Reviews-top{rows_number}.csv')

# %%
loader = CSVLoader(file_path=rf'C:\Projetos\Testes_IA\API-FATEC-6-SEM-IA\out\B2W-Reviews-top{rows_number}.csv',
                   encoding='utf-8',
                   csv_args={
                       'delimiter': ',',
                       'quotechar': '"',
                       'fieldnames': ['product_name', 'review_title', 'review_text']
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke(
    "Me conte mais sobre o copo acrilico, responda em português do Brasil")

# %%