# %%
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

# %%
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model="llama3-8b-8192")

# %%


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

# %%


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


# %%
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
df_reduced.sort_values('site_category_lv1').to_csv(os.path.join(dir_management.get_out_dir(), result_file_name), index=False)

# %%
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

# %%
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

# %%
prompt_template = """

Responda sempre de forma clara e precisa em português do Brasil.
Você é um assistente especializado em marketing e feedback de clientes. Responda perguntas que estejam relacionadas a marketing, campanhas, review de clientes, avaliações de produtos ou publicidade. 
Para perguntas fora desse escopo, responda: "Essa pergunta está fora do escopo deste chatbot. Por favor, faça perguntas relacionadas a marketing."
Para perguntas sobre produtos, use apenas o feedback dos clientes fornecido no contexto para responder, não invente respostas. Se o contexto não tiver informações suficientes, responda: "Não há informações suficientes para responder a essa pergunta."

Contexto: {context}
Pergunta: {question}

Resposta:

"""

# %%


def custom_prompt(context, question):
    return prompt_template.format(context=context, question=question)

# RAG Chain com prompt customizado

def run_rag_chain(question):
    # # Recupera documentos e formata o contexto
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


# # %%
# response = run_rag_chain('Me fale sobre o clima de amanhã.')
# print(response)

# # %%
# response = run_rag_chain('Me fale recomende produtos de informatica')
# print(response)

# # %%
# response = run_rag_chain(
#     "Como podemos melhorar a experiência do cliente com base nas avaliações recebidas?")
# print(response)

# %% [markdown]
# perguntas para marketing:
#
# Quais são os principais pontos de satisfação mencionados pelos clientes em suas avaliações?
# Quais são os principais motivos de insatisfação dos clientes sobre os produtos?
# Existe alguma tendência nos feedbacks que indica um aumento ou diminuição da satisfação do cliente?
#
# Quais características dos produtos são mais frequentemente elogiadas pelos clientes?
# Como os produto se comparam a concorrentes nas avaliações dos clientes?
# Quais produtos têm recebido as melhores e as piores avaliações, e por quê?
#
# Com base nos feedbacks, que ações de marketing podemos implementar para melhorar a percepção da marca?
# Quais campanhas anteriores parecem ter gerado mais impacto positivo, de acordo com as avaliações?
# Como podemos usar os feedbacks dos clientes para criar uma nova campanha de marketing?
#
# Existem padrões nas avaliações que indicam diferentes preferências entre diferentes grupos de clientes (por exemplo, idade, gênero, localização)?
# Que tipo de clientes estão mais satisfeitos com nosso produto e por quê?
#
# Quais tendências recentes podem ser observadas nos feedbacks que poderiam impactar nossas estratégias de marketing futuras?
# O que os clientes estão buscando atualmente que pode não estar sendo atendido pelos nossos produtos?
#
# Como podemos melhorar a experiência do cliente com base nas avaliações recebidas?
# Quais atributos dos produtos mais impactam a decisão de compra, de acordo com as opiniões dos clientes?
# Os clientes recomendariam o nosso produto a um amigo? Qual é a porcentagem de respostas positivas?
#
# Qual o produto mais bem avaliado pelos clientes ?
# Recomendações para melhorar a satisfação do cliente?
# Qual é o feedback positivos dos clientes sobre os produtos?

# %% [markdown]
# s campanhas anteriores parecem ter gerado mais impacto positivo, de acordo com as avaliações?
# Como podemos usar os feedbacks dos clientes para criar uma nova campanha de marketing?
#
# Existem padrões nas avaliações que indicam diferentes preferências entre diferentes grupos de clientes (por exemplo, idade, gênero, localização)?
# Que tipo de clientes estão mais satisfeitos com nosso produto e por quê?
#
# Quais tendências recentes podem ser observadas nos feedbacks que poderiam impactar nossas estratégias de marketing futuras?
# O que os clientes estão buscando atualmente que pode não estar sendo atendido pelos nossos produtos?
#
# Como podemos melhorar a experiência do cliente com base nas avaliações recebidas?
# Quais atributos dos produtos mais impactam a decisão de compra, de acordo com as opiniões dos clientes?
# Os clientes recomendariam o nosso produto a um amigo? Qual é a porcentagem de respostas positivas?
#
# Qual o produto mais bem avaliado pelos clientes ?
# Recomendações para melhorar a satisfação do cliente?
# Qual é o feedback positivos dos clientes sobre os produtos?
