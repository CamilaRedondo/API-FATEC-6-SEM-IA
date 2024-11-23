import os
import re
import spacy
import nltk
import logging
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from util import dir_management


def load_stopwords():
    try:
        return set(stopwords.words('portuguese'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('portuguese'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_exclamations_and_periods(text):
    text = re.sub(r'[!.,@]', '', text)
    return text


def remove_stop_words(text, stop_words):
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


def main():
    logging.info('Obtendo dados...')
    csv_columns = ['product_name', 'site_category_lv1',
                   'site_category_lv2', 'overall_rating', 'review_text']
    df = pd.read_csv(os.path.join(
        dir_management.get_project_dir(), 'B2W-Reviews.csv'))
    df_reduced = df.drop(
        columns=[col for col in df.columns if col not in csv_columns])

    logging.info('Realizando processo de PLN...')
    for column in csv_columns:
        df_reduced[column] = df_reduced[column].apply(
            lambda x: clean_text(str(x)))
        df_reduced[column] = df_reduced[column].apply(
            lambda x: remove_exclamations_and_periods(str(x)))
        df_reduced[column] = df_reduced[column].apply(
            lambda x: remove_accents(str(x)))
        df_reduced[column] = df_reduced[column].apply(
            lambda x: remove_stop_words(str(x), stop_words))

    df_reduced["review_text"] = df_reduced["review_text"].apply(
        lambda x: remove_filling_words(str(x)))
    df_reduced["review_text"] = df_reduced["review_text"].apply(
        lambda x: remove_repetitive_words(str(x)))
    df_reduced["review_text"] = df_reduced["review_text"].apply(
        lambda x: remove_repetitive_letters(str(x)))

    result_file_name = f'B2W-Reviews-After-PLN.csv'
    df_reduced.sort_values('site_category_lv1').to_csv(
        os.path.join(dir_management.get_out_dir(), result_file_name), index=False)

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

    logging.info('Aplicando estrategia de chunk...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    logging.info('Aplicando estrategia de embedding...')
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    max_batch_size = 3000
    vector_db_path = os.path.join(
        dir_management.get_project_dir(), 'chroma_db')
    if not os.path.exists(vector_db_path):
        logging.info('Persistindo vetores em banco...')
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
        logging.info('Vetores ja persistidos...')
        vectorstore = Chroma(
            embedding_function=hf,
            collection_name='reviews',
            persist_directory=vector_db_path
        )


if __name__ == '__main__':
    log_file = os.path.join(
        dir_management.get_logs_dir('data_processing'), 'log.txt')
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log_handler = RotatingFileHandler(
        log_file, mode='a', maxBytes=500*1024*1024, backupCount=3, encoding=None, delay=0)
    log_handler.setFormatter(log_formatter)
    log_handler.setLevel(logging.INFO)
    app_log = logging.getLogger()
    app_log.setLevel(logging.INFO)
    app_log.addHandler(log_handler)

    logging.info('Iniciando processo...')
    load_dotenv()
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2,
        model_kwargs={
            "top_p": 0.85,
        }
    )
    stop_words = load_stopwords()
    nlp = spacy.load('pt_core_news_sm')
    main()
    logging.info('Processo finalizado.')
