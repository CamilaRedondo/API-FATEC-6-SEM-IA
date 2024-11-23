# API-FATEC-6-SEM-IA

## Caso necessário instalar as bibliotecas manualmente, executar os seguintes comandos:
- pip install --upgrade langchain langchain-community langchain-chroma
- pip install sentence-transformers
- pip install langchain-huggingface
- pip install bs4
- pip install -qU langchain-groq
- pip install python-dotenv
- pip install pandas
- pip install datasets spacy textblob nltk
- python -m spacy download pt_core_news_sm

## Arquivo data_processing_startup.py:
Realiza o tratamento de dados e o processamento de linguagem natural para persistência em banco vetorial, o qual é resultado do processo na raiz do projeto como ./chroma_db.

## Arquivo server_startup.py:
Levanta a instância que permite a comunicação via websocket com o charbot.
