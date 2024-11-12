from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationTokenBufferMemory

MAX_TOKEN_BUFFER_LIMIT: int = 1000
PROMPT_TEMPLATE: str = """
    Responda sempre de forma clara e precisa em português do Brasil.
    Você é um assistente especializado em marketing e feedback de clientes. Responda perguntas que estejam relacionadas a marketing, campanhas, review de clientes, avaliações de produtos ou publicidade e também responda dúvidas de cliente sobre as suas respostas anteriores. 
    Para perguntas fora desse escopo, responda: "Essa pergunta está fora do escopo deste chatbot. Por favor, faça perguntas relacionadas a marketing."
    Para perguntas sobre produtos, use apenas o feedback dos clientes fornecido no contexto para responder, não invente respostas. Se o contexto não tiver informações suficientes, responda: "Não há informações suficientes para responder a essa pergunta."

    Memória: {memory}
    Contexto: {context}
    Pergunta: {question}
"""

memory: ConversationTokenBufferMemory = False

def configure(llm: ChatGroq) -> None:
    global memory

    if memory == False:
        memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=MAX_TOKEN_BUFFER_LIMIT)

def ask(question: str, retriever: VectorStoreRetriever, llm: ChatGroq) -> str:
    global memory

    configure(llm)

    doc_list: list[Document] = retriever.invoke(question)
    formated_docs: str = '\n\n'.join([doc.page_content for doc in doc_list])

    prompt: str = PROMPT_TEMPLATE.format(context=formated_docs, question=question, memory=memory)
    response: BaseMessage = llm.invoke(prompt)

    memory.save_context({"input": question}, {"output": response.content})
    return response.content