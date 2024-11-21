import asyncio
import websockets
from services import retriever_service

async def handler(websocket):
    while True:
        try:
            question = await websocket.recv()
            response = retriever_service.run_rag_chain(question)
            if websocket.open:
                await websocket.send(response)
        except websockets.exceptions.ConnectionClosed:
            print("A conexão foi fechada pelo cliente.")
            break
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}")
            if websocket.open:
                await websocket.send("Desculpe, ocorreu um erro ao processar sua pergunta.")


async def main():
    # Inicia o servidor WebSocket
    server = await websockets.serve(handler, "localhost", 8081)
    print("Servidor WebSocket rodando...")  

    # Mantém o servidor rodando 
    await server.wait_closed()

# Executa a função main
asyncio.run(main())

