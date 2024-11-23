import os
import asyncio
import websockets
import logging
from util import dir_management
from services import retriever_service
from logging.handlers import RotatingFileHandler


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
    server = await websockets.serve(handler, "localhost", 8081)
    print("Servidor WebSocket rodando...")

    await server.wait_closed()

if __name__ == '__main__':
    log_file = os.path.join(
        dir_management.get_logs_dir('server'), 'log.txt')
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log_handler = RotatingFileHandler(
        log_file, mode='a', maxBytes=500*1024*1024, backupCount=3, encoding=None, delay=0)
    log_handler.setFormatter(log_formatter)
    log_handler.setLevel(logging.INFO)
    app_log = logging.getLogger()
    app_log.setLevel(logging.INFO)
    app_log.addHandler(log_handler)

    logging.info('Start server')
    asyncio.run(main())
