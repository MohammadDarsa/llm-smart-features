from fastapi import FastAPI

from language_model.llm.llm_config import LlmConfig
from language_model.memory.vector_db_config import VectorDBConfig
from language_model.routers.chat_api import ChatController
app = FastAPI()

vector_db_config = VectorDBConfig(db_path='./documents', chunk_size=1000, chunk_overlap=20, batch_size=1)
llm_config = LlmConfig(vector_db_config=vector_db_config)

# Routes
chat_controller = ChatController(llm_config)
app.include_router(chat_controller.router)
