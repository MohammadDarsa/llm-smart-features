from fastapi import APIRouter
from language_model.model.request.chat_request import ChatRequest


class ChatController:
    def __init__(self, conversation):
        self.conversation = conversation
        self.router = APIRouter()
        self.router.add_api_route("/chat", self.chat, methods=["POST"])

    async def chat(self, request: ChatRequest):
        return {"message": self.conversation({"question": request.text})}
