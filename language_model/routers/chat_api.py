from fastapi import APIRouter
from langchain import PromptTemplate, LLMChain

from language_model.model.request.chat_request import ChatRequest


class ChatController:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.router = APIRouter()
        self.router.add_api_route("/chat", self.chat, methods=["POST"])

    async def chat(self, request: ChatRequest):
        # Memory config
        self.llm_config.memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})

        # Template config
        _DEFAULT_TEMPLATE = """The following is a chat between an AI-powered online camgirl named "Lily" and her viewers, Lily is designed to provide entertainment and companionship to users who visit her stream. Her primary goal is to engage with users and make them feel comfortable and relaxed. Lily has been trained on a wide range of topics and is able to respond to various questions and requests. Users may interact with Lily by typing in the chat which is supplied below. Lily should respond in a friendly and conversational manner, using sexy enticing language to comfort & attract viewers and hold their attention.

Previous conversations with viewers:
{history}

(You do not need to use these pieces of information if not relevant)

New chat messages:
{question}

Lily:"""

        prompt = PromptTemplate.from_template(template=_DEFAULT_TEMPLATE)
        conversation = LLMChain(
            llm=self.llm_config.local_llm,
            prompt=prompt,
            verbose=True,
            memory=self.llm_config.memory
        )

        return {"message": conversation({"question": request.text})}
