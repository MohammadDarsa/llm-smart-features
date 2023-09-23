from fastapi import APIRouter
from langchain import PromptTemplate, LLMChain

from language_model.model.request.chat_request import ChatRequest


class ChatController:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.router = APIRouter()
        self.router.add_api_route("/chat", self.chat, methods=["POST"])

    async def chat(self, request: ChatRequest):
        _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Relevant pieces of previous conversation:
        {history}

        (You do not need to use these pieces of information if not relevant)

        Current conversation:
        Human: {question}"""
        prompt = PromptTemplate.from_template(template=_DEFAULT_TEMPLATE)
        conversation = LLMChain(
            llm=self.llm_config.local_llm,
            prompt=prompt,
            verbose=True,
            memory=self.llm_config.memory
        )

        return {"message": conversation({"question": request.text})}
