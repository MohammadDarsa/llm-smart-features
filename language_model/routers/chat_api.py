from fastapi import APIRouter
from langchain import PromptTemplate, LLMChain
import json

from language_model.model.request.chat_request import ChatRequest


def get_products():
    return {
        "products": [
            {
                "id": "1",
                "name": "Iphone 15 Pro Max",
                "brand": "Apple",
                "price": 1199,
                "specs": {
                    "screen": "6.7 inch",
                    "camera": "48MP",
                    "battery": "4422mAh",
                    "ram": "8GB",
                    "storage": "128GB",
                    "cpu": "A17 Pro",
                    "os": "iOS 17"
                }
            },
            {
                "id": "2",
                "name": "Samsung Galaxy S23 Ultra",
                "brand": "Samsung",
                "price": 899,
                "specs": {
                    "screen": "6.8 inch",
                    "camera": "200MP",
                    "battery": "5000mAh",
                    "ram": "12GB",
                    "storage": "256GB",
                    "cpu": "Snapdragon 8 Gen 2",
                    "os": "Android 13"
                }
            },
            {
                "id": "3",
                "name": "Google Pixel 7 Pro",
                "brand": "Google",
                "price": 699,
                "specs": {
                    "screen": "6.7 inch",
                    "camera": "50MP",
                    "battery": "5000mAh",
                    "ram": "8GB",
                    "storage": "128GB",
                    "cpu": "Google Tensor G2",
                    "os": "Android 13"
                }
            },
            {
                "id": "4",
                "name": "OnePlus 11",
                "brand": "OnePlus",
                "price": 899,
                "specs": {
                    "screen": "6.7 inch",
                    "camera": "50MP",
                    "battery": "5000mAh",
                    "ram": "8GB",
                    "storage": "128GB",
                    "cpu": "Snapdragon 8 Gen 2",
                    "os": "Android 12"
                }
            },
            {
                "id": "5",
                "name": "Iphone 15",
                "brand": "Apple",
                "price": 799,
                "specs": {
                    "screen": "6.1 inch",
                    "camera": "48MP",
                    "battery": "3349mAh",
                    "ram": "6GB",
                    "storage": "128GB",
                    "cpu": "A16 bionic",
                    "os": "iOS 17"
                }
            },
            {
                "id": "6",
                "name": "Samsung Galaxy S23",
                "brand": "Samsung",
                "price": 699,
                "specs": {
                    "screen": "6.1 inch",
                    "camera": "50MP",
                    "battery": "3900mAh",
                    "ram": "8GB",
                    "storage": "128GB",
                    "cpu": "Snapdragon 8 Gen 2",
                    "os": "Android 13"
                }
            }
        ]
    }


class ChatController:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.router = APIRouter()
        self.router.add_api_route("/chat", self.smart_search, methods=["POST"])

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

    async def smart_search(self, request: ChatRequest):
        # get the products
        products = get_products()

        # get products as a string using json
        products_str = json.dumps(products)

        # create a template
        _DEFAULT_TEMPLATE = """
            Here's a list of products in json format. Only return the product ids that match the query the most.
            Output example: id: 1, 3, 5;
            Multiple products can be returned.
            
            Here's the list of products in json format:

            {products_str}

            Query: {query}
        """
        prompt = PromptTemplate(template=_DEFAULT_TEMPLATE, input_variables=["products_str", "query"])
        conversation = LLMChain(
            llm=self.llm_config.local_llm,
            prompt=prompt,
            verbose=True
        )

        return {"message": conversation({"query": request.text, "products_str": products_str})}

