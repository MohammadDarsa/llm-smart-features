import json

from fastapi import APIRouter
from langchain import PromptTemplate, LLMChain
from langchain.memory import VectorStoreRetrieverMemory

from language_model.model.request.chat_request import ChatRequest


def get_products():
    return """id:1  name:Iphone 15 Pro Max  brand:Apple  price1199  ;  id:2  name:Samsung Galaxy S23 Ultra  brand:Samsung  price:899  ;  id:3  name:Google Pixel 7 Pro  brand:Google  price:699  ;  id:4  name:OnePlus 11  brand:OnePlus  price:899  ;  id:5  name:Iphone 15  brand:Apple  price:799  ;  id:6  name:Samsung Galaxy S23  brand:Samsung  price:699  ;"""


class ChatController:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.router = APIRouter()
        self.router.add_api_route("/smart-search", self.smart_search, methods=["POST"])
        self.router.add_api_route("/chat", self.chat, methods=["POST"])

    async def smart_search(self, request: ChatRequest):
        # get the products
        products = get_products()

        # get products as a string using json
        products_str = json.dumps(products)

        # create a template
        _DEFAULT_TEMPLATE = """You're a shop keeper in an electronics store. A customer comes in and asks for a phone with the following specs presented in the query.
Here's the product list containing all the attributes as key value pair separated by a column (:). The user will search on these key value pairs, The products are separated by a semi-column(;):
{products_str}

The user will input a query and you should find the most suitable phone or phones.

The answer to the customer's query should only include the ids of the phones that matches the query and nothing else. The answer should only contain ths ids.

Examples of query-question:

query: "google branded phones"
answer: 3

query: "Phones with more than 20MP camera:
answer: 1,2,3

Answer this query of the user:

query: {query}
answer:"""
        prompt = PromptTemplate(template=_DEFAULT_TEMPLATE, input_variables=["products_str", "query"])
        conversation = LLMChain(
            llm=self.llm_config.local_llm,
            prompt=prompt,
            verbose=True
        )

        return {"message": conversation({"query": request.text, "products_str": products_str})}

    async def chat(self, request: ChatRequest):
        # Memory config
        self.llm_config.memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})

        # Template config
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
