import json

from fastapi import APIRouter
from langchain import PromptTemplate, LLMChain

from language_model.model.request.chat_request import ChatRequest


def get_headers():
    return "id  name  brand  price  screen  camera  battery  ram  storage  cpu  os"


def get_products():
    return """1  Iphone 15 Pro Max  Apple  1199  6.7 inch  48MP  4422mAh  8GB  128GB  A17 Pro  iOS 17
2  Samsung Galaxy S23 Ultra  Samsung  899  6.8 inch  200MP  5000mAh  12GB  256GB  Snapdragon 8 Gen 2  Android 13
3  Google Pixel 7 Pro  Google  699  6.7 inch  50MP  5000mAh  8GB  128GB  Google Tensor G2  Android 13
4  OnePlus 11  OnePlus  899  6.7 inch  50MP  5000mAh  8GB  128GB  Snapdragon 8 Gen 2  Android 12
5  Iphone 15  Apple  799  6.1 inch  48MP  3349mAh  6GB  128GB  A16 bionic  iOS 17
6  Samsung Galaxy S23  Samsung  699  6.1 inch  50MP  3900mAh  8GB  128GB  Snapdragon 8 Gen 2  Android 13"""


class ChatController:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.router = APIRouter()
        self.router.add_api_route("/chat", self.smart_search, methods=["POST"])

    async def smart_search(self, request: ChatRequest):
        # get the products
        headers = get_headers()
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
answer: "3"

query: "Phones with more than 20MP camera:
answer: "1,2,3"

Answer this query of the user:

query: {query}
answer:"""
        prompt = PromptTemplate(template=_DEFAULT_TEMPLATE, input_variables=["products_str", "query"])
        conversation = LLMChain(
            llm=self.llm_config.local_llm,
            prompt=prompt,
            verbose=True
        )

        return {"message": conversation({"query": request.text, "products_str": products_str, "headers": headers})}
