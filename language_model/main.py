from fastapi import FastAPI
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from language_model.routers.chat_api import ChatController

app = FastAPI()

directory = './documents'


def load_docs(path):
    loader = DirectoryLoader(path, loader_cls=TextLoader)
    loaded_docs = loader.load()
    return loaded_docs


print("loading docs")
documents = load_docs(directory)
print("loading docs done")


def split_docs(doc, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_doc = text_splitter.split_documents(doc)
    return split_doc


print("splitting docs")
docs = split_docs(documents)
print("splitting docs done")

print("embeddings")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2",  model_kwargs={"device": "cuda"}, encode_kwargs={"batch_size": 1})
print("embeddings done")

print("init chroma db")
db = Chroma.from_documents(docs, embeddings, persist_directory="./db")
print("chroma db initialized")
print("persisting db")
db.persist()
print("persisting db done")

# model_id = 'PygmalionAI/pygmalion-2-13b'  # go for a smaller model if you dont have the VRAM
model_name_or_path = "TheBloke/Manticore-13B-Chat-Pyg-SuperHOT-8K-GPTQ"
model_basename = "manticore-13b-chat-pyg-superhot-8k-GPTQ-4bit-128g.no-act.order"

use_triton = False

# tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)

# model = AutoGPTQForCausalLM.from_pretrained(model_name_or_path,
#         model_basename=model_basename,
#         use_safetensors=True,
#         trust_remote_code=True,
#         device_map='auto',
#         use_triton=use_triton)

# model.seqlen = 8192

pipe = pipeline(
    "text-generation",
    model=model_name_or_path,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.7,
#     top_p=0.95,
#     repetition_penalty=1.15,
#     device=0
# )

local_llm = HuggingFacePipeline(pipeline=pipe)

# Memory config
retriever = db.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})

# Template config
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {question}"""
prompt = PromptTemplate.from_template(template=_DEFAULT_TEMPLATE)
conversation = LLMChain(
    llm=local_llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Routes
chat_controller = ChatController(conversation)
app.include_router(chat_controller.router)
