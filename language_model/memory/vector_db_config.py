from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


class VectorDBConfig:
    def __init__(self, db_path='./documents', chunk_size=1000, chunk_overlap=20, batch_size=1):
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.db = self.config()
        self.db.persist()

    def config(self):
        docs = self.load_docs()
        split_docs = self.split_docs(docs)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"},
                                                   encode_kwargs={"batch_size": self.batch_size})
        return Chroma.from_documents(split_docs, embeddings, persist_directory="./db")

    def load_docs(self):
        loader = DirectoryLoader(self.db_path, loader_cls=TextLoader)
        loaded_docs = loader.load()
        return loaded_docs

    def split_docs(self, doc):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_doc = text_splitter.split_documents(doc)
        return split_doc
