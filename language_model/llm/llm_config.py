from langchain import HuggingFacePipeline
from langchain.memory import VectorStoreRetrieverMemory
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


class LlmConfig:

    def __init__(self, vector_db_config):
        self.model_name_or_path = "TheBloke/Llama-2-70B-Ensemble-v5-GPTQ"
        self.model_basename = "model"
        self.local_llm = None
        self.memory = None
        self.vector_db_config = vector_db_config
        self.config()

    def config(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path,
                                                     device_map="auto",
                                                     trust_remote_code=False,
                                                     revision="main")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
        )

        self.local_llm = HuggingFacePipeline(pipeline=pipe)

        retriever = self.vector_db_config.db.as_retriever(search_kwargs=dict(k=1))
        self.memory = VectorStoreRetrieverMemory(retriever=retriever)
