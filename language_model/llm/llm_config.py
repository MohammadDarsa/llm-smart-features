from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LlmConfig:

    def __init__(self):
        self.model_id = 'PygmalionAI/pygmalion-2-13b'
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.local_llm = None
        self.config()

    def config(self):
        # go for a smaller model if you don't have the VRAM

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, use_cache=True)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=300
        )

        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)
