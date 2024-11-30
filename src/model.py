import os
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

class LLM:
    def __init__(self):
        self.llm = LlamaAzureOpenAI(
            deployment_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        self.model_name = "gpt-4o-mini"

    def get_response(self, prompt):
        return self.llm.complete(prompt).text

    def get_model(self):
        return self.llm
    def get_model_name(self):
        return self.model_name


class Embedding:
    def __init__(self):
        self.embed_model = AzureOpenAIEmbedding(
            deployment_name="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01",
        )
        # self.embed_model = HuggingFaceEmbedding(
        #     model_name="BAAI/bge-small-en-v1.5"
        # )
    def get_text_embedding(self, text):
        return self.embed_model.get_text_embedding(text)

    def get_model(self):
        return self.embed_model

class Batch_Embedding:
    def __init__(self, batch_size=4096):
        self.embed_model = AzureOpenAIEmbedding(
            # deployment_name="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01"
        )
        self.batch_size = batch_size

    def get_text_embedding(self, texts):

        # Initialize list to store embeddings
        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embed_model.get_text_embedding(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_model(self):
        return self.embed_model
