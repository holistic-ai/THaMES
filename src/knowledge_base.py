import random
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
# from model import LLM,Embedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)

import numpy as np

def load_documents(path):
    documents = SimpleDirectoryReader(
        input_dir=path
    ).load_data(show_progress=True)
    if len(documents) == 0:
        raise ValueError("No documents found in the specified directory")
    if len(documents) > 50:
        print("Randomly sampling 50 documents...")
        # random sample 50 documents
        documents = random.sample(documents, 50)
    return documents

class KnowledgeBase:
    def __init__(self, llm, embedding, documents, splits_type="semantic"):
        print("Initializing Knowledge Base...")
        self.llm = llm
        self.embedding = embedding

        print("Loading models...")
        Settings.llm = self.llm.get_model()
        Settings.embed_model = self.embedding.get_model()

        print("Loading documents...")
        self.documents = documents

        print("Splitting documents...")
        if splits_type == "semantic":
            print("Using SemanticSplitterNodeParser...")
            self.splitter = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
            )
        else:
            print("Using SentenceSplitter...")
            self.splitter = SentenceSplitter(chunk_size=2048)

        self.nodes = self.splitter.get_nodes_from_documents(documents, show_progress=True)
        print(f"Number of nodes: {len(self.nodes)}")
        self.index = VectorStoreIndex(self.nodes, show_progress=True)

        self.node_retrieval_counts = {node.id_: 0 for node in self.nodes}

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=4,
            alpha=None,
            doc_ids=None,
            
        )


    def get_random_node(self):
        node = random.choice(self.nodes)
        # self.node_retrieval_counts[node.id_] += 1
        return node

    def get_random_node_weighted(self):
        # Inverse retrieval counts to calculate probabilities
        counts = np.array(list(self.node_retrieval_counts.values()))
        probabilities = 1 / (counts + 1)  # Add 1 to avoid division by zero
        probabilities /= probabilities.sum()  # Normalize to get a valid probability distribution

        # Select a node index based on the weighted probabilities
        node_index = np.random.choice(len(self.nodes), p=probabilities)
        node = self.nodes[node_index]
        # self.node_retrieval_counts[node.id_] += 1
        return node

    def get_index(self):
        return self.index

    def get_embedding_dict(self):
        return self.index.vector_store.data.embedding_dict

    def get_neighbors(self, node, n_neighbors=10, similarity_threshold=0.7):

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=n_neighbors,
            alpha=None,
            doc_ids=None,
        )

        neighbors = retriever.retrieve(node.text)
        filtered_neighbors = []
        for x in neighbors:
            if x.score >= similarity_threshold:
                filtered_neighbors.append(x)
                self.node_retrieval_counts[x.id_] += 1

        return filtered_neighbors
