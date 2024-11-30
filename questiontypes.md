# Question Types

* **Simple:** Basic questions that do not require complex reasoning or multiple contexts. (note: Giskard explicitly defines these as being generated from a straightforward piece of the knowledge base)
* **Reasoning:** Questions designed to enhance the need for reasoning to answer them effectively (at least one leap of intuition required to correlate the answer to the correct information from the knowledge base)
  * **Multi-Context:** Questions that necessitate information from multiple related sections or chunks to formulate an answer.
* **Situational:** Questions including user context to evaluate the ability of the generation to produce relevant answer according to the context (first part of the question establishes some [correct OR distracting] context prior to the question itself)
  * **Distracting:** Questions made to confuse the retrieval part of the RAG with a distracting element from the knowledge base but irrelevant to the question. (Designed to mess with embedding engines - leaves more reasoning work for the LLM)
* **Double Questions:** Questions with two distinct parts to evaluate the capabilities of the query rewriter of the RAG
* [IGNORE FOR NOW]: **Conversational:** Questions that simulate a chat-based question-and-follow-up interaction, mimicking a chat-Q&A pipeline.
