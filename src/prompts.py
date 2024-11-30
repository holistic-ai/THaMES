################################################################## AGENT PROMPTS ##################################################################
QUESTION_GENERATION_SYSTEM_PROMPT = """Your role is to generate evaluation questions based a knowledge base, a subsection of which is provided to you as a list of context paragraphs. We are building a testset for a model. Assume the model that is being tested cannot see the context when answering the question.


Your question must be related to a provided context.  
Please respect the following rules to generate the question:
- The answer to the question should be found inside the provided context
- The question must be self-contained - in other words, the question must explicitly state any necessary titles, names, or terms. Assume that the model to be tested with this dataset has no prior knowledge of the context. Avoid using pronouns or references that are not explicitly mentioned in the context. Additionally, there must be a singular and clear answer to the question. Ensure that the potential answer to the question has not changed since the context was provided.
- The question and answer must be in this language: {language}

The user will provide the context, consisting of multiple paragraphs delimited by dashes "------".
You will return a question based exclusively on the provided context. 
You will output a list of {num_questions} JSON objects with key 'question' mapping to the generated question, without any other wrapping text or markdown. Ensure that you will only return valid JSON for example:
[{{"question": "xx", "question_type": "simple"}}, {{"question": "xx", "question_type": "simple"}}, ...].\n"""

ANSWER_GENERATION_SYSTEM_PROMPT = """Your role is to generate answers for a test set of evaluation questions generated from a knowledge base. You will be given a list of questions and a subsection of the knowledge base in the form of relevant context paragraphs to each question. The context paragraphs should include the information necessary to answer each question. Do not make up information that is not present in the context.

The type of question you are generating answers for is:
{agent_description}

Your answer must be found in the provided context.  
Please respect the following rules to generate the question:
- The answer to the question should be found inside the provided context
- The question should be self-contained, hence the answer must be found in the context provided. Do not make up information that is not present in the context.
- The answer must be in this language: {language}

the "question_type" property will ALWAYS be one of the following: ['simple', 'reasoning', 'multi_context', 'situational', 'distracting', 'double', 'conditionals']. do not change this property during the answer generation, simply copy it from the input.

The user will provide the questions and the context, consisting of multiple paragraphs delimited by dashes "------".
You will append the precise answer to each question based exclusively on the provided context. Ensure that your answers are complete sentences that directly answer the question. Do not provide additional information that is not directly related to your answer to the question.
\n\n
You will be given an input of questions that have the following format: 
[{{"question": "xx1", "question_type":"<question type of xx1>"}}, {{"question": "xx2", "question_type":"<question type of xx2>"}}, ...]
as well as some context paragraphs.
You will output a list of {num_questions} JSON objects with keys 'question' mapping to the original question and 'answer' mapping to the answer you generate for the respective question, without any other wrapping text or markdown. Ensure that you will only return valid JSON, for example:
[{{"question": "xx1", "answer": "<answer to xx1>", "question_type":"<question type of xx1>"}}, {{"question": "xx2", "answer": "<answer to xx2>", "question_type":"<question type of xx2>"}}, ...].\n
"""


QUESTION_EVOLUTION_PROMPT_MAP = {
    "reasoning": """Your role is to modify evaluation questions generated from a knowledge base, a subsection of which is provided to you as a list of context paragraphs. You will be given simple questions that need to be modified to include complex reasoning and logical connections between different pieces of data within the provided context. Evolve the provided simple questions into questions into questions designed to enhance the need for reasoning to answer them effectively (at least one leap of intuition required to correlate the answer to the correct information from the knowledge base)

        You will be given an input of questions that looks like the following: 
        [{{"question": "xx1", "question_type":"simple"}}, {{"question": "xx2", "question_type":"simple"}}, ...]
        as well as some context paragraphs.
        The user will provide the context, consisting of multiple paragraphs delimited by dashes "------".
        You will output a list of JSON objects with the same length as the input, with key 'question' mapping to the modified question, without any other wrapping text or markdown. Additionally, modify the question_type JSON parameter for all of these questions to be 'reasoning'. Ensure that you will only return valid JSON for example:
        [{{"question": "modified xx1 to include complex reasoning", "question_type": "reasoning"}}, {{"question": "modified xx2 to include complex reasoning", "question_type": "reasoning"}}, ...].\n""",
        
    "multi_context":"""Your role is to modify evaluation questions generated from a knowledge base, a subsection of which is provided to you as a list of context paragraphs. You will be given simple questions that need to be modified to have non-deterministic layers, potentially created from additionally provided context. The questions should require the integration of information from multiple sources or sections within the knowledge base, and the modifications should lead the model to use comprehension to get to the correct solution, rather than straight information retrieval. Evolve the provided simple questions into questions that necessitate information from multiple related sections or chunks of knowledge to formulate a single answer.
        You will be given an input of questions that looks like the following: 
        [{{"question": "xx1", "question_type":"simple"}}, {{"question": "xx2", "question_type":"simple"}}, ...]
        as well as some context paragraphs. 
        The user will provide the context, consisting of multiple paragraphs delimited by dashes "------". The additional context will be provided subsequently, in the same format.
        You will output a list of JSON objects with the same length as the input, with key 'question' mapping to the modified question, without any other wrapping text or markdown. Additionally, modify the question_type JSON parameter for all of these questions to be 'multi_context'. Ensure that you will only return valid JSON for example:
        [{{"question": "modified xx1 to include multiple contexts", "question_type":"multi_context"}}, {{"question": "modified xx2 to include multiple contexts",  "question_type":"multi_context"}}, ...].\n""",
    
    "situational": """
    Your role is to modify evaluation questions generated from a knowledge base, a subsection of which is provided to you as a list of context paragraphs. You will be given simple questions that need to be modified to have situational context about the subject matter inside the question. 
Please respect the following rules to generate the question:
- The question must include the information from the situational context.
- The question must sound plausible and coming from a real human user.
- The question can start with any form of greetings or not, choose randomly
- The original question and answer should be preserved.
- The question must be self-contained and understandable by humans. 
- The question must be in this language: {language}.        You will be given an input of questions that looks like the following: 
        [{{"question": "xx1", "question_type":"simple"}}, {{"question": "xx2", "question_type":"simple"}}, ...]
        as well as some context paragraphs. 
        The user will provide the context, consisting of multiple paragraphs delimited by dashes "------". The additional context from which to generate the situational statements will be provided subsequently, in the same format.
        You will output a list of JSON objects with the same length as the input, with key 'question' mapping to the modified question, without any other wrapping text or markdown. Additionally, modify the question_type JSON parameter for all of these questions to be 'situational'. Ensure that you will only return valid JSON for example:
        [{{"question": "modified xx1 to include additional statements", "question_type": "situational"}}, {{"question": "modified xx2 to include additional statements", "question_type":"situational"}}, ...].\n"""
,
    "distracting":"""Your role is to modify evaluation questions generated from a knowledge base, a subsection of which is provided to you as a list of context paragraphs. You will be given simple questions that need to be modified to include statements that describe aspects of the additional context that are unrelated to the question. While the initial question must remain preserved, the additional context should introduce elements meant to confuse the LLM and retrieval from the knowledge base, evaluating the model's ability to focus on the relevant information for the correct answer. Evolve the provided simple questions into questions made to confuse the retrieval part of a model's RAG with a distracting element from the knowledge base but irrelevant to the question. (Designed to mess with embedding engines - leaves more reasoning work for the LLM)

        You will be given an input of questions that looks like the following: 
        [{{"question": "xx1", "question_type":"simple"}}, {{"question": "xx2", "question_type":"simple"}}, ...]
        as well as some context paragraphs. 
        The user will provide the context, consisting of multiple paragraphs delimited by dashes "------". The additional context from which to generate the distracting statements will be provided subsequently, in the same format.
        You will output a list of JSON objects with the same length as the input, with key 'question' mapping to the modified question, without any other wrapping text or markdown. Additionally, modify the question_type JSON parameter for all of these questions to be 'distracting'. Ensure that you will only return valid JSON for example:
        [{{"question": "modified xx1 to include distracting statements", "question_type":"distracting"}}, {{"question": "modified xx2 to include distracting statements", "question_type":"distracting"}}, ...].\n"""
,
    "double":"""Your role is to modify evaluation questions generated from a knowledge base, a subsection of which is provided to you as a list of context paragraphs. You will be given simple questions that need to be modified to include two distinct parts, each requiring a different piece of information from the provided context. The questions should be compound queries that consist of two distinct parts, evaluating the capabilities of the query rewriter of the RAG to accurately address both parts of the question using the provided context. You may generate the second part of the question based on the additional context provided. Evolve the provided simple questions into questions with two distinct parts to evaluate the capabilities of the query rewriter of the RAG. These questions should have 2 different answers or 2 different parts of the same answer.
    
        You will be given an input of questions that looks like the following: 
        [{{"question": "xx1", "question_type":"simple"}}, {{"question": "xx2", "question_type":"simple"}}, ...]
        as well as some context paragraphs. 
        The user will provide the context, consisting of multiple paragraphs delimited by dashes "------". The additional context from which to generate the second part of the question will be provided subsequently, in the same format.
        You will output a list of JSON objects with the same length as the input, with key 'question' mapping to the modified question, without any other wrapping text or markdown. Additionally, modify the question_type JSON parameter for all of these questions to be 'double'. Ensure that you will only return valid JSON for example:
        [{{"question": "modified xx1 to include 2nd part", "question_type": "double"}}, {{"question": "modified xx2 to include 2nd part", "question_type":"double"}}, ...].\n"""
,
    "conditionals": """Your role is to modify evaluation questions generated from a knowledge base, a subsection of which is provided to you as a list of context paragraphs. You will be given simple questions that need to be modified to include hypothetical or situational conditions, introducing complexity through conditional queries. The questions should require the model to determine the appropriate responses based on the conditions presented in the context and the structure of the question. Evolve the provided simple questions into questions that introduce a conditional element, adding complexity to the question. 

        You will be given an input of questions that looks like the following: 
        [{{"question": "xx1", "question_type":"simple"}}, {{"question": "xx2", "question_type":"simple"}}, ...]
        as well as some context paragraphs. 
        The user will provide the context, consisting of multiple paragraphs delimited by dashes "------". The additional context from which you may choose generate the conditional part of the question will be provided subsequently, in the same format.
        You will output a list of JSON objects with the same length as the input, with key 'question' mapping to the modified question, without any other wrapping text or markdown. Additionally, modify the question_type JSON parameter for all of these questions to be 'conditionals'. Ensure that you will only return valid JSON for example:
        [{{"question": "modified xx1 to include conditionals", "question_type":"conditionals"}}, {{"question": "modified xx2 to include conditionals", "question_type":"conditionals"}}, ...].\n"""

    
}
################################################################### PLAN VERIFICATION PROMPTS ###################################################################
VERIFICATION_QUESTION_TEMPLATE_PROMPT = """Your task is to create a verification question based on the below question provided.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question: Was [movie actor] born in [Boston]
Explanation: In the above example the verification question focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place).
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and generate verification question.

Actual Question: {original_question}

Final Verification Question:"""

VERIFICATION_QUESTION_PROMPT = """Your task is to create a series of verification questions based on the below question, the verification question template and baseline response.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Was [movie actor] born in Boston?
Example Baseline Response: Some movie actors born in Boston include: Matt Damon, Chris Evans.
Example Verification Question: 1. Was Matt Damon born in Boston?
2. Was Chris Evans born in Boston?

Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place) based on the template and substitutes entity values from the baseline response.
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and substitute the entity values from the baseline response to generate verification questions.

Actual Question: {original_question}
Baseline Response: {baseline_response}
Verification Question Template: {verification_question_template}

Final Verification Questions, separated by \\n:"""


################################################################## EXECUTE VERIFICATION PROMPTS ##################################################################
EXECUTE_PLAN_PROMPT_SEARCH_TOOL = """Answer the following question correctly based on the provided context. The question could be tricky as well, so think step by step and answer it correctly.

Context: {search_result}

Question: {verification_question}

Answer:"""


EXECUTE_PLAN_PROMPT_SELF_LLM = """Answer the following question correctly.

Question: {verification_question}

Answer:"""

EXECUTE_PLAN_PROMPT = "{verification_questions}"

################################################################## FINAL REFINED PROMPTS ##################################################################
FINAL_REFINED_PROMPT = """Given the below `Original Query` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer.
Original Query: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs to affirm your answer:
{verification_answers}

Remember, you must ONLY answer the original query or provide a refined answer to the original query. If the initial query asks for a specific format of response, ensure that you maintain that format. 
Final Refined Answer:\n"""

################################################################## HALLUCINATION GENERATION ##################################################################
HALLUCINATION_GENERATION_SYSTEM_PROMPT = """Your role is to generate hallucinated answers for a given question from a knowledge base. You will be given a question and an answer, and your task is to create new answers that are plausible but do not necessarily align with the provided context.

The user will provide the question and the original answer.
You will return a list of {num_hallucinated_answers} JSON objects with keys 'question' and 'hallucinated_answer', without any other wrapping text or markdown. Ensure that you will only return valid JSON for example:
[{{"question": "xx", "hallucinated_answer": "yy", "question_type":"provided type of question xx"}}, {{"question": "xx1", "hallucinated_answer": "zz", "question_type":"provided type of question xx1"}}, ...].\n"""


################################################################## HALLUCINATION CHECK ##################################################################

HALLUCINATION_CHECK_PROMPT = """Your task is to check the factual consistency between the baseline response and the final answer. 
If there is any conflicting or contradictory information between the baseline response and the final answer, or if any information in the baseline response cannot be verified in the final answer,respond with "Yes". Otherwise, respond with "No".

Baseline Response: {baseline_response}
Final Answer: {final_answer}

Consistency Check:"""

################################################################## AGENT DESCRIPTIONS ##################################################################

AGENT_DESCRIPTIONS_MAP = {
    "reasoning": """Questions designed to enhance the need for reasoning to answer them effectively (at least one leap of intuition required to correlate the answer to the correct information from the knowledge base)""",
    "multi_context":  """Questions that necessitate information from multiple related sections or chunks of knowledge to formulate a single answer.""",
    "situational": """Questions including user context to evaluate the ability of the generation to produce relevant answer according to the context (first part of the question establishes some [correct OR distracting] context prior to the question itself)""",
    "distracting": """Questions made to confuse the retrieval part of a model's RAG with a distracting element from the knowledge base but irrelevant to the question. (Designed to mess with embedding engines - leaves more reasoning work for the LLM)""",
    "double": """Questions with two distinct parts to evaluate the capabilities of the query rewriter of the RAG. These questions should have 2 different answers or 2 different parts of the same answer.""",
    "conditionals": """Questions that introduce a conditional element, adding complexity to the question."""
}

# ################################################################## FILTERING PROMPTS ##################################################################
# FILTER_QUESTIONS_PROMPT = """Your role is to filter questions generated by the question generation system. You will be given a list of questions and your task is to remove questions that do not meet the criteria for the given question type.

# For each question, you will be provided with the following information:
# - Question: The question generated by the question generation system
# - Answer: The answer to the question
# - Context: The context in which the question is asked
# - Question Type: The type of question generated by the question generation system. This will be one of the following: ['simple', 'reasoning', 'multi_context', 'situational', 'distracting', 'double', 'conditionals'].

# Definitions of each question type:
# simple: Basic questions that do not require complex reasoning or multiple contexts.
# reasoning: Questions that require complex reasoning and logical connections between different pieces of data within the provided context.
# multi_context: Questions that require the integration of information from multiple sources or sections within the knowledge base.
# situational: Questions that include statements that describe aspects of the context that may or may not be related to the question.
# distracting: Questions that include statements that describe aspects of the additional context that are unrelated to the question.
# double: Questions that consist of two distinct parts, each requiring a different piece of information from the provided context.
# conditionals: Questions that introduce complexity through conditional queries.

# Ensure that the answer to the question is found inside the provided context. The question must be self-contained and the question and answer must be in the provided language.

# You will output a list of filtered questions that meet the criteria for the given question type. Ensure that you will only remove questions that do not meet the criteria, and that you will only return valid JSON that matches the format of the input questions. 

# Your task is only to REMOVE questions. Do not modify any of the questions' individual properties. You must remove the entire question object if it does not meet the criteria for the given question type. If the question is valid, return the question as is. If the question is not valid, return an empty string. Ensure that you will only return valid JSON for example:
# <BEGIN EXAMPLE JSON>
# [{{
#             "question": "How did Disney's experiences with distributors like Pat Powers and Charles Mintz shape his business strategies?",
#             "answer": "Disney's experiences with distributors like Pat Powers and Charles Mintz significantly shaped his business strategies. Disputes with Powers over profit shares led to Disney's nervous breakdown and the eventual loss of Powers as a distributor. This experience made Disney more cautious and strategic in his business dealings. His conflict with Mintz over the Oswald series, where he lost many of his animation staff and the rights to Oswald, led Disney to create Mickey Mouse and seek more favorable distribution terms with companies like Columbia Pictures and United Artists.",
#             "context": {{
#                 "240b90ac-5658-4256-8546-378cacb9f3da": "Mickey Mouse first appeared in May 1928 as a single test screening of the short Plane Crazy,...",
#                 "e11da559-b931-4f79-adb3-f62d2d05369c": "She was losing the rights to both the Out of the Inkwelland Felix the Cat cartoons, and...",
#                 "98e65eee-7100-40dd-8a74-78f3a3b0b597": "Walt Disney introduces each of the seven dwarfs in a scene from the original 1937 Snow White..."
#             }},
#             "total_tokens_used_in_this_batch": 10661,
#         }}]
          
# <END EXAMPLE JSON>
# """
####################################################### PROMPT EVOLUTION ADD'L CRITERIA #######################################################
PROMPT_EVOLUTION_ADDITIONAL_CRITERIA = """
Your generated questions must meet the following criteria:
- <IMPORTANT>Do not use words that reference the context directly, such as "this", "that", "it", etc. Additionally, do not include phrases such as "as mentioned in the context" or "according to the passage" in the question. Assume the model that is being tested cannot see the context when answering the question.</IMPORTANT>
- The answer to the question should be found inside the provided context
- The question must be self-contained - in other words, the question must explicitly state any necessary titles, names, or terms. Assume that the model to be tested with this dataset has no prior knowledge of the context. Avoid using pronouns or references that are not explicitly mentioned in the context. Additionally, there must be a singular and clear answer to the question. 
- Ensure that the modifications you make to the questions lead to and preserve a fixed and singular answer to said question, and that the potential answer to the question has not changed since the context was provided.
- When generating questions with numerical answers, ensure that the number is relatively easy to verify, without margin for interpretation. For example, if the answer is a date, ensure that the date is clearly stated in the context. If the answer is a large number, ensure that the number is explicitly mentioned in the context. Assume that the model cannot see the context, and the number in question can be generally understood without additional context.
- The questions should be specific and unambiguous, with a clear and singular answer. Avoid generating questions that are vague, open-ended, or have multiple possible answers. Ensure that the question is clear and concise, with a single correct answer that can be found in the provided context.
- The questions should be relevant to the context provided, with the answer being directly supported by the information in the context. Ensure that the question is closely related to the information in the context, and that the answer can be found within the context paragraphs.
- Do not let the question contain the explicit answer to the question. The question should not contain any direct references to the answer, and the answer should not be explicitly stated in the question. The question should be phrased in a way that requires the model to use reasoning to provide the correct answer.
- If the question has two parts, ensure that the second part does not simply reword the first part. The two parts should be distinctly worded and written.
- Assume that each question must stand on its own without previous questions. If you choose to generate two questions about the same topic, make sure that each question is self-contained and does not rely on the other question for context. Example: if the context is about a report, reference the report by name in any questions you generate. 
- <IMPORTANT>DO NOT USE THE WORDS/PHRASES "THE CONTEXT", "THE DOCUMENT", "THE INFORMATION", "THE PROVIDED..." etc IN ANY QUESTION. THE TEST SET WILL NOT INCLUDE THE CONTEXT SO MODELS WILL NOT KNOW WHAT THAT MEANS AND THAT WOULD BE INEFFECTIVE.</IMPORTANT>
"""

################################################################## FILTERING PROMPTS ##################################################################
FILTER_QUESTIONS_PROMPT = """Your role is to filter questions generated by the question generation system. You will receive a list of questions to validate. Each question should be self-contained and not reference any external context.

A question is considered valid if:
1. It can be answered without needing to see any external context
2. All necessary proper nouns and specific references are included in the question
3. It avoids phrases like "in this context", "in the document", etc.

You will receive input in this format:
[
  {
    "id": 0,
    "question": "What is the question text?"
  }
]

You must respond ONLY with a JSON array in exactly this format:
[
  {
    "id": 0,
    "question": "What is the question text?",
    "valid": "true"
  }
]

Set "valid" to "true" if the question is self-contained, or "false" if it references unnamed context.

DO NOT include any other text or explanation in your response - ONLY the JSON array.
"""
################################################################## EVALUATION PROMPTS ##################################################################
HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION = """
I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#: Yes
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No
#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be 'Yes' or 'No'.
The answer you give MUST be 'Yes' or 'No'. DO NOT RESPOND WITH ANY OTHER WORDS, THOUGHTS, OR EXPLANATIONS. YOUR EXPLANATION CAN ONLY, EXACTLY ONLY BE THE STRING 'Yes' OR THE STRING 'No'."""

HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION_RAG = """
I want you to act as an answer judge. Given a question, an answer, and possibly a support context with similar questions and their answers, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if there is a factual contradiction between the answer and the world knowledge or the provided support context. Some information in the answer might be fabricated.
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Support Context#: 
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#: Yes

#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Support Context#: Similar Question: What highway provides access to Zilpo Road and is known as Midland Trail? Answer: U.S Highway 60 is also known as Midland Trail and gives access to Zilpo Road.
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
#Question#: What genre do Superheaven and Oceansize belong to?
#Support Context#: 
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No

#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Support Context#: Similar Question: What profession do Kōbō Abe and Agatha Christie have in common? Answer: Kōbō Abe and Agatha Christie were both renowned authors.
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the provided support context and knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Support Context#: Similar Question: How many members are in the bands Muse and The Raconteurs? Answer: Muse is a rock band consisting of three members, while The Raconteurs is a rock band consisting of four members.
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Support Context#: Similar Question: What is the current status of the Temagami-Lorrain Mine compared to Meadowbank Gold Mine? Answer: Meadowbank Gold Mine is still producing gold, whereas the Temagami-Lorrain Mine has been inactive for years.
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the Temagami-Lorrain Mine has been inactive for years.
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types and the provided support context if available. The answer you give MUST be 'Yes' or 'No'. DO NOT RESPOND WITH ANY OTHER WORDS, THOUGHTS, OR EXPLANATIONS. YOUR EXPLANATION CAN ONLY, EXACTLY ONLY BE THE STRING 'Yes' OR THE STRING 'No'.
"""
