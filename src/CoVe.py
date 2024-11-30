import os
import itertools
from typing import Any, Dict, List, Optional

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

import prompts

# openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# if openai.api_key is None:
#     raise ValueError("OpenAI API key not found in environment variables")


class ExecuteVerificationChain(Chain):
    """
    Implements the logic to execute the verification question for factual accuracy
    """
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    input_key: str = "verification_questions"
    output_key: str = "verification_answers"
    use_search_tool: bool = True
    search_tool: Any = DuckDuckGoSearchRun()

    class Config:
        """Configuration for this pydantic object."""
        extra = 'allow'
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key."""
        return [self.output_key]

    def search_for_verification_question(self, verification_question: str) -> str:
        if not verification_question.strip():
            return " "

        search_result = self.search_tool.invoke(verification_question)
        return search_result

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, str]:
        verification_answers_list = list()  # Will contain the answers of each verification questions
        question_answer_pair = ""  # Final output of verification question and answer pair

        # Convert all the verification questions into a list of string
        sub_inputs = {k: v for k, v in inputs.items() if k == self.input_key}
        verification_questions_prompt_value = self.prompt.format_prompt(**sub_inputs)
        verification_questions_str = verification_questions_prompt_value.text
        verification_questions_list = verification_questions_str.split("\n")

        # Setting up prompt for both search tool and llm self evaluation
        execution_prompt_search_tool = PromptTemplate.from_template(prompts.EXECUTE_PLAN_PROMPT_SEARCH_TOOL)
        execution_prompt_self_llm = PromptTemplate.from_template(prompts.EXECUTE_PLAN_PROMPT_SELF_LLM)

        # Executing the verification questions, either using search tool or self llm
        for question in verification_questions_list:
            if self.use_search_tool:
                search_result = self.search_for_verification_question(question)
                execution_prompt_value = execution_prompt_search_tool.format_prompt(
                    **{"search_result": search_result, "verification_question": question})
            else:
                execution_prompt_value = execution_prompt_self_llm.format_prompt(**{"verification_question": question})
            verification_answer_llm_result = self.llm.generate_prompt([execution_prompt_value],
                                                                      callbacks=run_manager.get_child() if run_manager else None)
            verification_answer_str = verification_answer_llm_result.generations[0][0].text
            verification_answers_list.append(verification_answer_str)

        # Create verification question and answer pair
        for question, answer in itertools.zip_longest(verification_questions_list, verification_answers_list):
            question_answer_pair += "Question: {} Answer: {}\n".format(question, answer)

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: question_answer_pair}

    @property
    def _chain_type(self) -> str:
        return "execute_verification_chain"


class ListCOVEChain(object):
    def __init__(self, llm):
        self.llm = llm

    def __call__(self):
        # Create plan verification chain
        verification_question_template_prompt_template = PromptTemplate(input_variables=["original_question"],
                                                                        template=prompts.VERIFICATION_QUESTION_TEMPLATE_PROMPT)
        verification_question_template_chain = LLMChain(llm=self.llm,
                                                        prompt=verification_question_template_prompt_template,
                                                        output_key="verification_question_template")
        # Create plan verification questions
        verification_question_generation_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                                           "baseline_response",
                                                                                           "verification_question_template"],
                                                                          template=prompts.VERIFICATION_QUESTION_PROMPT)
        verification_question_generation_chain = LLMChain(llm=self.llm,
                                                          prompt=verification_question_generation_prompt_template,
                                                          output_key="verification_questions")
        # Create execution verification
        execute_verification_question_prompt_template = PromptTemplate(input_variables=["verification_questions"],
                                                                       template=prompts.EXECUTE_PLAN_PROMPT)
        execute_verification_question_chain = ExecuteVerificationChain(llm=self.llm,
                                                                       prompt=execute_verification_question_prompt_template,
                                                                       output_key="verification_answers")
        # Create final refined response
        final_answer_prompt_template = PromptTemplate(input_variables=["original_question",
                                                                       "baseline_response",
                                                                       "verification_answers"],
                                                      template=prompts.FINAL_REFINED_PROMPT)
        final_answer_chain = LLMChain(llm=self.llm,
                                      prompt=final_answer_prompt_template,
                                      output_key="final_answer")

        # Hallucination check
        hallucination_check_prompt_template = PromptTemplate(input_variables=["baseline_response", "final_answer"],
                                                             template=prompts.HALLUCINATION_CHECK_PROMPT)
        hallucination_check_chain = LLMChain(llm=self.llm,
                                             prompt=hallucination_check_prompt_template,
                                             output_key="hallucination_check")
        # Create sequential chain
        list_cove_chain = SequentialChain(
            chains=[  # baseline_response_chain,
                verification_question_template_chain,
                verification_question_generation_chain,
                execute_verification_question_chain,
                final_answer_chain,
                hallucination_check_chain],
            input_variables=["original_question", "baseline_response"],
            output_variables=["original_question",
                              "baseline_response",
                              "verification_question_template",
                              "verification_questions",
                              "verification_answers",
                              "final_answer",
                              "hallucination_check"],
            verbose=False)
        return list_cove_chain

1
class RouteCOVEChain(object):
    def __init__(self, question, response, llm):
        self.llm = llm
        self.question = question
        self.response = response

        list_cove_chain_instance = ListCOVEChain(llm)
        self.list_cove_chain = list_cove_chain_instance()

    def __call__(self):
        return self.list_cove_chain


if __name__ == "__main__":
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=512)
    llm = AzureChatOpenAI(deployment_name="gpt35", temperature=0.3, max_tokens=512, api_version="2024-02-15-preview")


    # original_query = "What is Breakthrough Starshot?"
    original_query = "Which team won the Euro Cup 2024?"
    baseline_response = "The Euro Cup 2024 was won by USA."
    router_cove_chain_instance = RouteCOVEChain(original_query, baseline_response, llm)
    router_cove_chain = router_cove_chain_instance()
    router_cove_chain_result = router_cove_chain(
        {"original_question": original_query, "baseline_response": baseline_response})

    print("\n" + 80 * "#" + "\n")
    print("Original Question: {}".format(router_cove_chain_result["original_question"]))
    print("Baseline Response: {}".format(router_cove_chain_result["baseline_response"]))
    print("Verification Question Template: {}".format(router_cove_chain_result["verification_question_template"]))
    print("Verification Question: {}".format(router_cove_chain_result["verification_questions"]))
    print("Verification Answer: {}".format(router_cove_chain_result["verification_answers"]))
    print("Final Answer: {}".format(router_cove_chain_result["final_answer"]))
    print("Hallucination Check: {}".format(router_cove_chain_result["hallucination_check"]))
    print("\n" + 80 * "#" + "\n")
