import json
import csv
import os
from typing import NotRequired, TypedDict
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
import time
import string
 
 
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, answer_similarity
from peft import PeftModel # type: ignore

from finetune import configure_and_train_model, load_and_prepare_data, tokenize_datasets # type: ignore
from prompts import FINAL_REFINED_PROMPT, HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION, HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION_RAG, VERIFICATION_QUESTION_PROMPT, VERIFICATION_QUESTION_TEMPLATE_PROMPT

from llama_index.llms.ollama import Ollama
import os

from enum import Enum
from typing import List, Optional

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0"
# model_name = 'meta-llama/Llama-2-7b'
# model_name = 'EleutherAI/gpt-neo-2.7B'

from model import Embedding
from langchain_openai import AzureChatOpenAI
# from CoVe import RouteCOVEChain

embed_model = Embedding()

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
login()

# TODO: SET TEMPERATURES AS NEEDED TO BE MOST OPTIMAL

class UserModel():
    name: str
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None
    model: Ollama | AzureChatOpenAI | AutoModelForCausalLM | PeftModel
    
    def __init__(self, name: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None, model: Ollama | AzureChatOpenAI | AutoModelForCausalLM):
        self.name = name
        self.tokenizer = tokenizer.from_pretrained(name) if tokenizer != None else None
        self.model = model.from_pretrained(name) if isinstance(model, AutoModelForCausalLM) else model
    
    # options param takes in parameters for generating the response - which experiments and mitigation types to use, if any
    def generate_response(self, question: str, answer: str | None, context: str | None, options: dict) -> dict[str, str]:
        # TODO: ADD CoVe and RAG mitigations for each evaluation type
        
        if(options['evaluation_type'] == 'halueval'):
            # TODO: ADD RAG MITIGATION and/or CoVe MITIGATION
            prompt = HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION + f"#Question#: {question}\n #Answer#: {answer}\n #Your Judgement#:"
            if('mitigation_techniques' in options):
                if('RAG' in options['mitigation_techniques']):
                    if(os.path.exists(options['dataset_parent_directory'] + f"/{self.name.replace('/', '_')}_failed_qa_pairs.json")):
                    # grab similar failed questions from the dataset
                        with open(options['dataset_parent_directory'] + f"/{self.name.replace('/', '_')}_failed_qa_pairs.json", 'r') as f:
                            failed_qa_pairs = json.load(f)
                        
                        new_question_embedding = embed_model.get_text_embedding(question)
                        indexed_embeddings = np.array([qa['embedding'] for qa in failed_qa_pairs])
                        similarities = cosine_similarity([new_question_embedding], indexed_embeddings)[0] # type: ignore

                        # Filter the top indices based on the similarity threshold of 0.9
                        top_indices = similarities.argsort()[-3:][::-1]
                        relevant_qa_pairs = [failed_qa_pairs[idx] for idx in top_indices if similarities[idx] > 0.8]
                        support_context = "You have previously failed to judge the hallucination status of the following questions. Please use these questions to help you make a more informed judgement: \n"
                        
                        for i, failed_qa_pair in enumerate(relevant_qa_pairs):
                            support_context += f"Question {i+1}: {failed_qa_pair['question']}\n"
                            support_context += f"Your incorrect choice of answer: {failed_qa_pair['hallucinated_answer']}\n"

                        prompt = HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION_RAG + f"#Question#: {question}\n #Answer#: {answer}\n #Support Context#: {context}\n #Your Judgement#:"
                    else:
                        prompt = HALUEVAL_FEW_SHOT_PROMPT_INSTRUCTION_RAG + f"#Question#: {question}\n #Answer#: {answer}\n #Your Judgement#:"
            
        elif(options['evaluation_type'] == 'ragas'):
            prompt = f"""Answer the following question: {question}"""
            if 'RAG' in options['mitigation_techniques']:
                if(os.path.exists(options['dataset_parent_directory'] + f"/{self.name.replace('/', '_')}_{options['mitigation_techniques'][0] if options['mitigation_techniques'] else ''}_failed_qa_pairs.json")):
                    # grab similar failed questions from the dataset
                    with open(options['dataset_parent_directory'] + f"/{self.name.replace('/', '_')}_{options['mitigation_techniques'][0] if options['mitigation_techniques'] else ''}_failed_qa_pairs.json", 'r') as f:
                        failed_qa_pairs = json.load(f)
                    
                    new_question_embedding = embed_model.get_text_embedding(question)
                    indexed_embeddings = np.array([qa['embedding'] for qa in failed_qa_pairs])
                    similarities = cosine_similarity([new_question_embedding], indexed_embeddings)[0] # type: ignore

                    # Filter the top indices based on the similarity threshold of 0.9
                    top_indices = similarities.argsort()[-3:][::-1]
                    relevant_qa_pairs = [failed_qa_pairs[idx] for idx in top_indices if similarities[idx] > 0.8]
                    support_context = "You have previously answered these questions incorrectly: \n"
                    
                    for i, failed_qa_pair in enumerate(relevant_qa_pairs):
                        support_context += f"Question {i+1}: {failed_qa_pair['question']}\n"
                        support_context += f"Your incorrect answer: {failed_qa_pair['hallucinated_answer']}\n"
                    
                    prompt = support_context + "\n" + prompt
                    
        elif(options['evaluation_type'] == 'general_question'):
            prompt = question
        else:
            return {"response": "Invalid evaluation type", "status": "failure"}
        
        
            

        if isinstance(self.model, AutoModelForCausalLM) or (isinstance(self.model, PeftModel)):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            if(self.tokenizer == None or self.model == None):
                print('failed to initialize model or tokenizer')
                return {"response": "local tokenizer or model not initalized properly. have you installed them?", "status": "failure"}
            else:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                outputs = self.model.generate(input_ids=inputs['input_ids'].to('cuda'), pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=1)# type: ignore
                prompt_length = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][prompt_length:]).strip()
        elif isinstance(self.model, Ollama):
            raw_response = self.model.complete(prompt=prompt)
            response = raw_response.text
        elif isinstance(self.model, AzureChatOpenAI):
            raw_response = self.model.invoke(prompt).content
            response = str(raw_response)
        else:
            return {"response": "Model not supported for RAGAS.", "status": "failure"}
        
        if len(response) == 0:
            return {"response": "No judgement generated", "status": "failure"}
        else:
            if('mitigation_techniques' in options and 'CoVe' in options['mitigation_techniques']):
                try:
                    baseline_response = response
                    original_question = prompt
                    self.generate_cove(baseline_response, original_question)    
                except Exception as e:
                    return {"response": str(e), "status": "failure"}
            print(response)
            return {"response": response, "status": "success"} 
    
        
    def generate_cove(self, baseline_response, original_question):
        # define verification questions by creating a template to generate questions
        verification_question_prompt = VERIFICATION_QUESTION_TEMPLATE_PROMPT.format(original_question=original_question)
        verification_question_template_response = self.generate_response(verification_question_prompt, None, None, options={"evaluation_type": "general_question"})
        if(verification_question_template_response['status'] != "success"):
            raise Exception("Failed to generate verification question plan")
        verification_question_template = verification_question_template_response['response']
        
        # generate verification questions
        verification_question_prompt = VERIFICATION_QUESTION_PROMPT.format(original_question=original_question, baseline_response=baseline_response, verification_question_template=verification_question_template)
        verification_question_response = self.generate_response(verification_question_template, None, None, {"evaluation_type": "general_question"})
        if verification_question_response['status'] != "success":
            raise Exception("Failed to generate verification questions")
        verification_questions = verification_question_response['response']
        
        # generate answers to verification questions
        verification_answers_response = self.generate_response("Answer the following questions, putting each answer in the line after its corresponding question, returning your question-answer pairs in the order provided. ##QUESTIONS##:\n" + verification_questions, None, None, {"evaluation_type": "general_question"})
        if verification_answers_response['status'] != "success":
            raise Exception("Failed to generate verification answers")
        verification_answers = verification_answers_response['response']
        
        # format into final answer to original question
        final_execution_prompt = FINAL_REFINED_PROMPT.format(original_question=original_question, baseline_response=baseline_response, verification_answers=verification_answers_response['response'])
        final_execution_response = self.generate_response(final_execution_prompt, None, None, {"evaluation_type": "general_question"})
        if final_execution_response['status'] != "success":
            raise Exception("Failed to generate final execution response")
        response = final_execution_response['response']
        return response
    
        
            

    

# Runs HaluEval evaluation, given a formatted dataset  where each question has a corresponding answer and context, with the answer category being either a hallucination or not.
def halueval_pipeline(formatted_data, dataset_parent_directory, model: UserModel, mitigation_techniques=["CoVe", "RAG"]):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    print("Starting HaluEval evaluation...")
    failed_qa_pairs = []    
    metrics = {}
    question_types = set()
    
    for i in range(len(formatted_data['question'])):
        # get all unique question types
        question_type = formatted_data['question_type'][i]
        question_types.add(question_type)
        
    for question_type in question_types:
        if question_type not in metrics.keys():
            metrics[question_type] = {'false_positive': 0, 'false_negative': 0, 'true_positive': 0, 'true_negative': 0}
    for i in tqdm(range(len(formatted_data['question'])), desc="Generating responses for HaluEval evaluation"):
        question = formatted_data['question'][i]
        answer = formatted_data['answer'][i]
        context = formatted_data['contexts'][i]
        is_hallucination = formatted_data['is_hallucination'][i]
        correct_answer = formatted_data['correct_answer'][i]
        distractor = formatted_data['hallucinated_answer'][i]
        question_type = formatted_data['question_type'][i]
        
        embedding = embed_model.get_text_embedding(question)
        try:
            judgement = model.generate_response(question, answer, context, {"evaluation_type": "halueval", "mitigation_techniques": mitigation_techniques, 'dataset_parent_directory': dataset_parent_directory})
            print(judgement)
        except Exception as e:
            print("Failed to generate judgement for question: ", question)
            print(e)
            continue
        
        indexed_qa_pair = {
                'id': i,
                'question': question,
                'correct_answer': correct_answer,
                'hallucinated_answer': distractor,
                'judgement': judgement['response'],
                'context': context,
                'embedding': embedding,
                'question_type': question_type
        }
                
            
        
        
        
        if judgement['status'] == "success":
            if judgement['response'] == "Yes":
                if is_hallucination:
                    metrics[question_type]['true_positive'] += 1
                    true_positive += 1
                else:
                    metrics[question_type]['false_positive'] += 1
                    false_positive += 1
                    failed_qa_pairs.append(indexed_qa_pair)
            elif judgement['response'] == "No":
                if not is_hallucination:
                    metrics[question_type]['true_negative'] += 1
                    true_negative += 1
                else:
                    metrics[question_type]['false_negative'] += 1
                    false_negative += 1
                    failed_qa_pairs.append(indexed_qa_pair)

            else:
                print("Invalid judgement: ", judgement['response'])
        else:
            print("Failed to generate valid judgement for question: ", question)
            continue
    # calculate metrics for each question type
    for question_type in metrics.keys():
        metrics[question_type]['accuracy'] = (metrics[question_type]['true_positive'] + metrics[question_type]['true_negative']) / (metrics[question_type]['true_positive'] + metrics[question_type]['true_negative'] + metrics[question_type]['false_positive'] + metrics[question_type]['false_negative']) if (metrics[question_type]['true_positive'] + metrics[question_type]['true_negative'] + metrics[question_type]['false_positive'] + metrics[question_type]['false_negative']) else 0
        metrics[question_type]['precision'] = metrics[question_type]['true_positive'] / (metrics[question_type]['true_positive'] + metrics[question_type]['false_positive']) if (metrics[question_type]['true_positive'] + metrics[question_type]['false_positive']) else 0
        metrics[question_type]['recall'] = metrics[question_type]['true_positive'] / (metrics[question_type]['true_positive'] + metrics[question_type]['false_negative']) if (metrics[question_type]['true_positive'] + metrics[question_type]['false_negative']) else 0
        metrics[question_type]['f1_score'] = 2 * (metrics[question_type]['precision'] * metrics[question_type]['recall']) / (metrics[question_type]['precision'] + metrics[question_type]['recall']) if (metrics[question_type]['precision'] + metrics[question_type]['recall']) else 0
        
           
                
    # calculate total metrics
    if 'total' not in metrics.keys():
        metrics['total'] = {}
    metrics['total']['accuracy'] = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) else 0
    metrics['total']['precision'] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
    metrics['total']['recall'] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
    metrics['total']['f1_score'] = 2 * (metrics['total']['precision'] * metrics['total']['recall']) / (metrics['total']['precision'] + metrics['total']['recall']) if (metrics['total']['precision'] + metrics['total']['recall']) else 0

    print("Finished generating responses for HaluEval evaluation.")
    
    with open(dataset_parent_directory + f"/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) + '_' if len(mitigation_techniques) > 0 else ''}halueval_results.json", 'w') as f:
        f.write(json.dumps(metrics))
    with open(dataset_parent_directory + f"/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) + '_' if len(mitigation_techniques) > 0 else ''}failed_qa_pairs.json", 'w') as f:
        f.write(json.dumps(failed_qa_pairs))
    print(f"Results written to {dataset_parent_directory}/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) + '_' if len(mitigation_techniques) > 0 else ''}halueval_results.json")
    
        
        
   
    



    

def ragas_pipeline(formatted_data, dataset_parent_directory, model: UserModel, mitigation_techniques=["CoVe", "RAG"]):
   
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    azure_configs = {
        "base_url": AZURE_OPENAI_ENDPOINT,
        "model_deployment": "gpt35",
        "model_name": "gpt35",
        "embedding_deployment": "text-embedding-3-large",
        "embedding_name": "text-embedding-3-large",  # most likely
    }

    azure_model = AzureChatOpenAI(
        openai_api_version="2023-05-15", # type: ignore
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    )

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    azure_embeddings = AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15", # type: ignore
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    )
    print("Starting RAGAS evaluation...")
    if len(formatted_data['answer']) > 0 or formatted_data['answer'] == None:
        formatted_data['answer'] = []
    # generate answers for each question in the dataset
    for i in tqdm(range(len(formatted_data['question'])), desc="Generating responses for RAGAS evaluation"):
        question = formatted_data['question'][i]
        ground_truth = formatted_data['ground_truth'][i]
        context = formatted_data['contexts'][i]
        try:
            answer = model.generate_response(question, None, context, {"evaluation_type": "ragas", "mitigation_techniques": mitigation_techniques, "dataset_parent_directory": dataset_parent_directory})['response']
        except Exception as e:
            answer = ""
        formatted_data['answer'].append(answer)
    # split the formatted_data into a list of arrays grouped by question type
    question_types = set(formatted_data['question_type'])
    question_type_data = {}
    for question_type in question_types:
        question_type_data[question_type] = {
            'question': [],
            'answer': [],
            'ground_truth': [],
            'contexts': []
        }
    
    for i in range(len(formatted_data['question'])):
        question = formatted_data['question'][i]
        ground_truth = formatted_data['ground_truth'][i]
        context = formatted_data['contexts'][i]
        answer = formatted_data['answer'][i]
        question_type = formatted_data['question_type'][i]
        question_type_data[question_type]['question'].append(question)
        question_type_data[question_type]['ground_truth'].append(ground_truth)
        question_type_data[question_type]['contexts'].append(context)
        question_type_data[question_type]['answer'].append(answer)
        
    
    
    metrics = {}
    
    for question_type in question_type_data.keys():
        ragas_dataset = Dataset.from_dict(question_type_data[question_type])
        print(f"Evaluating {question_type} questions...")
        score = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy, answer_correctness, answer_similarity], llm=azure_model, embeddings=azure_embeddings)
        metrics[question_type] = score
    if 'total' not in metrics.keys():
        metrics['total'] = {}
    total_score = evaluate(Dataset.from_dict(formatted_data), metrics=[faithfulness, answer_relevancy, answer_correctness, answer_similarity], llm=azure_model, embeddings=azure_embeddings)
    metrics['total'] = total_score
    metrics['mitigation_techniques'] = str(mitigation_techniques)
    try:
        with open(dataset_parent_directory + f"/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) if len(mitigation_techniques) > 0 else ''}_ragas_results.json", 'w') as f:
            f.write(json.dumps(metrics))
        print(f"Results written to {dataset_parent_directory}/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) + '_' if len(mitigation_techniques) > 0 else ''}ragas_results.json")
    except Exception as e:
        print("Error writing to file: ", e, "Writing to text file instead")
        with open(dataset_parent_directory + f"/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) if len(mitigation_techniques) > 0 else ''}_ragas_results.txt", 'w') as f:
            f.write(str(metrics))
        print(f"Results written to {dataset_parent_directory}/{model.name.replace('/', '_')}_{str(mitigation_techniques[0]) + '_' if len(mitigation_techniques) > 0 else ''}ragas_results.txt")

def format_data_ragas(path_to_data):
 # extract questions, correct answers from each category in file in input directory
    with open(path_to_data, 'r') as file:
        data = json.load(file)
    formatted_data = {
        'question': [],
        'answer': [],
        'ground_truth': [],
        'contexts': [],
        'question_type': []
    }

    # assuming the same format as batches of files in ./output/final directories
    for question_type in data.keys():
        for qa_pair in data[question_type]:
            question = qa_pair['question']
            
            # take the testset answer as the ground_truth (we assume this to be the correct answer given an effective testset)
            ground_truth = qa_pair['answer']
            formatted_data['question'].append(question)
            formatted_data['ground_truth'].append(ground_truth)
            formatted_data['contexts'].append([qa_pair['context']]) if 'context' in qa_pair else formatted_data['contexts'].append(None)
            formatted_data['question_type'].append(question_type)
            
    if(len(formatted_data['question']) == 0):
        print("No data found in input file")
        return
    # CHECK THAT ALL LENGTHS ARE EQUAL
    if len(formatted_data['question']) == len(formatted_data['ground_truth']):
        print("successfully formatted data for RAGAS!")
        return formatted_data
    else:
        print("Data formatting error")
        return {}



def format_data_halueval(path_to_data):
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    
    formatted_data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'is_hallucination': [],
        'correct_answer': [],
        'hallucinated_answer': [],
        'question_type': []
    }


    question_types = [key for key in data.keys()]

    for question_type in question_types:
        for qa_pair in tqdm(data[question_type], desc=f"Processing {question_type}"):
            # for each question in the category, extract the question, correct answer, distractor, and context            
            question = qa_pair['question']
            correct_answer = qa_pair['answer']
            distractor = qa_pair['best_distractor']
            context = qa_pair['context'] if 'context' in qa_pair else None

            # randomly choose between correct answer and distractor
            if random.random() < 0.5:
                chosen_answer = correct_answer
                is_hallucination = False
            else:
                chosen_answer = distractor
                is_hallucination = True
                
            formatted_data['question'].append(question)
            formatted_data['answer'].append(chosen_answer)
            formatted_data['contexts'].append(context) if 'context' != None else formatted_data['contexts'].append(None)
            formatted_data['is_hallucination'].append(is_hallucination)
            formatted_data['question_type'].append(question_type)
            formatted_data['correct_answer'].append(correct_answer)
            formatted_data['hallucinated_answer'].append(distractor)
            
    # assert that all lengths are equal
    if len(formatted_data['question']) == len(formatted_data['answer']) == len(formatted_data['contexts']) == len(formatted_data['is_hallucination']):
        print("successfully formatted data for HaluEval!")
        return formatted_data
    print("Data formatting error")
    return {}



class ModelType(Enum):
    LOCAL = "local"
    AZURE = "azure"
    OLLAMA = "ollama"
    FINETUNED = "finetuned"
    FINETUNE_AND_SAVE = "finetune_and_save"  # New type for finetuning process

class MitigationType(Enum):
    NONE = "none"
    COVE = "CoVe"
    RAG = "RAG"
    COVE_RAG = "CoVe_RAG"

class EvaluationType(Enum):
    HALUEVAL = "halueval"
    RAGAS = "ragas"

class ModelConfig:
    def __init__(self, 
                 model_type: ModelType,
                 model_name: str,
                 mitigation_types: List[MitigationType],
                 evaluation_type: EvaluationType,
                 finetuned_path: Optional[str] = None,
                 training_data_path: Optional[str] = None,
                 push_to_hub: bool = False,
                 hub_model_id: Optional[str] = None):
        self.model_type = model_type
        self.model_name = model_name
        self.mitigation_types = mitigation_types
        self.evaluation_type = evaluation_type
        self.finetuned_path = finetuned_path
        self.training_data_path = training_data_path
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id

class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig) -> UserModel:
        tokenizer = None
        model = None

        if config.model_type == ModelType.FINETUNE_AND_SAVE:
            from finetune import (
                load_and_prepare_data,
                initialize_model_and_tokenizer,
                tokenize_datasets,
                configure_and_train_model
            )
            
            # Load and prepare training data
            train_dataset, val_dataset = load_and_prepare_data(config.training_data_path)
            
            # Initialize base model and tokenizer
            base_model, tokenizer = initialize_model_and_tokenizer(config.model_name)
            
            # Prepare datasets
            tokenized_train_dataset, tokenized_val_dataset = tokenize_datasets(
                train_dataset, val_dataset, tokenizer
            )
            
            # Configure and train
            trainer = configure_and_train_model(
                base_model, 
                tokenized_train_dataset, 
                tokenized_val_dataset, 
                tokenizer,
                config.model_name
            )
            
            # Train the model
            trainer.train()
            
            # Save the model locally
            if config.finetuned_path:
                trainer.save_model(config.finetuned_path)
            
            # Push to Hub if requested
            if config.push_to_hub and config.hub_model_id:
                trainer.push_to_hub(config.hub_model_id)
            
            model = trainer.model

        elif config.model_type == ModelType.FINETUNED:
            tokenizer = AutoTokenizer
            base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
            model = PeftModel.from_pretrained(
                base_model, 
                config.finetuned_path or "wu981526092/THAMES_Llama3.1_Finetuned"
            )
        elif config.model_type == ModelType.LOCAL:
            tokenizer = AutoTokenizer
            model = AutoModelForCausalLM
        elif config.model_type == ModelType.AZURE:
            model = AzureChatOpenAI(
                openai_api_version="2023-05-15",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment="gpt35",
                model="gpt35",
                validate_base_url=False,
            )
        elif config.model_type == ModelType.OLLAMA:
            model = Ollama(model=config.model_name)

        return UserModel(config.model_name, tokenizer, model)

class Evaluator:
    def __init__(self, model: UserModel, config: ModelConfig):
        self.model = model
        self.config = config

    def evaluate(self, formatted_data: dict, dataset_parent_directory: str):
        if self.config.evaluation_type == EvaluationType.HALUEVAL:
            return self._run_halueval(formatted_data, dataset_parent_directory)
        elif self.config.evaluation_type == EvaluationType.RAGAS:
            return self._run_ragas(formatted_data, dataset_parent_directory)

    def _run_halueval(self, formatted_data: dict, dataset_parent_directory: str):
        return halueval_pipeline(
            formatted_data=formatted_data,
            dataset_parent_directory=dataset_parent_directory,
            model=self.model,
            mitigation_techniques=[mt.value for mt in self.config.mitigation_types]
        )

    def _run_ragas(self, formatted_data: dict, dataset_parent_directory: str):
        return ragas_pipeline(
            formatted_data=formatted_data,
            dataset_parent_directory=dataset_parent_directory,
            model=self.model,
            mitigation_techniques=[mt.value for mt in self.config.mitigation_types]
        )

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM for hallucination detection.')
    parser.add_argument('--model_type', 
                       choices=[mt.value for mt in ModelType], 
                       required=True,
                       help='Choose which type of model to use')
    parser.add_argument('--model_name', 
                       required=True, 
                       help='Specify the model name/path')
    parser.add_argument('--mitigation_techniques', 
                       nargs="*", 
                       choices=[mt.value for mt in MitigationType],
                       default=[MitigationType.NONE.value],
                       help='Specify mitigation techniques to use')
    parser.add_argument('--evaluation_type', 
                       choices=[et.value for et in EvaluationType],
                       required=True,
                       help='Specify the type of evaluation to perform')
    parser.add_argument('--dataset', 
                       required=True,
                       help='Path to the evaluation dataset')
    parser.add_argument('--finetuned_path',
                       help='Path to save/load finetuned model weights')
    parser.add_argument('--training_data_path',
                       help='Path to training data for finetuning')
    parser.add_argument('--push_to_hub',
                       action='store_true',
                       help='Push finetuned model to Hugging Face Hub')
    parser.add_argument('--hub_model_id',
                       help='Hugging Face Hub model ID for uploading (e.g., "username/model-name")')
    
    args = parser.parse_args()

    # Create configuration
    config = ModelConfig(
        model_type=ModelType(args.model_type),
        model_name=args.model_name,
        mitigation_types=[MitigationType(mt) for mt in args.mitigation_techniques],
        evaluation_type=EvaluationType(args.evaluation_type),
        finetuned_path=args.finetuned_path,
        training_data_path=args.training_data_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )

    # Create model
    model = ModelFactory.create_model(config)

    # Format data based on evaluation type
    if config.evaluation_type == EvaluationType.HALUEVAL:
        formatted_data = format_data_halueval(args.dataset)
    else:
        formatted_data = format_data_ragas(args.dataset)

    if not formatted_data:
        print("Error formatting data")
        return

    # Run evaluation
    evaluator = Evaluator(model, config)
    evaluator.evaluate(formatted_data, os.path.dirname(args.dataset))

if __name__ == '__main__':
    main()
    
    
    

# Example commands:
# poetry run python evaluate_refactored.py --llm local --dataset ./output/final/academic_political_wikipedia_500 --model_name EleutherAI/gpt-neo-2.7B --mitigation_techniques CoVe 
# poetry run python evaluate_refactored.py --llm ollama --dataset ./output/final/academic_political_wikipedia_500 --model_name llama3.1 --mitigation_techniques RAG --evaluation_type RAGAS
# poetry run python evaluate_refactored.py --llm azure --dataset ./output/final/academic_political_wikipedia_500 --model_name gpt-4o --mitigation_techniques CoVe --evaluation_type RAGAS 

            
        
    
