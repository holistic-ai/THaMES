import copy
import json
import os
import re
import uuid
from typing import List, Dict, Set
import json_repair
from llmlingua import PromptCompressor
import numpy as np
import pandas as pd
import random
import tiktoken
from tqdm import tqdm
import time 
# import logger


import argparse


import shutil

from dotenv import load_dotenv

load_dotenv()

from model import LLM, Embedding

from knowledge_base import KnowledgeBase, load_documents

import dataset_filter as df

from prompts import HALLUCINATION_GENERATION_SYSTEM_PROMPT, QUESTION_GENERATION_SYSTEM_PROMPT, ANSWER_GENERATION_SYSTEM_PROMPT, QUESTION_EVOLUTION_PROMPT_MAP, AGENT_DESCRIPTIONS_MAP, PROMPT_EVOLUTION_ADDITIONAL_CRITERIA, FILTER_QUESTIONS_PROMPT

from pathlib import Path
import sys

from pathlib import Path
import sys
from typing import List

from llama_index.core import SimpleDirectoryReader
from pathlib import Path
import sys

import difflib

QUESTION_TYPES_MAP = {1: 'simple', 2: 'reasoning', 3:'multi_context', 4:'distracting', 5:'double', 6:'conditionals'}


def validate_questions(llm, input_tokens, output_tokens, questions: List[Dict[str, str]], output_dir: str, regex_module=re) -> tuple[List[Dict[str, str]], int, int]:
    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(validation_dir, f"validation_response_{timestamp}.txt")
    
    match_pattern = r"(?i).*?(this\s+context|the\s+context|this\s+document|the\s+document|mentioned\s+above|as\s+mentioned|according\s+to|provided\s+above).*?"
    
    print(f"\nValidating {len(questions)} questions...")
    
    valid_questions = []
    for question in questions:
        if isinstance(question, dict) and 'question' in question:
            if isinstance(question['question'], str) and question['question'].strip():
                valid_questions.append(question)
            else:
                print(f"Skipping malformed question: {question}")
    
    print(f"After format validation: {len(valid_questions)} questions")
    
    questions_to_validate = []
    direct_valid = []
    
    for question in valid_questions:
        if regex_module.match(match_pattern, question['question']):
            questions_to_validate.append(question)
        else:
            direct_valid.append(question)
    
    print(f"Questions needing LLM validation: {len(questions_to_validate)}")
    print(f"Directly valid questions: {len(direct_valid)}")
    
    if questions_to_validate:
        # Prepare questions for validation - simplify the structure
        validation_questions = []
        for i, question in enumerate(questions_to_validate):
            # Create a simplified version of the question for validation
            validation_questions.append({
                "id": i,
                "question": question['question']
            })
        
        batch_size = 10
        for i in range(0, len(validation_questions), batch_size):
            batch = validation_questions[i:i + batch_size]
            
            # Format the input as a clean JSON array
            user_input = f"Please validate these questions:\n{json.dumps(batch, indent=2)}"
            
            response = llm.get_response(prompt=FILTER_QUESTIONS_PROMPT + user_input)
            
            try:
                validated_batch = json.loads(response)
                if isinstance(validated_batch, list):
                    for validated_q in validated_batch:
                        if validated_q.get('valid', '').lower() == 'true':
                            # Find and add the original question
                            original_q = next((q for q in questions_to_validate 
                                            if q['question'] == validated_q['question']), None)
                            if original_q:
                                direct_valid.append(original_q)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                try:
                    # Try to extract JSON from the response using regex
                    import re
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        validated_batch = json.loads(json_match.group())
                        for validated_q in validated_batch:
                            if validated_q.get('valid', '').lower() == 'true':
                                original_q = next((q for q in questions_to_validate 
                                                if q['question'] == validated_q['question']), None)
                                if original_q:
                                    direct_valid.append(original_q)
                except Exception as e2:
                    print(f"Failed to parse response after cleanup attempt: {e2}")
                    with open(output_path, 'a') as f:
                        f.write(f"\nProblematic response:\n{response}\n")
    
    print(f"Final valid questions count: {len(direct_valid)}")
    return direct_valid, input_tokens, output_tokens

class OutputGenerator:
    

    def __init__(self, questions, question_types: list, output_file: str):
        self.questions = questions  # array of arrays
        self.question_types = question_types
        self.output_file = output_file
        self.keys = []


    def create_dict(self):
        questions_dict = dict()
        for question in self.questions:
            question_type = question['question_type']
            if question_type in self.question_types:
                if question_type not in questions_dict:
                    questions_dict[question_type] = []
                if (question['question'] != '' and question['answer'] != '' and question['context'] != ''):
                    questions_dict[question_type].append({
                        "question": question['question'],
                        "answer": question['answer'],
                        "context": question['context'],
                        "total_input_tokens_used_in_this_batch": question['total_input_tokens_used_in_this_batch'],
                        "total_output_tokens_used_in_this_batch": question['total_output_tokens_used_in_this_batch']
                        # "total_tokens_used_in_this_batch": question['total_tokens_used_in_this_batch']
                    })
        return questions_dict

    def output_data(self):
        questions_dict = self.create_dict()
        self.keys = list(questions_dict.keys())
        raw_json = json.dumps(questions_dict, indent=4)
        with open(self.output_file, 'w') as outfile:
            outfile.write(raw_json)
        print("Output file written to: ", self.output_file)
        print("length of each question type: ", [len(questions_dict[key]) for key in questions_dict.keys()])
        print("total number of questions: ", sum([len(questions_dict[key]) for key in questions_dict.keys()]))

    def get_keys(self):
        return self.keys


class Agent:
    def __init__(self, llm, agent_desc, agent_type, output_file: str):
        self.llm = llm
        self.agent_desc = agent_desc
        self.agent_type = agent_type
        self.tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.encoder = tiktoken.get_encoding("cl100k_base")
        # self.prompt_compressor = PromptCompressor("openai-community/gpt2", device_map="cuda")  # or 'cpu' if using cpu
        self.prompt_compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True, device_map="cpu")
        self.output_file = output_file
        self.used_document_combinations = set()

    def get_tokens_used(self):
        return self.tokens_used

    def get_input_tokens_used(self):
        return self.input_tokens

    def get_output_tokens_used(self):
        return self.output_tokens

    def get_agent_description(self):
        return self.agent_desc

    def generate_single_question_batch(self, knowledge_base: KnowledgeBase, num_questions: int, language: str,
                                       batch: int, context_str: str):
        agent_description = self.get_agent_description()
        prompt = QUESTION_GENERATION_SYSTEM_PROMPT.format(agent_description=agent_description, language=language,
                                                          batch=batch, num_questions=num_questions) + PROMPT_EVOLUTION_ADDITIONAL_CRITERIA
        user_input = f"Context:\n------\n{context_str}\n------"

        print(f"Generating batch {batch} with {num_questions} questions...")
        
        self.tokens_used += len(self.encoder.encode(prompt + user_input))
        self.input_tokens += len(self.encoder.encode(prompt + user_input))
        response = self.llm.get_response(prompt=prompt + user_input)
        self.tokens_used += len(self.encoder.encode(response))
        self.output_tokens += len(self.encoder.encode(response))
        
        # Debug logging
        print(f"Raw response received: {response[:200]}...")  # Print first 200 chars of response
        
        question_batch = []
        try:
            question_batch = json.loads(response)
            print(f"Successfully parsed JSON with {len(question_batch)} questions")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            # Attempt to repair the JSON if decoding fails
            try:
                repaired_json = json_repair.repair_json(response, skip_json_loads=True, return_objects=False)
                question_batch = json.loads(repaired_json)
                print(f"Successfully repaired JSON with {len(question_batch)} questions")
            except Exception as e:
                print(f"Failed to repair JSON: {e}")
                print("Response that caused error:", response)
                return []
        
        if not isinstance(question_batch, list):
            print(f"Unexpected response format. Expected list, got {type(question_batch)}")
            return []
        
        if not question_batch:
            print("No questions were generated in this batch")
            return []
        
        # validate questions
        question_batch, validation_input_tokens, validation_output_tokens = validate_questions(
            self.llm, 
            input_tokens=self.input_tokens, 
            output_tokens=self.output_tokens, 
            questions=question_batch, 
            output_dir=os.path.dirname(self.output_file),
            regex_module=re
        )
        
        print(f"After validation: {len(question_batch)} questions remaining")
        
        self.input_tokens += validation_input_tokens
        self.output_tokens += validation_output_tokens
        self.tokens_used += validation_input_tokens + validation_output_tokens
        
        return question_batch

    def evolve_question_batch(self, knowledge_base: KnowledgeBase, num_questions: int, language: str, batch: int,
                              raw_questions: str, context_str: str):
        return raw_questions, context_str


    def generate_single_answer_batch(self, knowledge_base: KnowledgeBase, num_questions: int, language: str, batch: int,
                                     raw_questions: str, context_str: str):
        agent_description = self.get_agent_description()
        prompt = ANSWER_GENERATION_SYSTEM_PROMPT.format(agent_description=agent_description, language=language,
                                                        batch=batch, num_questions=num_questions)
        user_input = f"Questions:\n-----\n{raw_questions}\n\nContext:\n------\n{context_str}\n------\n"

        self.tokens_used += len(self.encoder.encode(prompt + user_input))
        self.input_tokens += len(self.encoder.encode(prompt + user_input))
        response = self.llm.get_response(prompt=prompt + user_input)
        self.tokens_used += len(self.encoder.encode(response))
        self.output_tokens += len(self.encoder.encode(response))
        qa = []
        try:
            qa = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to decode JSON response even after repair attempt. Skipping this batch.")
        return qa

    def generate_qa_pairs(self, knowledge_base: KnowledgeBase, num_questions: int, language: str, batch: int):
        min_questions_threshold = int(num_questions * 0.8)  # 80% of requested questions
        max_attempts = 5  # Maximum number of full generation attempts
        attempt = 0
        
        # Add check for minimum unique documents
        if len(knowledge_base.nodes) < 3:
            print("Warning: Not enough unique documents to generate diverse questions")
            return [], [], [], []
        
        while attempt < max_attempts:
            attempt += 1
            print(f"\nAttempt {attempt}/{max_attempts} to generate {num_questions} questions (minimum {min_questions_threshold})...")
            
            # Get seed document and context
            seed_attempts = 1
            seed_document = None
            raw_context_documents = []
            while seed_document is None or raw_context_documents == []:
                seed_document = knowledge_base.get_random_node_weighted()
                raw_context_documents = knowledge_base.get_neighbors(seed_document, n_neighbors=4, similarity_threshold=0.7)
                
                # Check document combination hasn't been used before
                doc_combination = frozenset([doc.node_id for doc in raw_context_documents])
                
                # Add content similarity check
                content_too_similar = self._check_content_similarity(raw_context_documents)
                
                if (doc_combination in self.used_document_combinations or 
                    content_too_similar or 
                    not self._meets_content_requirements(raw_context_documents)):
                    seed_document = None
                    raw_context_documents = []
                    continue
                
                seed_attempts += 1
                if(seed_attempts > 10):
                    print("Failed to generate suitable seed document after 10 attempts. Trying new batch.")
                    break
                    
            if seed_document is None or raw_context_documents == []:
                continue  # Try the next full attempt
                
            # Add the document combination to used set
            doc_combination = frozenset([doc.node_id for doc in raw_context_documents])
            self.used_document_combinations.add(doc_combination)
            
            doc_ids = [doc.node_id for doc in raw_context_documents]
            doc_scores = [doc.get_score() for doc in raw_context_documents]

            context_documents = [document.get_content() for document in raw_context_documents]
            compressed_context_documents = [
                self.prompt_compressor.compress_prompt(doc, instruction="", question="", rate=0.4)[
                    'compressed_prompt']
                for doc in context_documents
            ]
            context_str = "\n------\n".join(compressed_context_documents)
            
            # Generate questions with increased number to account for filtering
            adjustment_factor = 1.5  # Request 50% more questions than needed
            adjusted_num_questions = int(num_questions * adjustment_factor)
            questions = self.generate_single_question_batch(knowledge_base, adjusted_num_questions, language, batch, context_str)
            
            if not questions:
                continue
                
            # update context if necessary based on evolution of questions
            questions, context_str = self.evolve_question_batch(knowledge_base, adjusted_num_questions, language, batch, 
                                                              questions, context_str)
                                                              
            qa_pairs = self.generate_single_answer_batch(knowledge_base, adjusted_num_questions, language, batch,
                                                       questions, context_str)
                                                       
            if len(qa_pairs) >= min_questions_threshold:
                print(f"Successfully generated {len(qa_pairs)} questions")
                return qa_pairs, context_documents, doc_ids, doc_scores
            else:
                print(f"Generated {len(qa_pairs)} questions, below minimum threshold of {min_questions_threshold}. Trying again...")
                
        # If we've exhausted all attempts
        print(f"Warning: After {max_attempts} attempts, could not generate minimum required questions.")
        if qa_pairs:  # Return what we have if it's not empty
            print(f"Returning {len(qa_pairs)} questions (below requested minimum of {min_questions_threshold})")
            return qa_pairs, context_documents, doc_ids, doc_scores
        return [], [], [], []

    def generate_questions_and_answers(self, knowledge_base: KnowledgeBase, num_questions: int, language: str,
                                       batch: int):
        data = []
        qa_pairs, context_documents, doc_ids, doc_scores = self.generate_qa_pairs(knowledge_base, num_questions,
                                                                                  language, batch)
        for qa_pair in qa_pairs:
            if isinstance(qa_pair, dict):  # Ensure qa_pair is a dictionary
                qa_pair['context'] = dict(zip(doc_ids, context_documents))
                if(qa_pair['context'] == {}):
                    print("Skipping invalid qa_pair due to context being empty: ", qa_pair)
                    continue
                qa_pair['total_input_tokens_used_in_this_batch'] = self.get_input_tokens_used()
                qa_pair['total_output_tokens_used_in_this_batch'] = self.get_output_tokens_used()
                data.append(qa_pair)
            else:
                print(f"Skipping invalid qa_pair: {qa_pair}")

        return data

    def dump_output(self, data, output_file_path="output.json"):
        with open(output_file_path, 'w') as outfile:
            outfile.write(data)

    def _check_content_similarity(self, documents: List) -> bool:
        """Check if documents are too similar in content."""
        if not documents:
            return True
            
        # Compare each pair of documents
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                content1 = self._normalize_content(documents[i].get_content())
                content2 = self._normalize_content(documents[j].get_content())
                
                # Calculate similarity ratio
                similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
                if similarity > 0.8:  # Threshold for considering documents too similar
                    return True
        
        return False
    
    def _meets_content_requirements(self, documents: List) -> bool:
        """Check if documents meet minimum content requirements."""
        if not documents or len(documents) < 3:
            return False
            
        # Check total content length
        total_length = sum(len(doc.get_content()) for doc in documents)
        if total_length < 2000:
            return False
            
        # Check individual document lengths
        if any(len(doc.get_content()) < 200 for doc in documents):
            return False
            
        return True
    
    def _normalize_content(self, content: str) -> str:
        """Normalize document content for comparison."""
        # Remove whitespace and convert to lowercase
        normalized = ' '.join(content.lower().split())
        
        # Remove common formatting artifacts
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove very common words that don't affect meaning
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = normalized.split()
        normalized = ' '.join(word for word in words if word not in stopwords)
        
        return normalized

class EvolutionAgent(Agent):
    def __init__(self, llm, agent_desc, agent_type, output_file: str):
        super().__init__(llm, agent_desc, agent_type, output_file)

        self.QUESTION_EVOLUTION_SYSTEM_PROMPT = QUESTION_EVOLUTION_PROMPT_MAP[agent_type].format(language="en") + PROMPT_EVOLUTION_ADDITIONAL_CRITERIA
    def evolve_question_batch(self, knowledge_base: KnowledgeBase, num_questions: int, language: str, batch: int,
                              raw_questions: str, context_str: str):
        prompt = self.QUESTION_EVOLUTION_SYSTEM_PROMPT
        user_input = f"""{raw_questions}"""

        self.tokens_used += len(self.encoder.encode(prompt + user_input))
        self.input_tokens += len(self.encoder.encode(prompt + user_input))
        response = self.llm.get_response(prompt=prompt + user_input)
        self.tokens_used += len(self.encoder.encode(response))
        self.output_tokens += len(self.encoder.encode(response))

        evolved_questions = ""
        try:
            evolved_questions = json.loads(response)
        except json.JSONDecodeError:
            # Attempt to repair the JSON if decoding fails
            try:
                repaired_json = json_repair.repair_json(response, skip_json_loads=True, return_objects=False)
                evolved_questions = json.loads(repaired_json) # type: ignore
            except json.JSONDecodeError:
                print("Failed to decode JSON response even after repair attempt. Skipping this batch.")
        evolved_questions, validation_input_tokens, validation_output_tokens = validate_questions(self.llm, input_tokens=self.input_tokens, output_tokens=self.output_tokens, questions=evolved_questions, output_dir=os.path.dirname(self.output_file)) # type: ignore
        self.input_tokens += validation_input_tokens
        self.output_tokens += validation_output_tokens
        self.tokens_used += validation_input_tokens + validation_output_tokens
        return evolved_questions, context_str

    


class QuestionGenerator:
    def __init__(self, llm, question_type, output_file: str):
        self.llm = llm
        self.question_type = question_type
        self.agent = None
        self.output_file = output_file
        self.used_document_combinations = set()
    def create_agent(self):
        if self.question_type != "simple":
            # print(f"Creating agent for question type: {self.question_type}")
            self.agent = EvolutionAgent(self.llm, AGENT_DESCRIPTIONS_MAP[self.question_type], self.question_type, self.output_file)
        else:
            # print("creating simple agent")
            self.agent = Agent(self.llm,
                                "This agent generates basic questions that do not require complex reasoning or multiple contexts.",
                                'simple', self.output_file)

        if self.agent is None:
            raise ValueError(f"Unknown question type: {self.question_type}")
        return self.agent

    def generate_qa_pairs(self, knowledge_base: KnowledgeBase, num_questions: int, language: str, batch: int):
        if (self.agent is None):
            self.create_agent()
            
        # Check if we've used too many document combinations
        total_docs = len(knowledge_base.nodes)
        max_combinations = min(total_docs * (total_docs - 1) // 2, 1000)  # Theoretical max combinations
        
        if len(self.used_document_combinations) >= max_combinations:
            print("Warning: Running out of unique document combinations. Resetting combination history.")
            self.used_document_combinations.clear()
            
        # Share the used combinations with the agent
        self.agent.used_document_combinations = self.used_document_combinations  # type: ignore
        
        result = self.agent.generate_questions_and_answers(knowledge_base, num_questions, language, batch)  # type: ignore
        
        # Update our record of used combinations
        self.used_document_combinations = self.agent.used_document_combinations  # type: ignore
        
        return result


class Hallucinator:
    def __init__(self, output_file: str, llm: LLM):
        self.output_file = output_file
        self.llm = llm
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generate_hallucinations(self, question: str, answer: str, num_hallucinated_answers: int, language: str,
                                batch: int) -> List[Dict[str, str]]:
        prompt = HALLUCINATION_GENERATION_SYSTEM_PROMPT.format(num_hallucinated_answers=num_hallucinated_answers)
        user_input = f"Question:\n{question}\nAnswer:\n{answer}\n"

        self.tokens_used += len(self.encoder.encode(prompt + user_input))
        self.input_tokens += len(self.encoder.encode(prompt + user_input))
        response = self.llm.get_response(prompt=prompt + user_input)
        self.tokens_used += len(self.encoder.encode(response))
        self.output_tokens += len(self.encoder.encode(response))

        hallucinated_answers = []
        try:
            hallucinated_answers = json.loads(response)
        except json.JSONDecodeError:
            # Attempt to repair the JSON if decoding fails
            try:
                repaired_json = json_repair.repair_json(response, skip_json_loads=True, return_objects=False)
                hallucinated_answers = json.loads(repaired_json) # type: ignore
            except json.JSONDecodeError:
                print("Failed to decode JSON response even after repair attempt. Skipping this batch.")
        return hallucinated_answers

    def process_file(self, input_file: str, num_hallucinated_answers: int, language: str, batch: int):
        with open(input_file, 'r') as infile:
            data = json.load(infile)

        for category, qa_pairs in data.items():
            for qa_pair in tqdm(qa_pairs, desc=f"Generate Hallu QA Pairs in {category}"):
                question = qa_pair['question']
                answer = qa_pair['answer']
                hallucinated_answers = self.generate_hallucinations(question, answer, num_hallucinated_answers,
                                                                    language, batch)
                qa_pair['hallucinated_answers'] = [hallucinated_answer["hallucinated_answer"] for hallucinated_answer in
                                                   hallucinated_answers]

        with open(self.output_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)

def get_document_sources(test_docs_path: Path) -> tuple[list, bool]:
    """
    Get available document sources from test_docs directory.
    Returns tuple of (list of paths, is_subdirectory_structure)
    """
    try:
        # Get all items in test_docs
        items = list(test_docs_path.iterdir())
        
        # Filter for directories and files
        directories = [d for d in items if d.is_dir()]
        files = [f for f in items if f.is_file() and f.suffix in ['.txt', '.pdf', '.doc', '.docx']]
        
        # Determine structure and return appropriate list
        if directories:
            return directories, True
        elif files:
            return files, False
        else:
            sys.exit("Error: No valid documents or directories found in 'test_docs'")
            
    except Exception as e:
        sys.exit(f"Error accessing test_docs directory: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate QA pairs from documents')
    
    # Required arguments
    parser.add_argument('--categories', type=str, required=True,
                       help='Comma-separated list of document category numbers')
    
    # Optional arguments with defaults
    parser.add_argument('--question_types', nargs='+', type=int, 
                       default=list(range(1, 7)),
                       help='Space-separated list of question type numbers (1-6)')
    parser.add_argument('--num_questions', type=int, default=10,
                       help='Number of questions to generate per type')
    parser.add_argument('--num_batches', type=int,
                       help='Number of batches to split generation into')
    parser.add_argument('--hallucination', choices=['y', 'n'], default='y',
                       help='Whether to perform hallucination tasks')
    parser.add_argument('--filename', type=str,
                       help='Base filename for output')
    
    return parser.parse_args()



def load_documents(categories_str: str) -> list:
    """Load documents from the specified category paths using LlamaIndex."""
    documents = []
    category_paths = categories_str.split(',')
    
    for path in category_paths:
        path = Path(path)
        try:
            if path.is_dir():
                # Handle directory
                loader = SimpleDirectoryReader(
                    input_dir=str(path),
                    filename_as_id=True,
                    recursive=True,
                    required_exts=['.txt', '.pdf', '.doc', '.docx']
                )
                documents.extend(loader.load_data())
            elif path.is_file() and path.suffix in ['.txt', '.pdf', '.doc', '.docx']:
                # Handle single file
                loader = SimpleDirectoryReader(
                    input_files=[str(path)],
                    filename_as_id=True
                )
                documents.extend(loader.load_data())
        except Exception as e:
            print(f"Error loading from {path}: {str(e)}")
            continue
            
    if not documents:
        sys.exit("Error: No documents were successfully loaded.")
        
    print(f"Successfully loaded {len(documents)} documents")
    return documents

def main():
    args = parse_args()
    
    # Convert question types to QUESTION_TYPES format
    QUESTION_TYPES = [QUESTION_TYPES_MAP[num] for num in args.question_types if num in QUESTION_TYPES_MAP]
    if not QUESTION_TYPES:
        sys.exit("Error: No valid question types provided.")

    # Create output directory structure
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_base_dir = Path("output/final")
    output_dir = output_base_dir / f"{args.filename}_{timestamp}"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        sys.exit(f"Error creating output directory: {str(e)}")

    # Update file paths
    step1_path = output_dir / f"{args.filename}_STEP1.json"
    step2_path = output_dir / f"{args.filename}_STEP2.json"
    step3_path = output_dir / f"{args.filename}_STEP3.json"
    step4_path = output_dir / f"{args.filename}_STEP4.json"
    step5_path = output_dir / f"{args.filename}_STEP5.json"
    final_path = output_dir / f"{args.filename}_FINAL.json"

    # Initialize LLM and Knowledge Base
    llm = LLM()
    embedding = Embedding()
    documents = load_documents(args.categories)
    knowledge_base = KnowledgeBase(llm, embedding, documents)

    # Generate questions
    QUESTIONS_PER_BATCH_THRESHOLD = 20
    if args.num_questions > QUESTIONS_PER_BATCH_THRESHOLD:
        num_batches = args.num_batches or max(2, min(10, args.num_questions // QUESTIONS_PER_BATCH_THRESHOLD))
        questions_per_batch = args.num_questions // num_batches
    else:
        num_batches = 1
        questions_per_batch = args.num_questions

    # Generate QA pairs
    generators = [QuestionGenerator(llm, qt, step1_path) for qt in QUESTION_TYPES]
    all_questions = []
    
    for i in tqdm(range(num_batches), desc="QA loop"):
        for generator in generators:
            raw_output = generator.generate_qa_pairs(
                knowledge_base, 
                num_questions=questions_per_batch,
                language="en",
                batch=i
            )
            all_questions.extend(raw_output)

    # Process and filter questions
    output_generator = OutputGenerator(all_questions, QUESTION_TYPES, str(step1_path))
    output_generator.output_data()
    
    df.dataset_preprocess(str(step1_path), str(step2_path))
    df.remove_duplicate_qs(str(step2_path), str(step3_path))
    df.ragas_evaluation(str(step3_path), str(step4_path))
    
    if args.hallucination.lower() == "y":
        df.Answer_filtering(str(step4_path), str(step5_path))
        hallucinator = Hallucinator(output_file=str(step5_path), llm=llm)
        hallucinator.process_file(input_file=str(step4_path), num_hallucinated_answers=3, language="en", batch=2)
        df.NLI_filtering(str(step5_path), str(final_path))
    else:
        df.Answer_filtering(str(step4_path), str(final_path))

    return str(final_path)

if __name__ == "__main__":
    main()
#     parser = argparse.ArgumentParser(description='Generate QA pairs from documents')
#     parser.add_argument('--categories', type=str, help='Comma-separated list of document category numbers')
#     parser.add_argument('--filename', type=str, help='Output filename')
#     parser.add_argument('--question-types', type=str, help='Comma-separated list of question type numbers (1-6)')
#     parser.add_argument('--num-questions', type=int, help='Number of questions to generate per type')
#     parser.add_argument('--hallucination', choices=['y', 'n'], help='Whether to perform hallucination and NLI tasks')
#     parser.add_argument('--num-batches', type=int, help='Number of batches to generate (only used if num_questions > 20)')

#     args = parser.parse_args()

#     # Validate test_docs directory exists
#     test_docs_path = Path("test_docs")
#     if not test_docs_path.exists():
#         sys.exit("Error: 'test_docs' directory not found. Please ensure it exists in the current directory.")

#     # Get document sources and determine structure
#     sources, is_subdirectory = get_document_sources(test_docs_path)
    
#     # Validate categories input
#     if args.categories:
#         try:
#             choices = [int(x.strip()) for x in args.categories.split(',')]
#             # Validate each choice is within range
#             valid_choices = []
#             for choice in choices:
#                 if 1 <= choice <= len(sources):
#                     valid_choices.append(choice)
#                 else:
#                     print(f"Warning: Skipping invalid number {choice}")
#             if not valid_choices:
#                 sys.exit("Error: No valid source numbers provided.")
#             choices = valid_choices
#         except ValueError:
#             sys.exit("Error: Source numbers must be comma-separated numbers.")
#     else:
#         while True:
#             print("\nAvailable document sources:")
#             for i, source in enumerate(sources, 1):
#                 print(f"{i}: {source.name}")  # Just show the name, not full path
#             try:
#                 choices_input = input("\nEnter the numbers of the document sources you want to use (comma-separated): ")
#                 choices = [int(x.strip()) for x in choices_input.split(",")]
#                 if all(1 <= choice <= len(sources) for choice in choices):
#                     break
#                 print("Error: Please enter valid numbers.")
#             except ValueError:
#                 print("Error: Please enter numbers separated by commas.")

#     # Load documents based on structure
#     documents = []
#     if is_subdirectory:
#         for choice in choices:
#             try:
#                 documents.extend(load_documents(str(sources[choice - 1])))
#             except Exception as e:
#                 print(f"Error loading documents from {sources[choice - 1].name}: {str(e)}")
#                 continue
#     else:
#         for choice in choices:
#             try:
#                 documents.extend(load_documents(str(sources[choice - 1])))
#             except Exception as e:
#                 print(f"Error loading file {sources[choice - 1].name}: {str(e)}")
#                 continue

#     if not documents:
#         sys.exit("Error: No documents were successfully loaded.")

#     print(f"Successfully loaded {len(documents)} documents")

#     # Create output directory structure
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     if args.filename:
#         base_filename = args.filename
#     else:
#         while True:
#             base_filename = input("Enter the filename to save the output: ").strip()
#             if base_filename:
#                 break
#             print("Filename cannot be empty. Please try again.")

#     output_base_dir = Path("output/final")
#     output_dir = output_base_dir / f"{base_filename}_{timestamp}"
    
#     try:
#         output_dir.mkdir(parents=True, exist_ok=True)
#     except Exception as e:
#         sys.exit(f"Error creating output directory: {str(e)}")

#     # Validate question types input
#     QUESTION_TYPES_MAP = {1: 'simple', 2: 'reasoning', 3:'multi_context', 4:'distracting', 5:'double', 6:'conditionals'}
    
#     if args.question_types:
#         try:
#             question_type_nums = [int(x.strip()) for x in args.question_types.split(',')]
#             QUESTION_TYPES = []
#             for num in question_type_nums:
#                 if num in QUESTION_TYPES_MAP:
#                     QUESTION_TYPES.append(QUESTION_TYPES_MAP[num])
#                 else:
#                     print(f"Warning: Skipping invalid question type {num}")
#             if not QUESTION_TYPES:
#                 sys.exit("Error: No valid question types provided.")
#         except ValueError:
#             sys.exit("Error: Question types must be comma-separated numbers.")
#     else:
#         while True:
#             print("""
# Enter question types (comma-separated):
#     1: simple
#     2: reasoning
#     3: multi_context
#     4: distracting
#     5: double
#     6: conditionals
# Example: 1,2""")
#             try:
#                 choices_input = input("Your choice(s): ")
#                 question_type_nums = [int(x.strip()) for x in choices_input.split(",")]
#                 QUESTION_TYPES = []
#                 for num in question_type_nums:
#                     if num in QUESTION_TYPES_MAP:
#                         QUESTION_TYPES.append(QUESTION_TYPES_MAP[num])
#                     else:
#                         print(f"Warning: Skipping invalid question type {num}")
#                 if QUESTION_TYPES:
#                     break
#                 print("Error: Please enter at least one valid question type.")
#             except ValueError:
#                 print("Error: Please enter numbers separated by commas.")

#     # Validate number of questions
#     if args.num_questions:
#         if args.num_questions <= 0:
#             sys.exit("Error: Number of questions must be positive.")
#         num_questions = args.num_questions * 2
#     else:
#         while True:
#             try:
#                 num_questions = int(input("Enter the number of questions to generate for each question type: ")) * 2
#                 if num_questions <= 0:
#                     print("Error: Please enter a positive number.")
#                     continue
#                 break
#             except ValueError:
#                 print("Error: Please enter a valid number.")

#     # Validate hallucination choice
#     if not args.hallucination:
#          while True:
#             perform_halu_tasks = input("Do you want to perform hallucination and NLI tasks? (y/n): ").lower()
#             if perform_halu_tasks in ['y', 'n']:
#                 break
#             print("Error: Please enter 'y' or 'n'.")
#     else:
#         perform_halu_tasks = args.hallucination
       

#     # Update file paths to use the timestamped directory
#     step1_path = output_dir / f"{base_filename}_STEP1.json"
#     step2_path = output_dir / f"{base_filename}_STEP2.json"
#     step3_path = output_dir / f"{base_filename}_STEP3.json"
#     step4_path = output_dir / f"{base_filename}_STEP4.json"
#     step5_path = output_dir / f"{base_filename}_STEP5.json"
#     final_path = output_dir / f"{base_filename}_FINAL.json"

#     # Initialize LLM and Knowledge Base
#     llm = LLM()
#     embedding = Embedding()
    
#     print("creating knowledge base from documents of length "+str(len(documents)))
    
#     knowledge_base = KnowledgeBase(llm, embedding, documents)

    
    
#     print("generating questions and dumping into fileset: "+ base_filename)
    
#     # Determine if batching is needed
#     QUESTIONS_PER_BATCH_THRESHOLD = 20
#     if num_questions > QUESTIONS_PER_BATCH_THRESHOLD:
#         if args.num_batches:
#             num_batches = args.num_batches
#         else:
#             # Calculate reasonable number of batches based on question count
#             num_batches = max(2, min(10, num_questions // QUESTIONS_PER_BATCH_THRESHOLD))
#         questions_per_batch = num_questions // num_batches
#     else:
#         num_batches = 1
#         questions_per_batch = num_questions

#     print(f"Generating {num_questions} questions {'in ' + str(num_batches) + ' batches' if num_batches > 1 else 'in a single batch'}")
    
#     generators = []
#     all_questions = []
    
#     for question_type in QUESTION_TYPES:
#         generators.append(QuestionGenerator(llm, question_type, step1_path))

#     for i in tqdm(range(num_batches), desc="QA loop"):
#         for generator in generators:
#             raw_output = generator.generate_qa_pairs(knowledge_base, 
#                                                    num_questions=questions_per_batch,
#                                                    language="en",
#                                                    batch=i)
#             all_questions.extend(raw_output)
    
#     if(not os.path.exists("output/final/"+base_filename)):
#         os.mkdir("output/final/"+base_filename)
#     else:
#         try:
#             shutil.rmtree("output/final/"+base_filename)
#             os.mkdir("output/final/"+base_filename)
#         except OSError as e:
#             print("Error: %s - %s." % (e.filename, e.strerror))

        
    
#     print(len(all_questions))
#     output_generator = OutputGenerator(all_questions, QUESTION_TYPES, str(step1_path))
#     output_generator.output_data()
#     print("done generating qa pairs. Now filter out low quality qa pairs.")
#     df.dataset_preprocess(str(step1_path), str(step2_path))
#     df.remove_duplicate_qs(str(step2_path), str(step3_path))
#     print("done generating qa pairs. Now filter out low quality qa pairs.")
#     df.ragas_evaluation(str(step3_path), str(step4_path))
    
#     if perform_halu_tasks.lower() == "y":
#         df.Answer_filtering(str(step4_path), str(step5_path))
#         hallucinator = Hallucinator(output_file=str(step5_path), llm=llm)
#         hallucinator.process_file(input_file=str(step4_path), num_hallucinated_answers=3, language="en", batch=2)
#         print("done generating hallucinated answers. trying to perform NLI now.")
#         df.NLI_filtering(str(step5_path), str(final_path))
#     else:
#         print("Skipping Hallucination AND NLI tasks.")
#         df.Answer_filtering(str(step4_path), str(final_path))
#     print("done with all tasks!")
