import argparse
import os
from pathlib import Path
import sys
from typing import Optional, List
from enum import Enum

from evaluate_refactored import ModelType, MitigationType, EvaluationType, ModelConfig, ModelFactory, Evaluator
from qa_pair_generator import main as generate_main

def get_document_sources(test_docs_path: Path) -> tuple[list, bool]:
    """
    Get available document sources from test_docs directory.
    Returns tuple of (list of paths, is_subdirectory_structure)
    """
    try:
        items = list(test_docs_path.iterdir())
        directories = [d for d in items if d.is_dir()]
        files = [f for f in items if f.is_file() and f.suffix in ['.txt', '.pdf', '.doc', '.docx']]
        
        if directories:
            return directories, True
        elif files:
            return files, False
        else:
            sys.exit("Error: No valid documents or directories found in 'test_docs'")
            
    except Exception as e:
        sys.exit(f"Error accessing test_docs directory: {str(e)}")

class Pipeline:
    def __init__(self):
        self.args = self.parse_arguments()
        self.output_dir = None
        self.final_dataset_path = None
        self.QUESTION_TYPES_MAP = {
            1: 'simple', 
            2: 'reasoning', 
            3: 'multi_context', 
            4: 'distracting', 
            5: 'double', 
            6: 'conditionals'
        }

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Generate and evaluate QA pairs')
        
        # Generation arguments
        parser.add_argument('--categories', type=str, help='Comma-separated list of document category numbers (default: all)')
        parser.add_argument('--question_types', nargs='+', type=int,
                           default=[1,2,3,4,5,6],
                           help='Space-separated list of question type numbers (1-6)')
        parser.add_argument('--num_questions', type=int, default=10,
                           help='Number of questions to generate per type (default: 10)')
        parser.add_argument('--num_batches', type=int,
                           help='Number of batches to split generation into (default: auto-calculated)')
        parser.add_argument('--hallucination', choices=['y', 'n'], default='y',
                           help='Whether to perform hallucination tasks (default: y)')
        parser.add_argument('--filename', type=str,
                           help='Base filename for output (default: auto-generated from categories)')
        
        # Evaluation arguments
        parser.add_argument('--model_type', choices=[mt.value for mt in ModelType],
                           help=f'Model type for evaluation (default: {ModelType.AZURE.value})')
        parser.add_argument('--model_name', type=str,
                           help='Model name/path for evaluation (default: gpt-4)')
        parser.add_argument('--mitigation_techniques', nargs="*",
                           choices=[mt.value for mt in MitigationType],
                           help='Mitigation techniques to use (default: all)', 
                           default=[mt for mt in MitigationType])
        parser.add_argument('--evaluation_type', 
                           choices=[et.value for et in EvaluationType],
                           help=f'Type of evaluation to perform (default: {EvaluationType.HALUEVAL.value})')
        
        # Pipeline control
        parser.add_argument('--skip_generation', action='store_true',
                           help='Skip generation phase')
        parser.add_argument('--skip_evaluation', action='store_true',
                           help='Skip evaluation phase')
        
        return parser.parse_args()

    def prompt_for_missing_args(self):
        if not self.args.skip_generation:
            # Handle document sources
            test_docs_path = Path("test_docs")
            if not test_docs_path.exists():
                sys.exit("Error: 'test_docs' directory not found. Please ensure it exists in the current directory.")

            sources, is_subdirectory = get_document_sources(test_docs_path)
            
            if not self.args.categories:
                print("\nAvailable document sources:")
                for i, source in enumerate(sources, 1):
                    print(f"{i}: {source.name}")
                choices_input = input("\nEnter the numbers of the document sources to use (comma-separated, or press Enter for all): ").strip()
                if not choices_input:  # User pressed Enter
                    self.args.categories = ','.join(str(i) for i in range(1, len(sources) + 1))
                    print(f"Selected all categories: {self.args.categories}")
                else:
                    try:
                        choices = [int(x.strip()) for x in choices_input.split(",")]
                        if all(1 <= choice <= len(sources) for choice in choices):
                            self.args.categories = ','.join(map(str, choices))
                        else:
                            print("Invalid numbers detected, using all categories")
                            self.args.categories = ','.join(str(i) for i in range(1, len(sources) + 1))
                    except ValueError:
                        print("Invalid input, using all categories")
                        self.args.categories = ','.join(str(i) for i in range(1, len(sources) + 1))

            # Handle question types
            print("""
Enter question types (space-separated, or press Enter for all):
    1: simple
    2: reasoning
    3: multi_context
    4: distracting
    5: double
    6: conditionals
Example: 1 2""")
            choices_input = input("Your choice(s): ").strip()
            if not choices_input:  # User pressed Enter
                self.args.question_types = list(range(1, 7))
                print(f"Selected all question types: {self.args.question_types}")
            else:
                try:
                    question_type_nums = [int(x) for x in choices_input.split()]
                    valid_types = [num for num in question_type_nums if num in self.QUESTION_TYPES_MAP]
                    if valid_types:
                        self.args.question_types = valid_types
                    else:
                        print("No valid question types entered, using all types")
                        self.args.question_types = list(range(1, 7))
                except ValueError:
                    print("Invalid input, using all question types")
                    self.args.question_types = list(range(1, 7))

            # Handle number of questions
            if not self.args.num_questions:
                num_input = input("Enter the number of questions to generate for each type (or press Enter for default: 10): ").strip()
                if not num_input:  # User pressed Enter
                    self.args.num_questions = 10
                    print("Using default: 10 questions per type")
                else:
                    try:
                        num = int(num_input)
                        if num > 0:
                            self.args.num_questions = num
                        else:
                            print("Invalid number, using default: 10")
                            self.args.num_questions = 10
                    except ValueError:
                        print("Invalid input, using default: 10")
                        self.args.num_questions = 10

            # Handle hallucination choice
            if not self.args.hallucination:
                choice = input("Do you want to perform hallucination tasks? (y/n, press Enter for default: y): ").strip().lower()
                if not choice:  # User pressed Enter
                    self.args.hallucination = 'y'
                    print("Using default: performing hallucination tasks")
                elif choice in ['y', 'n']:
                    self.args.hallucination = choice
                else:
                    print("Invalid input, using default: y")
                    self.args.hallucination = 'y'

            # Handle filename - auto-generate if not provided
            if not self.args.filename:
                category_nums = self.args.categories.split(',')
                source_names = [sources[int(num)-1].name for num in category_nums]
                default_filename = '_'.join(source_names)
                filename_input = input(f"Enter the filename to save the output (or press Enter for default: {default_filename}): ").strip()
                if not filename_input:  # User pressed Enter
                    self.args.filename = default_filename
                    print(f"Using auto-generated filename: {self.args.filename}")
                else:
                    self.args.filename = filename_input

        if not self.args.skip_evaluation:
            # Model configuration defaults to Azure/GPT-4 if not specified
            if not self.args.model_type:
                model_input = input(f"Enter model type for evaluation (or press Enter for default: {ModelType.AZURE.value}): ").strip()
                if not model_input:  # User pressed Enter
                    self.args.model_type = ModelType.AZURE.value
                    print(f"Using default model type: {self.args.model_type}")
                elif model_input in [mt.value for mt in ModelType]:
                    self.args.model_type = model_input
                else:
                    print(f"Invalid model type, using default: {ModelType.AZURE.value}")
                    self.args.model_type = ModelType.AZURE.value

            if not self.args.model_name:
                model_name_input = input("Enter model name/path for evaluation (or press Enter for default: gpt-4): ").strip()
                if not model_name_input:  # User pressed Enter
                    self.args.model_name = "gpt-4"
                    print("Using default model name: gpt-4")
                else:
                    self.args.model_name = model_name_input

            if not self.args.evaluation_type:
                eval_input = input(f"Enter evaluation type (or press Enter for default: {EvaluationType.HALUEVAL.value}): ").strip()
                if not eval_input:  # User pressed Enter
                    self.args.evaluation_type = EvaluationType.HALUEVAL.value
                    print(f"Using default evaluation type: {self.args.evaluation_type}")
                elif eval_input in [et.value for et in EvaluationType]:
                    self.args.evaluation_type = eval_input
                else:
                    print(f"Invalid evaluation type, using default: {EvaluationType.HALUEVAL.value}")
                    self.args.evaluation_type = EvaluationType.HALUEVAL.value

    def run_generation(self):
        from qa_pair_generator import main as generate_main
        
        # Get source directories/files
        test_docs_path = Path("test_docs")
        sources, _ = get_document_sources(test_docs_path)
        
        # Convert category numbers to actual paths
        category_nums = [int(x.strip()) for x in self.args.categories.split(',')]
        category_paths = [str(sources[num-1]) for num in category_nums]
        categories_str = ','.join(category_paths)
        
        # Prepare arguments for generation
        sys.argv = [
            'qa_pair_generator.py',
            '--categories', categories_str,
            '--question_types'
        ]
        # Add question types as separate arguments
        sys.argv.extend(str(qt) for qt in self.args.question_types)
        
        # Add remaining arguments
        sys.argv.extend([
            '--num_questions', str(self.args.num_questions),
            '--hallucination', self.args.hallucination
        ])
        
        if self.args.filename:
            sys.argv.extend(['--filename', self.args.filename])
        if self.args.num_batches:
            sys.argv.extend(['--num_batches', str(self.args.num_batches)])
        
        print("\nGeneration settings:")
        print(f"Categories: {categories_str}")
        print(f"Question types: {self.args.question_types}")
        print(f"Questions per type: {self.args.num_questions}")
        print(f"Hallucination tasks: {self.args.hallucination}")
        print(f"Output filename: {self.args.filename}\n")
        
        # Run generation and capture the final dataset path
        self.final_dataset_path = generate_main()
        return self.final_dataset_path

    def run_evaluation(self):
        if not self.final_dataset_path:
            print("Error: No dataset available for evaluation")
            return
        
        # Create configuration
        config = ModelConfig(
            model_type=ModelType(self.args.model_type),
            model_name=self.args.model_name,
            mitigation_types=[MitigationType(mt) for mt in self.args.mitigation_techniques],
            evaluation_type=EvaluationType(self.args.evaluation_type)
        )

        # Create model and evaluator
        model = ModelFactory.create_model(config)
        evaluator = Evaluator(model, config)

        # Format and evaluate data
        if config.evaluation_type == EvaluationType.HALUEVAL:
            from evaluate_refactored import format_data_halueval as format_data
        else:
            from evaluate_refactored import format_data_ragas as format_data

        formatted_data = format_data(self.final_dataset_path)
        if not formatted_data:
            print("Error formatting data")
            return

        evaluator.evaluate(formatted_data, os.path.dirname(self.final_dataset_path))

    def run(self):
        self.prompt_for_missing_args()
        print(self.args)
        if not self.args.skip_generation:
            print("\nStarting generation phase...")
            self.final_dataset_path = self.run_generation()

        if not self.args.skip_evaluation:
            print("\nStarting evaluation phase...")
            self.run_evaluation()

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
