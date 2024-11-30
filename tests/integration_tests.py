import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import MagicMock
from qa_testset_generator.model import LLM, Embedding
from qa_testset_generator.knowledge_base import KnowledgeBase, load_documents
from qa_testset_generator.question_generator import OutputGenerator, QuestionGenerator, Agent, Hallucinator
import json

class TestIntegration(unittest.TestCase):

    def setUp(self):
        print("Setting up integration tests...")
        # Initialize LLM, Embedding, and KnowledgeBase
        self.llm = LLM()
        self.embedding = Embedding()
        self.documents = load_documents("test_docs")  # Use test documents for the knowledge base
        self.knowledge_base = KnowledgeBase(self.llm, self.embedding, self.documents)

    def test_question_and_answer_generation(self):
        print("Testing question and answer generation...")
        # Test question generation for the 'simple' question type
        generator = QuestionGenerator(self.llm, 'simple')
        questions_and_answers, context_documents, doc_ids, doc_scores = generator.generate_qa_pairs(
            self.knowledge_base, num_questions=2, language="en", batch=1)
        
        # Check the structure of the output
        self.assertTrue(isinstance(questions_and_answers, list))
        self.assertGreater(len(questions_and_answers), 0)
        for qa in questions_and_answers:
            self.assertIn('question', qa)
            self.assertIn('answer', qa)
            self.assertIn('context', qa)

    def test_output_generation(self):
        print("Testing output generation...")
        # Test output file generation
        questions = [
            {"question": "Test question 1", "answer": "Test answer 1", "context": "Context 1", "question_type": "simple"},
            {"question": "Test question 2", "answer": "Test answer 2", "context": "Context 2", "question_type": "reasoning"}
        ]
        output_generator = OutputGenerator(questions, ["simple", "reasoning"], "output.json")
        output_generator.output_data()

        # Check if output file was created and has the correct content
        with open("output.json", 'r') as file:
            data = json.load(file)
            self.assertIn("simple", data)
            self.assertIn("reasoning", data)
            self.assertEqual(len(data["simple"]), 1)
            self.assertEqual(len(data["reasoning"]), 1)

    def test_hallucination_generation(self):
        # Test hallucination generation
        hallucinator = Hallucinator(output_file="hallucinations.json", llm=self.llm)
        hallucinated_answers = hallucinator.generate_hallucinations(
            question="What is AI?", answer="AI is artificial intelligence.", num_hallucinated_answers=2, language="en", batch=1)

        # Check the structure of the hallucinated answers
        self.assertTrue(isinstance(hallucinated_answers, list))
        self.assertEqual(len(hallucinated_answers), 2)
        for ha in hallucinated_answers:
            self.assertIn('question', ha)
            self.assertIn('hallucinated_answer', ha)

    def test_end_to_end_process(self):
        
        # Test the full end-to-end process
        question_generator = QuestionGenerator(self.llm, 'simple')
        qa_pairs, context_documents, doc_ids, doc_scores = question_generator.generate_qa_pairs(
            self.knowledge_base, num_questions=2, language="en", batch=1)

        output_generator = OutputGenerator(qa_pairs, ["simple"], "final_output.json")
        output_generator.output_data()

        hallucinator = Hallucinator(output_file="hallucinations_final.json", llm=self.llm)
        hallucinator.process_file(input_file="final_output.json", num_hallucinated_answers=2, language="en", batch=2)

        # Ensure that all files are created and have content
        with open("final_output.json", 'r') as file:
            final_output = json.load(file)
            self.assertIn("simple", final_output)

        with open("hallucinations_final.json", 'r') as file:
            hallucinations_output = json.load(file)
            self.assertIn("simple", hallucinations_output)
            self.assertIn('hallucinated_answers', hallucinations_output['simple'][0])

if __name__ == '__main__':
    print("Running integration tests...")
    unittest.main()