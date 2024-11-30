import unittest
from unittest.mock import MagicMock, patch
# import ultraimport
# question_generator = ultraimport('qa_testset_generator/question_generator.py', "QuestionGenerator", package=2)
from qa_testset_generator.qa_pair_generator import QuestionGenerator, Agent, OutputGenerator, Hallucinator
from qa_testset_generator.model import LLM, Embedding
from qa_testset_generator.knowledge_base import KnowledgeBase
import json

AGENT_TYPES = ['description', 'simple', 'reasoning', 'multi_context', 'situational', 'double', 'conditional']

class TestQuestionGenerator(unittest.TestCase):

    def setUp(self):
        self.llm = MagicMock(spec=LLM)
        self.embedding = MagicMock(spec=Embedding)
        self.knowledge_base = MagicMock(spec=KnowledgeBase)

    def test_create_agent_valid(self):
        for agent_type in AGENT_TYPES:
            generator = QuestionGenerator(self.llm, agent_type)
            agent = generator.create_agent()
            self.assertIsInstance(agent, Agent)

    def test_create_agent_invalid(self):
        generator = QuestionGenerator(self.llm, 'unknown')
        with self.assertRaises(ValueError):
            generator.create_agent()

    def test_generate_single_question_batch_valid(self):
        agent = Agent(self.llm, 'description', 'simple')
        self.llm.get_response.return_value = '[{"question": "What is AI?", "question_type": "simple"}]'
        
        questions = agent.generate_single_question_batch(
            self.knowledge_base, num_questions=1, language='en', batch=1, context_str='Some context'
        )
        self.assertIsInstance(questions, str)
        self.assertIn('question', questions)

    def test_generate_single_question_batch_invalid_json(self):
        agent = Agent(self.llm, 'description', 'simple')
        self.llm.get_response.return_value = 'invalid json'
        
        with patch('json.loads', side_effect=json.JSONDecodeError("Expecting value", "doc", 0)):
            questions = agent.generate_single_question_batch(
                self.knowledge_base, num_questions=1, language='en', batch=1, context_str='Some context'
            )
            self.assertEqual(questions, '')

    def test_generate_single_answer_batch_valid(self):
        agent = Agent(self.llm, 'description', 'simple')
        self.llm.get_response.return_value = '[{"question": "What is AI?", "answer": "Artificial Intelligence", "question_type": "simple"}]'
        
        answers = agent.generate_single_answer_batch(
            self.knowledge_base, num_questions=1, language='en', batch=1, raw_questions='Some question', context_str='Some context'
        )
        self.assertIsInstance(answers, list)
        self.assertGreater(len(answers), 0)
        self.assertIn('answer', answers[0])

    def test_generate_single_answer_batch_invalid_json(self):
        agent = Agent(self.llm, 'description', 'simple')
        self.llm.get_response.return_value = 'invalid json'
        
        with patch('json.loads', side_effect=json.JSONDecodeError("Expecting value", "doc", 0)):
            answers = agent.generate_single_answer_batch(
                self.knowledge_base, num_questions=1, language='en', batch=1, raw_questions='Some question', context_str='Some context'
            )
            self.assertEqual(answers, [])

    def test_output_generator_create_dict(self):
        questions = [
            {"question": "Q1", "answer": "A1", "context": "C1", "question_type": "simple"},
            {"question": "Q2", "answer": "A2", "context": "C2", "question_type": "reasoning"},
        ]
        output_generator = OutputGenerator(questions, ["simple", "reasoning"], "output.json")
        result = output_generator.create_dict()
        
        self.assertIn("simple", result)
        self.assertIn("reasoning", result)
        self.assertEqual(len(result["simple"]), 1)
        self.assertEqual(len(result["reasoning"]), 1)

    def test_output_generator_output_data(self):
        questions = [
            {"question": "Q1", "answer": "A1", "context": "C1", "question_type": "simple"},
            {"question": "Q2", "answer": "A2", "context": "C2", "question_type": "reasoning"},
        ]
        output_generator = OutputGenerator(questions, ["simple", "reasoning"], "output.json")
        output_generator.output_data()

        with open("output.json", 'r') as file:
            data = json.load(file)
            self.assertIn("simple", data)
            self.assertIn("reasoning", data)
            self.assertEqual(len(data["simple"]), 1)
            self.assertEqual(len(data["reasoning"]), 1)

    def test_hallucinator_generate_hallucinations(self):
        hallucinator = Hallucinator(output_file="hallucinations.json", llm=self.llm)
        self.llm.get_response.return_value = '[{"question": "What is AI?", "hallucinated_answer": "Some wrong answer", "question_type": "simple"}]'
        
        hallucinations = hallucinator.generate_hallucinations(
            question="What is AI?", answer="AI is artificial intelligence.", num_hallucinated_answers=2, language="en", batch=1
        )
        self.assertIsInstance(hallucinations, list)
        self.assertEqual(len(hallucinations), 1)
        self.assertIn('hallucinated_answer', hallucinations[0])


if __name__ == '__main__':
    unittest.main()
