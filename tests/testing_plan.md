# Testing Plan for `QuestionGenerator`

This testing plan outlines the approach for both unit and integration tests for the `QuestionGenerator` class and its associated components. The goal is to ensure that each component functions correctly in isolation (unit tests) and that they work together as expected when integrated (integration tests).

---

## 1. **Unit Testing Plan**

Unit tests focus on testing individual methods and components in isolation, ensuring they behave as expected under various conditions.

### **1.1. Unit Test Objectives**

- Verify that each method in `QuestionGenerator` and its associated classes (e.g., `Agent`, `LLM`, `KnowledgeBase`) performs its intended function correctly.
- Test edge cases, input validation, and error handling.
- Mock dependencies to isolate the functionality of each method.

### **1.2. Unit Test Cases**

1. **`QuestionGenerator.create_agent()`**
   - **Test Case 1.1**: Verify that the correct agent is created based on the `question_type`.
     - **Input**: `question_type = 'simple'`, `question_type = 'reasoning'`, etc.
     - **Expected Output**: Instance of the appropriate agent class (e.g., `Agent`, `ReasoningAgent`).
   - **Test Case 1.2**: Verify that an exception is raised for an unknown `question_type`.
     - **Input**: `question_type = 'unknown'`
     - **Expected Output**: `ValueError` is raised.

2. **`Agent.generate_single_question_batch()`**
   - **Test Case 2.1**: Test successful question generation with valid inputs.
     - **Input**: Mocked `KnowledgeBase`, `num_questions`, `language`, `batch`, `context_str`.
     - **Expected Output**: List of generated questions in JSON format.
   - **Test Case 2.2**: Test behavior when the JSON response is invalid.
     - **Input**: Mocked `KnowledgeBase`, invalid JSON response from LLM.
     - **Expected Output**: Error handling and appropriate logging.
   - **Test Case 2.3**: Test token usage calculation.
     - **Input**: Various `prompt` and `user_input` sizes.
     - **Expected Output**: Correct token count for `input_tokens`, `output_tokens`, and `tokens_used`.

3. **`Agent.generate_single_answer_batch()`**
   - **Test Case 3.1**: Test successful answer generation with valid questions.
     - **Input**: Mocked `KnowledgeBase`, valid question JSON.
     - **Expected Output**: List of question-answer pairs in JSON format.
   - **Test Case 3.2**: Test error handling when the LLM returns an invalid JSON.
     - **Input**: Mocked `KnowledgeBase`, invalid JSON response from LLM.
     - **Expected Output**: Proper error handling and logging.

4. **`OutputGenerator.create_dict()`**
   - **Test Case 4.1**: Test correct dictionary creation based on question types.
     - **Input**: A list of questions with different `question_type`.
     - **Expected Output**: Dictionary with keys corresponding to `question_type` and values as lists of questions.

5. **`OutputGenerator.output_data()`**
   - **Test Case 5.1**: Test file output creation with valid data.
     - **Input**: Properly formatted dictionary of questions.
     - **Expected Output**: JSON file written to disk with the correct content.

6. **`Hallucinator.generate_hallucinations()`**
   - **Test Case 6.1**: Test generation of hallucinated answers.
     - **Input**: Valid `question`, `answer`, `num_hallucinated_answers`, `language`, `batch`.
     - **Expected Output**: List of hallucinated answers in JSON format.

---

## 2. **Integration Testing Plan**

Integration tests focus on ensuring that the components in the `QuestionGenerator` system work together as expected when combined.

### **2.1. Integration Test Objectives**

- Validate that the `QuestionGenerator` workflow (from generating questions to producing output files) works seamlessly.
- Ensure that data flows correctly between components such as `LLM`, `Agent`, `KnowledgeBase`, `OutputGenerator`, and `Hallucinator`.
- Detect any integration issues that might arise from the interaction between different modules.

### **2.2. Integration Test Cases**

1. **`QuestionGenerator` and `Agent` Integration**
   - **Test Case 1.1**: Test the full cycle of generating questions and answers for different `question_types`.
     - **Input**: Mocked `KnowledgeBase`, various `question_types`, `num_questions`, `language`.
     - **Expected Output**: Valid JSON containing questions and answers, with correct context associations.

2. **`QuestionGenerator` and `OutputGenerator` Integration**
   - **Test Case 2.1**: Test the process of generating questions and answers and writing them to a file.
     - **Input**: Mocked `KnowledgeBase`, various `question_types`, `num_questions`, `language`.
     - **Expected Output**: Output JSON file with correctly structured question-answer pairs.

3. **`QuestionGenerator`, `OutputGenerator`, and `Hallucinator` Integration**
   - **Test Case 3.1**: Test the complete process from question generation to hallucination generation.
     - **Input**: Mocked `KnowledgeBase`, various `question_types`, `num_questions`, `language`.
     - **Expected Output**: Final output JSON file containing original questions, answers, and hallucinated answers.

4. **End-to-End Workflow**
   - **Test Case 4.1**: Simulate a full end-to-end scenario where a user provides input to generate questions, answers, and hallucinations.
     - **Input**: Mocked `KnowledgeBase`, various `question_types`, `num_questions`, `language`, batch processing.
     - **Expected Output**: Comprehensive output JSON file containing all stages (questions, answers, hallucinations) with valid data.

5. **Error Handling in Integration**
   - **Test Case 5.1**: Simulate failures in one component (e.g., LLM returning invalid JSON) and ensure the system handles the failure gracefully.
     - **Input**: Mocked components with injected faults (e.g., invalid JSON).
     - **Expected Output**: The system logs errors and skips or retries operations as expected without crashing.

---

## 3. **Tools and Environment Setup**

- **Testing Framework**: Use `unittest` for both unit and integration tests.
- **Mocking**: Utilize `unittest.mock` to mock dependencies like `LLM`, API calls, and file operations.
- **Test Data**: Prepare test documents and mock data for the `KnowledgeBase`. To be stored in `../test_docs` (parent directory in which package is installed)
- **Continuous Integration (CI)**: Set up a CI pipeline (e.g., GitHub Actions, Jenkins) to run these tests automatically on each commit.

---

## 4. **Execution and Reporting**

- **Execution**: Run unit tests first to ensure individual components are functioning, followed by integration tests.
- **Reporting**: Collect test results, highlighting any failed cases, and analyze logs for debugging.
- **Coverage**: Use a coverage tool (e.g., `coverage.py`) to measure test coverage and identify untested parts of the code.

---

This plan ensures that the `QuestionGenerator` and its related components are thoroughly tested both in isolation and in concert, providing confidence in the overall system's reliability.
