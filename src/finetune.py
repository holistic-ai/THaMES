from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import csv
import time
import torch
import transformers
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel # type: ignore
from huggingface_hub import interpreter_login
from tqdm import tqdm
import random



def load_and_prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset

def initialize_model_and_tokenizer(model_name):
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map=device_map,
        device='mps',
        # device_map="cpu",
        trust_remote_code=True,
        token=True,
        is_trainable=True
    ).to('mps')


    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left',
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return model, tokenizer

def evaluate_llm_hallucination(question, answer, model, tokenizer):
    prompt = instruction + f"\n\n#Question#: {question}\n#Answer#: {answer}\n#Your Judgement#:"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs.input_ids, max_length=inputs.input_ids.shape[1] + 3, num_return_sequences=1,
                             pad_token_id=tokenizer.eos_token_id)
    judgement = tokenizer.decode(outputs[0], skip_special_tokens=True).split('#Your Judgement#:')[-1].strip()
    # judgement = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(judgement)
    # Check for 'Yes' or 'No' in the response
    if 'Yes' in judgement:
        judgement = 'Yes'
    elif 'No' in judgement:
        judgement = 'No'
    else:
        raise ValueError(f"Unexpected response: {judgement}. Expected 'Yes' or 'No'.")

    #     print(judgement)
    return judgement

def preprocess_function(examples, tokenizer):
    prompt = [f"{q}" for q in (examples['question'])]
    targets = [f"{a}" for a in (examples['correct_answer'])]
    model_inputs = tokenizer(prompt, max_length=1024, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

def tokenize_datasets(train_dataset, val_dataset, tokenizer):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    return tokenized_train_dataset, tokenized_val_dataset

def configure_and_train_model(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer, model_name):
    # LoRA Configuration
    config = LoraConfig(
        r=8, #Rank
        lora_alpha=16,
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM", # For question answering tasks or 'CAUSAL_LM'
    )

    model.enable_input_require_grads()
    # Apply LoRA configuration
    peft_model = get_peft_model(model, config)

    output_dir = f'model/peft-QA-training-{model_name.replace("/", "-")}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=False,
        use_mps_device=False,
        use_cpu=True
    )

    peft_model.config.use_cache = False

    trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return trainer


def evaluate_qa_pair(peft_model, tokenizer, use_cove, if_finetune, output_dir):
    tokenizer = tokenizer
    model = peft_model

    with open('output/final/filtered_data_small.json', 'r') as file:
        data = json.load(file)

    #     indexed_qa_pairs = []
    metrics = []

    overall_true_positive = 0
    overall_true_negative = 0
    overall_false_positive = 0
    overall_false_negative = 0

    categories = ['simple', 'reasoning', 'multi_context', 'situational', 'distracting', 'double', 'conditionals']
    #     categories = ['simple']
    for category in categories:
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for idx, qa in enumerate(tqdm(data[category], desc=f"Processing {category}")):
            question = qa['question']
            correct_answer = qa['answer']
            distractor = qa['best_distractor']
            context = qa['context']

            if random.random() < 0.5:
                chosen_answer = correct_answer
                is_hallucination = False
            else:
                chosen_answer = distractor
                is_hallucination = True

            llm_response = evaluate_llm_hallucination(question, chosen_answer, model, tokenizer)

            #             check if llm_response is valid
            #             if llm_response not in ['Yes', 'No']:
            #                 raise ValueError(f"Unexpected response: {llm_response}. Expected 'Yes' or 'No'.")

            # if the distractor is selected and llm reply there is hallucination
            if is_hallucination and llm_response == 'Yes':
                true_positive += 1
                overall_true_positive += 1
            # if the distractor is selected and llm reply no hallucination
            elif is_hallucination and llm_response == 'No':
                false_negative += 1
                overall_false_negative += 1
            #                 indexed_qa_pairs.append(indexed_qa_pair)
            # if the ground truth answer is selected and llm reply there is hallucination
            elif not is_hallucination and llm_response == 'Yes':
                false_positive += 1
                overall_false_positive += 1
            #                 indexed_qa_pairs.append(indexed_qa_pair)
            # if the ground truth answer is selected and llm reply no hallucination
            elif not is_hallucination and llm_response == 'No':
                true_negative += 1
                overall_true_negative += 1

        total = true_positive + true_negative + false_positive + false_negative
        accuracy = (true_positive + true_negative) / total if total else 0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

        metrics.append({
            'category': category,
            'model': 'llama 2.1',
            'finetune': if_finetune,
            'use_cove': use_cove,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })

    overall_total = overall_true_positive + overall_true_negative + overall_false_positive + overall_false_negative
    overall_accuracy = (overall_true_positive + overall_true_negative) / overall_total if overall_total else 0
    overall_precision = overall_true_positive / (
            overall_true_positive + overall_false_positive) if overall_true_positive + overall_false_positive else 0
    overall_recall = overall_true_positive / (
            overall_true_positive + overall_false_negative) if overall_true_positive + overall_false_negative else 0
    overall_f1_score = 2 * (overall_precision * overall_recall) / (
            overall_precision + overall_recall) if overall_precision + overall_recall else 0

    metrics.append({
        'category': 'Total',
        'model': 'llama 2.1',
        'finetune': if_finetune,
        'use_cove': use_cove,
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1_score
    })

    #     with open('/kaggle/working/indexed__failed_qa_pairs.json', 'w') as file:
    #         json.dump(indexed_qa_pairs, file, indent=4)
    #     print(f'Indexed QA pairs are saved.')

    with open(output_dir, 'w', newline='') as csvfile:
        fieldnames = ['category', 'model', 'finetune', 'use_cove', 'accuracy', 'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
    print(f'Evaluation metrics are saved.')


if __name__ == "__main__":
    model_name = 'meta-llama/Meta-Llama-3.1-8B'


    instruction = """
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
    """


    # Main execution
    print("Loading dataset ...")
    file_path = 'output/final/indexed_failed_qa_pairs.json'
    train_dataset, val_dataset = load_and_prepare_data(file_path)

    print("Loading model config ...")
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    evaluate_qa_pair(model, tokenizer, False, False, 'output/final/llama2_metrics.csv')
    model_name = 'meta-llama/Meta-Llama-3.1-8B'
    print("Start finetuning")
    tokenized_train_dataset, tokenized_val_dataset = tokenize_datasets(train_dataset, val_dataset, tokenizer)
    trainer = configure_and_train_model(model, tokenized_train_dataset, tokenized_val_dataset, tokenizer, model_name)

    trainer.train()

    print("Finetune finished.")

    # Test after finetuning

    base_model, eval_tokenizer = initialize_model_and_tokenizer(model_name)


    ft_model = PeftModel.from_pretrained(base_model, "model/peft-QA-training-llama2/checkpoint-100",
                                         torch_dtype=torch.float16, is_trainable=False)

    evaluate_qa_pair(ft_model, eval_tokenizer, False, True, 'output/final/lora_llama2_metrics.csv')

    print("Evaluation finished.")