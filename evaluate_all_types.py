#!/usr/bin/env python3
"""
Evaluation script for type2, type3, and type4 questions.

Usage: python3 evaluate_all_types.py [question_type] [dataset] [--multiprocess]
"""
import os
import sys
import json
import pandas as pd
import requests
import time
import re
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# Evaluation folder
EVALUATION_FOLDER = "evaluation"
os.makedirs(EVALUATION_FOLDER, exist_ok=True)

# Multiprocessing configuration
MAX_WORKERS = 32  # Maximum parallel processes for type4 evaluations

# Model configuration for OpenRouter
MODEL_CONFIGS = {
    "gpt5": {
        "model_id": "openai/gpt-5",
        "model_name": "GPT-5"
    },
    "gpt4o": {
        "model_id": "openai/gpt-4o",
        "model_name": "GPT-4o"
    },
    "gemini2.5pro": {
        "model_id": "google/gemini-2.5-pro",
        "model_name": "Gemini 2.5 Pro"
    },
    "gemini-2.5-flash-lite": {
        "model_id": "google/gemini-2.0-flash-exp",
        "model_name": "Gemini 2.5 Flash Lite"
    },
    "gemini-2.0-flash-lite": {
        "model_id": "google/gemini-2.0-flash-exp",
        "model_name": "Gemini 2.0 Flash Lite"
    },
    "llama4": {
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "model_name": "Llama 4"
    }
}

ANSWER_MODELS = ["gpt5", "gpt4o", "gemini2.5pro", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "llama4", "qwen3"]
EVALUATOR_MODELS = ["gpt5", "gpt4o", "gemini2.5pro", "gemini-2.5-flash-lite", "llama4"]
SUPPORTED_DATASETS = ["shopify", "appstore", "clothing"]
SUPPORTED_QUESTION_TYPES = ["type2", "type3", "type4"]


class OpenRouterGenerator:
    def __init__(self, api_key, model_id, model_name):
        self.api_key = api_key
        self.model_id = model_id
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate(self, messages, max_retries=3, temperature=None, max_tokens=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Evaluation System"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=180)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        return content.strip()
                    else:
                        raise Exception(f"No choices in response: {result}")
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise Exception(error_msg)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise e
        
        raise Exception("All retry attempts failed")


def initialize_generator(model_key="gpt5", use_openrouter=True):
    if use_openrouter:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_key]
        generator = OpenRouterGenerator(
            api_key=api_key,
            model_id=config["model_id"],
            model_name=config["model_name"]
        )
        return generator
    else:
        try:
            from text2sql.engine.generation import GCPGenerator
            api_key = os.environ.get("GCP_KEY")
            if not api_key:
                raise ValueError("GCP_KEY not found in environment variables")
            generator = GCPGenerator(api_key=api_key, model="gemini-2.5-flash-lite")
            return generator
        except ImportError:
            raise ValueError("GCPGenerator not available. Please use OpenRouter.")


def load_notebook_code():
    """Load evaluation code from the notebook."""
    notebook_path = 'agent_eval.ipynb'
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if i == 1 and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            source = source.replace(
                'from text2sql.engine.generation import AzureGenerator, GCPGenerator\nfrom text2sql.engine.generation.postprocessing import extract_first_code_block',
                'try:\n    from text2sql.engine.generation import AzureGenerator, GCPGenerator\n    from text2sql.engine.generation.postprocessing import extract_first_code_block\nexcept ImportError:\n    AzureGenerator = None\n    GCPGenerator = None\n    extract_first_code_block = None'
            )
            exec(source, globals())
    
    for i, cell in enumerate(nb['cells']):
        if i == 4 and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            exec(source, globals())
    
    for i, cell in enumerate(nb['cells']):
        if i == 5 and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            exec(source, globals())
    
    for i, cell in enumerate(nb['cells']):
        if i == 7 and cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_source_lines = []
            for line in source_lines:
                if 'discriminator = DiscriminatorAgent(generator)' in line:
                    continue
                else:
                    new_source_lines.append(line)
            source = ''.join(new_source_lines)
            generator = type('DummyGenerator', (), {})()
            try:
                exec(source, globals())
            finally:
                if 'generator' in globals() and isinstance(globals()['generator'], type('DummyGenerator', (), {})):
                    del globals()['generator']
    
    for i, cell in enumerate(nb['cells']):
        if i == 8 and cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_source_lines = []
            for line in source_lines:
                if 'scoring_agents = create_scoring_agents()' in line:
                    continue
                elif 'comprehensive_evaluator = ComprehensiveEvaluator(discriminator, scoring_agents)' in line:
                    continue
                else:
                    new_source_lines.append(line)
            source = ''.join(new_source_lines)
            exec(source, globals())
    
    for i, cell in enumerate(nb['cells']):
        if i == 9 and cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_source_lines = []
            for line in source_lines:
                if 'comprehensive_evaluator = ComprehensiveEvaluator(discriminator, scoring_agents)' in line:
                    continue
                else:
                    new_source_lines.append(line)
            source = ''.join(new_source_lines)
            exec(source, globals())


def create_scoring_agents_with_generator(generator):
    agents = {}
    
    agents["Structure"] = CategoryScoringAgent(
        "Structure", 
        ["Argument Soundness", "Logical Coherence", "Verbosity"], 
        generator,
    )
    
    agents["Factuality"] = CategoryScoringAgent(
        "Factuality", 
        ["External Information Accuracy"],
        generator,
    )
    
    agents["Data Sense"] = CategoryScoringAgent(
        "Data Sense", 
        ["Information Adequacy", "Trend Awareness", "Model Selection rationale"], 
        generator,
    )
    
    agents["Insightfulness"] = CategoryScoringAgent(
        "Insightfulness", 
        ["Out-of-the-box Thinking", "Root Cause Depth", "Assumption Appropriateness"], 
        generator,
    )
    
    agents["Operational Implementability"] = CategoryScoringAgent(
        "Operational Implementability", 
        ["Actionability", "Time-Based Planning"], 
        generator,
    )
    
    agents["Purpose Alignment"] = CategoryScoringAgent(
        "Purpose Alignment", 
        ["Goal Orientation", "Stakeholder Orientation"], 
        generator,
    )
    
    agents["Compliance"] = CategoryScoringAgent(
        "Compliance", 
        ["Risk Management", "Regulatory Compliance", "Ethical Responsibility"], 
        generator,
    )
    
    return agents


def get_answer_file_path(question_type: str, dataset: str, answer_model: str) -> str:
    if question_type == "type2":
        return f"answers/answers_type2/{dataset}_type2_answers/{answer_model}/{answer_model}_type2_answers.csv"
    elif question_type == "type3":
        return f"answers/answers_type3/{dataset}_type3_answers/type3/{answer_model}/{answer_model}_type3_answers.csv"
    elif question_type == "type4":
        return f"answers/answers_type4/{dataset}_type4_answers/type4/{answer_model}/{answer_model}_type4_answers.csv"
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


def get_answer_column(df: pd.DataFrame, question_type: str) -> Optional[str]:
    possible_columns = ['Answers Generated', 'Answer', 'General Answers']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    return None


def evaluate_single_question(args):
    (question_data, evaluator_model, question_type, dataset, answer_model) = args
    
    question_number = question_data['question_number']
    question = question_data['question']
    answer = question_data['answer']
    
    try:
        eval_generator = initialize_generator(model_key=evaluator_model, use_openrouter=True)
        eval_discriminator = DiscriminatorAgent(eval_generator)
        eval_scoring_agents = create_scoring_agents_with_generator(eval_generator)
        eval_evaluator = ComprehensiveEvaluator(eval_discriminator, eval_scoring_agents)
        
        evaluation_results = eval_evaluator.evaluate_question_answer(
            question, answer, question_type
        )
        
        flattened_result = {
            'question_number': question_number,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'evaluator_model': evaluator_model,
            'answer_model': answer_model,
            'dataset': dataset
        }
        
        for category, metrics in evaluation_results.items():
            category_scores = [result['score'] for result in metrics.values()]
            category_avg_score = sum(category_scores) / len(category_scores) if category_scores else 0
            
            category_key = category.replace(' ', '_').replace('-', '_')
            flattened_result[f'{category_key}_score'] = round(category_avg_score, 2)
            
            for metric, result in metrics.items():
                metric_key = metric.replace(' ', '_').replace('-', '_')
                flattened_result[f'{metric_key}_score'] = result['score']
                flattened_result[f'{metric_key}_reasoning'] = result['reasoning']
        
        return {'success': True, 'result': flattened_result}
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'question_number': question_number,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def evaluate_answers(question_type: str, dataset: str, use_multiprocessing: bool = False) -> Dict:
    existing_files = set()
    for answer_model in ANSWER_MODELS:
        for evaluator_model in EVALUATOR_MODELS:
            filename = f"{dataset}_{question_type}_{answer_model}_{evaluator_model}.csv"
            filepath = os.path.join(EVALUATION_FOLDER, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 0:
                        existing_files.add((answer_model, evaluator_model))
                except:
                    pass
    
    all_combinations = [(am, em) for am in ANSWER_MODELS for em in EVALUATOR_MODELS]
    remaining_combinations = [combo for combo in all_combinations if combo not in existing_files]
    
    if not remaining_combinations:
        return {}
    
    all_evaluation_results = {}
    total_remaining = len(remaining_combinations)
    completed = 0
    
    answer_model_groups = {}
    for answer_model, evaluator_model in remaining_combinations:
        if answer_model not in answer_model_groups:
            answer_model_groups[answer_model] = []
        answer_model_groups[answer_model].append(evaluator_model)
    
    overall_pbar = None
    if use_multiprocessing:
        overall_pbar = tqdm(total=total_remaining, desc="Overall Progress", unit="file", position=0, leave=True, file=sys.stdout)
    
    for answer_model in ANSWER_MODELS:
        if answer_model not in answer_model_groups:
            continue
        
        answer_file = get_answer_file_path(question_type, dataset, answer_model)
        
        if not os.path.exists(answer_file):
            continue
        
        df = pd.read_csv(answer_file)
        answer_column = get_answer_column(df, question_type)
        if answer_column is None:
            continue
        
        all_evaluation_results[answer_model] = {}
        
        for evaluator_model in answer_model_groups[answer_model]:
            completed += 1
            filename = f"{dataset}_{question_type}_{answer_model}_{evaluator_model}.csv"
            
            if overall_pbar:
                overall_pbar.set_description(f"Overall Progress [{completed}/{total_remaining}]")
                overall_pbar.set_postfix_str(f"Current: {filename}")
            
            try:
                if use_multiprocessing:
                    evaluation_tasks = []
                    for idx, row in df.iterrows():
                        question_number = row.get('Question Number', idx + 1)
                        question = row['Question']
                        answer = row[answer_column]
                        
                        question_data = {
                            'question_number': question_number,
                            'question': question,
                            'answer': answer
                        }
                        
                        task_args = (
                            question_data,
                            evaluator_model,
                            question_type,
                            dataset,
                            answer_model
                        )
                        evaluation_tasks.append(task_args)
                    
                    num_workers = min(cpu_count(), len(evaluation_tasks), MAX_WORKERS)
                    
                    all_results = []
                    errors = []
                    
                    with Pool(processes=num_workers) as pool:
                        with tqdm(total=len(evaluation_tasks), desc=f"  {filename[:40]:<40}", unit="question", position=1, leave=False) as question_pbar:
                            for result in pool.imap_unordered(evaluate_single_question, evaluation_tasks):
                                if result['success']:
                                    all_results.append(result['result'])
                                else:
                                    errors.append({
                                        'question_number': result.get('question_number', 'unknown'),
                                        'error': result.get('error', 'unknown error'),
                                        'traceback': result.get('traceback', '')
                                    })
                                
                                question_pbar.update(1)
                    
                    all_results.sort(key=lambda x: x['question_number'])
                    
                    if all_results:
                        df_results = pd.DataFrame(all_results)
                        output_filename = os.path.join(EVALUATION_FOLDER, filename)
                        df_results.to_csv(output_filename, index=False)
                        all_evaluation_results[answer_model][evaluator_model] = df_results
                    
                    overall_pbar.update(1)
                    
                else:
                    eval_generator = initialize_generator(model_key=evaluator_model, use_openrouter=True)
                    eval_discriminator = DiscriminatorAgent(eval_generator)
                    eval_scoring_agents = create_scoring_agents_with_generator(eval_generator)
                    eval_evaluator = ComprehensiveEvaluator(eval_discriminator, eval_scoring_agents)
                    
                    all_results = []
                    
                    for idx, row in df.iterrows():
                        question_number = row.get('Question Number', idx + 1)
                        question = row['Question']
                        answer = row[answer_column]
                        
                        try:
                            evaluation_results = eval_evaluator.evaluate_question_answer(
                                question, answer, question_type
                            )
                            
                            flattened_result = {
                                'question_number': question_number,
                                'question': question,
                                'answer': answer,
                                'question_type': question_type,
                                'evaluator_model': evaluator_model,
                                'answer_model': answer_model,
                                'dataset': dataset
                            }
                            
                            for category, metrics in evaluation_results.items():
                                category_scores = [result['score'] for result in metrics.values()]
                                category_avg_score = sum(category_scores) / len(category_scores) if category_scores else 0
                                
                                category_key = category.replace(' ', '_').replace('-', '_')
                                flattened_result[f'{category_key}_score'] = round(category_avg_score, 2)
                                
                                for metric, result in metrics.items():
                                    metric_key = metric.replace(' ', '_').replace('-', '_')
                                    flattened_result[f'{metric_key}_score'] = result['score']
                                    flattened_result[f'{metric_key}_reasoning'] = result['reasoning']
                            
                            all_results.append(flattened_result)
                            
                            if idx < len(df) - 1:
                                time.sleep(2)
                                
                        except Exception as e:
                            continue
                    
                    if all_results:
                        df_results = pd.DataFrame(all_results)
                        output_filename = os.path.join(EVALUATION_FOLDER, filename)
                        df_results.to_csv(output_filename, index=False)
                        all_evaluation_results[answer_model][evaluator_model] = df_results
                
            except Exception as e:
                continue
        
        if answer_model != ANSWER_MODELS[-1] or any(am in answer_model_groups for am in ANSWER_MODELS[ANSWER_MODELS.index(answer_model)+1:]):
            time.sleep(5)
    
    if overall_pbar:
        overall_pbar.close()
    
    return all_evaluation_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate answers for all question types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 evaluate_all_types.py type2 shopify
  python3 evaluate_all_types.py type3 appstore
  python3 evaluate_all_types.py type4 clothing --multiprocess
        """
    )
    
    parser.add_argument(
        'question_type',
        type=str,
        choices=SUPPORTED_QUESTION_TYPES,
        help='Question type to evaluate (type2, type3, type4)'
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        choices=SUPPORTED_DATASETS,
        help='Dataset to evaluate (shopify, appstore, clothing)'
    )
    
    parser.add_argument(
        '--multiprocess',
        action='store_true',
        help='Use multiprocessing for faster evaluation (recommended for type4)'
    )
    
    args = parser.parse_args()
    
    load_notebook_code()
    
    results = evaluate_answers(
        question_type=args.question_type,
        dataset=args.dataset,
        use_multiprocessing=args.multiprocess
    )


if __name__ == "__main__":
    main()


