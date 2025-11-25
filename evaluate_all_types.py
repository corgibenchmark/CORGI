#!/usr/bin/env python3
"""
Unified Evaluation Script for All Question Types
Evaluates answers for type2, type3, and type4 questions using GPT-5, Gemini 2.5 Pro, and Llama 4 as evaluators.

Evaluators: gpt5, gemini2.5pro, llama4
Answer Models: gpt5, gemini2.5pro, llama4, qwen3
Datasets: shopify, appstore, clothing
Question Types: type2, type3, type4

Generates evaluation files for each combination (3 evaluators × 4 answer models = 12 files per dataset/type)
Skips existing files to allow resuming interrupted evaluations.

Supports multiprocessing for type4 evaluations to speed up processing.

Usage: python3 evaluate_all_types.py [question_type] [dataset] [--multiprocess]
Example: python3 evaluate_all_types.py type3 shopify
Example: python3 evaluate_all_types.py type4 clothing --multiprocess
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

# Add paths
sys.path.insert(0, '/Users/liyue/Desktop/rebuttal')
project_root = "/Users/liyue/Desktop/副本"
src_path = os.path.join(project_root, "src")
if os.path.exists(src_path):
    sys.path.append(src_path)

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Set API key
api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    api_key = "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"
    os.environ['OPENROUTER_API_KEY'] = api_key
    print("✅ Using API key from environment")

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
    "gemini2.5pro": {
        "model_id": "google/gemini-2.5-pro",
        "model_name": "Gemini 2.5 Pro"
    },
    "llama4": {
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "model_name": "Llama 4"
    }
}

# Answer models and evaluator models
ANSWER_MODELS = ["gpt5", "gemini2.5pro", "llama4", "qwen3"]
EVALUATOR_MODELS = ["gpt5", "gemini2.5pro", "llama4"]
SUPPORTED_DATASETS = ["shopify", "appstore", "clothing"]
SUPPORTED_QUESTION_TYPES = ["type2", "type3", "type4"]


class OpenRouterGenerator:
    """OpenRouter API Generator supporting multiple models."""
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
    """Initialize generator for evaluation."""
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
        # Fallback to GCP generator if needed
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
    print("\nLoading notebook code...")
    
    with open('agent-eval-gemini2.5.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Cell 1: Imports (with optional text2sql)
    for i, cell in enumerate(nb['cells']):
        if i == 1 and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            source = source.replace(
                'from text2sql.engine.generation import AzureGenerator, GCPGenerator\nfrom text2sql.engine.generation.postprocessing import extract_first_code_block',
                'try:\n    from text2sql.engine.generation import AzureGenerator, GCPGenerator\n    from text2sql.engine.generation.postprocessing import extract_first_code_block\nexcept ImportError:\n    print("Note: text2sql module not found. Using OpenRouterGenerator only.")\n    AzureGenerator = None\n    GCPGenerator = None\n    extract_first_code_block = None'
            )
            exec(source, globals())
            print(f"✅ Loaded Cell 1 (Imports)")
    
    # Cell 4: METRIC_CATEGORIES and METRIC_DEFINITIONS
    for i, cell in enumerate(nb['cells']):
        if i == 4 and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            exec(source, globals())
            print(f"✅ Loaded Cell 4 (Metric Definitions)")
    
    # Cell 5: Evaluation criteria functions
    for i, cell in enumerate(nb['cells']):
        if i == 5 and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            exec(source, globals())
            print(f"✅ Loaded Cell 5 (Evaluation Criteria)")
    
    # Cell 7: DiscriminatorAgent
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
            print(f"✅ Loaded Cell 7 (DiscriminatorAgent)")
    
    # Cell 8: CategoryScoringAgent
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
            print(f"✅ Loaded Cell 8 (CategoryScoringAgent)")
    
    # Cell 9: ComprehensiveEvaluator
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
            print(f"✅ Loaded Cell 9 (ComprehensiveEvaluator)")


def create_scoring_agents_with_generator(generator):
    """Create scoring agents with a specific generator instance."""
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
    """Get the path to the answer file based on question type, dataset, and model."""
    if question_type == "type2":
        return f"answers/answers_type2/{dataset}_type2_answers/{answer_model}/{answer_model}_type2_answers.csv"
    elif question_type == "type3":
        return f"answers/answers_type3/{dataset}_type3_answers/type3/{answer_model}/{answer_model}_type3_answers.csv"
    elif question_type == "type4":
        return f"answers/answers_type4/{dataset}_type4_answers/type4/{answer_model}/{answer_model}_type4_answers.csv"
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


def get_answer_column(df: pd.DataFrame, question_type: str) -> Optional[str]:
    """Get the answer column name from the dataframe."""
    # Try different possible column names
    possible_columns = ['Answers Generated', 'Answer', 'General Answers']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    return None


def evaluate_single_question(args):
    """
    Worker function to evaluate a single question (for multiprocessing).
    This function is called by multiprocessing workers.
    """
    (question_data, evaluator_model, question_type, dataset, answer_model) = args
    
    question_number = question_data['question_number']
    question = question_data['question']
    answer = question_data['answer']
    
    try:
        # Initialize evaluator in this worker process
        eval_generator = initialize_generator(model_key=evaluator_model, use_openrouter=True)
        eval_discriminator = DiscriminatorAgent(eval_generator)
        eval_scoring_agents = create_scoring_agents_with_generator(eval_generator)
        eval_evaluator = ComprehensiveEvaluator(eval_discriminator, eval_scoring_agents)
        
        # Evaluate the question
        evaluation_results = eval_evaluator.evaluate_question_answer(
            question, answer, question_type
        )
        
        # Flatten results
        flattened_result = {
            'question_number': question_number,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'evaluator_model': evaluator_model,
            'answer_model': answer_model,
            'dataset': dataset
        }
        
        # Add each metric's score and reasoning
        for category, metrics in evaluation_results.items():
            # Calculate category average score
            category_scores = [result['score'] for result in metrics.values()]
            category_avg_score = sum(category_scores) / len(category_scores) if category_scores else 0
            
            # Add category average score
            category_key = category.replace(' ', '_').replace('-', '_')
            flattened_result[f'{category_key}_score'] = round(category_avg_score, 2)
            
            # Add individual metric scores and reasoning
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


def evaluate_answers(
    question_type: str,
    dataset: str,
    use_multiprocessing: bool = False
) -> Dict:
    """
    Evaluate answers for a specific question type and dataset.
    
    Args:
        question_type: Type of questions (type2, type3, type4)
        dataset: Dataset name (shopify, appstore, clothing)
        use_multiprocessing: Whether to use multiprocessing (recommended for type4)
    
    Returns:
        Dictionary containing evaluation results
    """
    print("=" * 80)
    print(f"Evaluating {dataset.upper()} {question_type.upper()} Answers")
    print("=" * 80)
    
    # Check which files already exist
    print("\nChecking existing evaluation files...")
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
                        print(f"  ✅ {filename} (exists, {len(df)} questions)")
                except:
                    print(f"  ⚠️  {filename} (exists but may be corrupted)")
    
    # Determine which combinations need evaluation
    all_combinations = [(am, em) for am in ANSWER_MODELS for em in EVALUATOR_MODELS]
    remaining_combinations = [combo for combo in all_combinations if combo not in existing_files]
    
    print(f"\nCombinations to evaluate: {len(remaining_combinations)}")
    for answer_model, evaluator_model in remaining_combinations:
        filename = f"{dataset}_{question_type}_{answer_model}_{evaluator_model}.csv"
        print(f"  - {filename}")
    
    if not remaining_combinations:
        print("\n✅ All evaluation files already exist, no evaluation needed!")
        return {}
    
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)
    
    all_evaluation_results = {}
    total_remaining = len(remaining_combinations)
    completed = 0
    
    # Group by answer model
    answer_model_groups = {}
    for answer_model, evaluator_model in remaining_combinations:
        if answer_model not in answer_model_groups:
            answer_model_groups[answer_model] = []
        answer_model_groups[answer_model].append(evaluator_model)
    
    # Create overall progress bar if using multiprocessing
    overall_pbar = None
    if use_multiprocessing:
        overall_pbar = tqdm(total=total_remaining, desc="Overall Progress", unit="file", position=0, leave=True, file=sys.stdout)
    
    # Process each answer model
    for answer_model in ANSWER_MODELS:
        if answer_model not in answer_model_groups:
            continue
        
        if overall_pbar:
            overall_pbar.write(f"\n{'='*80}")
            overall_pbar.write(f"Processing Answer Model: {answer_model.upper()}")
            overall_pbar.write(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"Processing Answer Model: {answer_model.upper()}")
            print(f"{'='*80}")
        
        # Get answer file path
        answer_file = get_answer_file_path(question_type, dataset, answer_model)
        
        if not os.path.exists(answer_file):
            msg = f"⚠️  Answer file does not exist: {answer_file}"
            if overall_pbar:
                overall_pbar.write(msg)
            else:
                print(msg)
            continue
        
        # Load answers
        df = pd.read_csv(answer_file)
        msg = f"✅ Loaded {len(df)} questions from {answer_file}"
        if overall_pbar:
            overall_pbar.write(msg)
        else:
            print(msg)
        
        # Check answer column
        answer_column = get_answer_column(df, question_type)
        if answer_column is None:
            msg = f"⚠️  Could not find answer column in {answer_file}"
            if overall_pbar:
                overall_pbar.write(msg)
                overall_pbar.write(f"    Available columns: {list(df.columns)}")
            else:
                print(msg)
                print(f"    Available columns: {list(df.columns)}")
            continue
        
        all_evaluation_results[answer_model] = {}
        
        # Process each evaluator model
        for evaluator_model in answer_model_groups[answer_model]:
            completed += 1
            filename = f"{dataset}_{question_type}_{answer_model}_{evaluator_model}.csv"
            
            if overall_pbar:
                overall_pbar.set_description(f"Overall Progress [{completed}/{total_remaining}]")
                overall_pbar.set_postfix_str(f"Current: {filename}")
            
            try:
                if use_multiprocessing:
                    # Prepare evaluation tasks
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
                    
                    overall_pbar.write(f"  [Step 1/4] Preparing evaluator configuration: {evaluator_model.upper()}...")
                    overall_pbar.write(f"  [Step 2/4] Evaluator will be initialized in worker processes...")
                    overall_pbar.write(f"  ✅ Evaluator configuration complete (will be initialized in each worker process)")
                    
                    overall_pbar.write(f"  [Step 3/4] Preparing to evaluate {len(evaluation_tasks)} questions...")
                    num_workers = min(cpu_count(), len(evaluation_tasks), MAX_WORKERS)
                    overall_pbar.write(f"     Using multiprocessing parallel evaluation (processes: {num_workers})...")
                    
                    # Use multiprocessing
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
                                    overall_pbar.write(f"  ❌ Error evaluating question {result.get('question_number', 'unknown')}: {result.get('error', 'unknown error')}")
                                
                                question_pbar.update(1)
                    
                    # Sort results by question number
                    all_results.sort(key=lambda x: x['question_number'])
                    
                    # Report errors
                    if errors:
                        overall_pbar.write(f"  ⚠️  {len(errors)} questions failed evaluation")
                        for err in errors:
                            overall_pbar.write(f"    Question {err['question_number']}: {err['error']}")
                    
                    # Save results
                    overall_pbar.write(f"  [Step 4/4] Saving evaluation results...")
                    if all_results:
                        df_results = pd.DataFrame(all_results)
                        output_filename = os.path.join(EVALUATION_FOLDER, filename)
                        overall_pbar.write(f"    → Writing to file: {output_filename}...")
                        df_results.to_csv(output_filename, index=False)
                        overall_pbar.write(f"    ✅ File saved successfully")
                        
                        all_evaluation_results[answer_model][evaluator_model] = df_results
                        
                        overall_pbar.write(f"✅ Saved: {filename} ({len(all_results)} questions)")
                        
                        # Print summary
                        score_cols = [col for col in df_results.columns if col.endswith('_score')]
                        category_cols = [col for col in score_cols if any(cat in col for cat in ['Structure', 'Factuality', 'Data_Sense', 'Insightfulness', 'Operational_Implementability', 'Purpose_Alignment', 'Compliance']) and col.count('_') <= 2]
                        overall_pbar.write(f"\n  Category Average Scores:")
                        for col in sorted(category_cols):
                            metric_name = col.replace('_score', '').replace('_', ' ').title()
                            avg_score = df_results[col].mean()
                            overall_pbar.write(f"    {metric_name:40s}: {avg_score:.2f}/5")
                    
                    overall_pbar.update(1)
                    
                else:
                    # Sequential evaluation
                    msg = f"\n{'='*80}"
                    msg += f"\n[{completed}/{total_remaining}] Evaluator: {evaluator_model.upper()} | Answer Model: {answer_model.upper()}"
                    msg += f"\n{'='*80}"
                    print(msg)
                    
                    # Initialize evaluator
                    print(f"  Initializing evaluator: {evaluator_model.upper()}...")
                    eval_generator = initialize_generator(model_key=evaluator_model, use_openrouter=True)
                    
                    # Create evaluator
                    eval_discriminator = DiscriminatorAgent(eval_generator)
                    eval_scoring_agents = create_scoring_agents_with_generator(eval_generator)
                    eval_evaluator = ComprehensiveEvaluator(eval_discriminator, eval_scoring_agents)
                    print(f"  ✅ Evaluator initialization complete")
                    
                    # Store results
                    all_results = []
                    
                    # Evaluate each question
                    for idx, row in df.iterrows():
                        question_number = row.get('Question Number', idx + 1)
                        question = row['Question']
                        answer = row[answer_column]
                        
                        print(f"\n  Question {question_number}/{len(df)}: {question[:70]}...")
                        
                        try:
                            # Evaluate
                            evaluation_results = eval_evaluator.evaluate_question_answer(
                                question, answer, question_type
                            )
                            
                            # Flatten results
                            flattened_result = {
                                'question_number': question_number,
                                'question': question,
                                'answer': answer,
                                'question_type': question_type,
                                'evaluator_model': evaluator_model,
                                'answer_model': answer_model,
                                'dataset': dataset
                            }
                            
                            # Add each metric's score and reasoning
                            for category, metrics in evaluation_results.items():
                                # Calculate category average score
                                category_scores = [result['score'] for result in metrics.values()]
                                category_avg_score = sum(category_scores) / len(category_scores) if category_scores else 0
                                
                                # Add category average score
                                category_key = category.replace(' ', '_').replace('-', '_')
                                flattened_result[f'{category_key}_score'] = round(category_avg_score, 2)
                                
                                # Add individual metric scores and reasoning
                                for metric, result in metrics.items():
                                    metric_key = metric.replace(' ', '_').replace('-', '_')
                                    flattened_result[f'{metric_key}_score'] = result['score']
                                    flattened_result[f'{metric_key}_reasoning'] = result['reasoning']
                            
                            all_results.append(flattened_result)
                            print(f"  ✅ Question {question_number} evaluation complete")
                            
                            # Delay to avoid rate limiting
                            if idx < len(df) - 1:
                                time.sleep(2)
                                
                        except Exception as e:
                            print(f"  ❌ Error evaluating question {question_number}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # Save results
                    if all_results:
                        df_results = pd.DataFrame(all_results)
                        output_filename = os.path.join(
                            EVALUATION_FOLDER,
                            filename
                        )
                        df_results.to_csv(output_filename, index=False)
                        
                        all_evaluation_results[answer_model][evaluator_model] = df_results
                        
                        print(f"\n  ✅ Saved: {output_filename}")
                        print(f"  ✅ Evaluated {len(all_results)} questions")
                        
                        # Print summary
                        score_cols = [col for col in df_results.columns if col.endswith('_score')]
                        category_cols = [col for col in score_cols if any(cat in col for cat in ['Structure', 'Factuality', 'Data_Sense', 'Insightfulness', 'Operational_Implementability', 'Purpose_Alignment', 'Compliance']) and col.count('_') <= 2]
                        print(f"\n  Category Average Scores:")
                        for col in sorted(category_cols):
                            metric_name = col.replace('_score', '').replace('_', ' ').title()
                            avg_score = df_results[col].mean()
                            print(f"    {metric_name:40s}: {avg_score:.2f}/5")
                
            except Exception as e:
                msg = f"  ❌ Error using {evaluator_model} evaluator: {e}"
                if overall_pbar:
                    overall_pbar.write(msg)
                else:
                    print(msg)
                import traceback
                traceback.print_exc()
                continue
        
        # Delay between answer models
        if answer_model != ANSWER_MODELS[-1] or any(am in answer_model_groups for am in ANSWER_MODELS[ANSWER_MODELS.index(answer_model)+1:]):
            msg = f"\nWaiting 5 seconds before processing next answer model..."
            if overall_pbar:
                overall_pbar.write(msg)
            else:
                print(msg)
            time.sleep(5)
    
    if overall_pbar:
        overall_pbar.write(f"\n{'='*80}")
        overall_pbar.write("Evaluation complete!")
        overall_pbar.write(f"{'='*80}")
        overall_pbar.close()
    else:
        print(f"\n{'='*80}")
        print("Evaluation complete!")
        print(f"{'='*80}")
    
    # Final check
    msg = "\nFinal file status:"
    if overall_pbar:
        overall_pbar.write(msg)
    else:
        print(msg)
    
    all_files = []
    for answer_model in ANSWER_MODELS:
        for evaluator_model in EVALUATOR_MODELS:
            filename = f"{dataset}_{question_type}_{answer_model}_{evaluator_model}.csv"
            filepath = os.path.join(EVALUATION_FOLDER, filename)
            if os.path.exists(filepath):
                try:
                    df_check = pd.read_csv(filepath)
                    all_files.append((filename, len(df_check)))
                    msg = f"  ✅ {filename} ({len(df_check)} questions)"
                    if overall_pbar:
                        overall_pbar.write(msg)
                    else:
                        print(msg)
                except:
                    msg = f"  ⚠️  {filename} (exists but cannot be read)"
                    if overall_pbar:
                        overall_pbar.write(msg)
                    else:
                        print(msg)
            else:
                msg = f"  ❌ {filename} (not generated)"
                if overall_pbar:
                    overall_pbar.write(msg)
                else:
                    print(msg)
    
    msg = f"\nTotal: {len(all_files)}/12 evaluation files"
    if overall_pbar:
        overall_pbar.write(msg)
    else:
        print(msg)
    
    return all_evaluation_results


def main():
    """Main function to parse arguments and run evaluation."""
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
    
    # Load notebook code
    load_notebook_code()
    
    print("✅ Defined create_scoring_agents_with_generator function")
    print(f"✅ Evaluation folder: {EVALUATION_FOLDER}/")
    
    # Run evaluation
    print(f"\nStarting evaluation of {args.dataset} {args.question_type} answers...")
    print("Skipping existing files, only evaluating missing files\n")
    
    results = evaluate_answers(
        question_type=args.question_type,
        dataset=args.dataset,
        use_multiprocessing=args.multiprocess
    )
    
    print("\n" + "=" * 80)
    print("Task complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


