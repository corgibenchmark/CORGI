# Multi-Agent Coordination Protocol


Implementation guide for the multi-agent evaluation system with one discriminator agent and seven category-specific scoring agents.

## System Architecture

The system uses a hierarchical coordination protocol with one discriminator agent and seven category-specific scoring agents:

- Discriminator Agent: Metric selection
- Scoring Agents:
  1. Structure Agent
  2. Factuality Agent
  3. Data Sense Agent
  4. Insightfulness Agent
  5. Operational Implementability Agent
  6. Purpose Alignment Agent
  7. Compliance Agent

## Coordination Protocol: Three-Phase Architecture

### Phase 1: Metric Selection

Agent: `DiscriminatorAgent`

Input:
- `question_type`: One of `["type2", "type3", "type4"]`
- `question`: Question text (string)
- `answer`: Answer text (string)

Output:
- `Dict[str, List[str]]`: Dictionary mapping category names to lists of metric names
  ```python
  {
      "Structure": ["Argument Soundness", "Logical Coherence", "Verbosity"],
      "Factuality": ["External Information Accuracy"],
      "Data Sense": ["Information Adequacy", "Trend Awareness", "Model Selection rationale"],
      ...
  }
  ```

Coordination Protocol:

1. Type2 Questions (Fixed Metric Set):
   - No LLM call required
   - Returns predefined metric set:
     ```python
     {
         "Structure": ["Argument Soundness", "Logical Coherence", "Verbosity"],
         "Factuality": ["External Information Accuracy"],
         "Data Sense": ["Information Adequacy", "Trend Awareness"],
         "Insightfulness": ["Out-of-the-box Thinking", "Root Cause Depth"]
     }
     ```

2. Type3 Questions (Dynamic Metric Selection):
   - Base metrics are predefined
   - Optional metrics determined via LLM binary classification:
     - Model Selection Rationale: LLM determines if question/answer involves numerical prediction
   - LLM call format:
     ```python
     prompt = f"""
     Analyze the following question and answer to determine if the question or answer involves numerical prediction.
     
     Question: {question}
     Answer: {answer}
     
     Does this question or answer involve numerical prediction or modeling that would require model selection rationale? Respond with only "YES" or "NO".
     """
     ```
   - Response parsing: Uppercase "YES" → `True`, otherwise → `False`
   - If `True`: Adds "Model Selection rationale" to "Data Sense" category

3. Type4 Questions (Dynamic Metric Selection):
   - Base metrics include all seven categories
   - Optional metrics determined via LLM binary classification:
     - Model Selection Rationale: Same as type3
   - LLM call format: Identical to type3
   - Response parsing: Same as type3

Tool Use:
- LLM Model: Gemini 2.0 Flash Lite, Gemini 2.5 Flash Lite (via `GCPGenerator`) or OpenRouter models (GPT-5, GPT-4, Gemini 2.5 Pro, Llama 4)
- API Call: Single-shot generation, no retry mechanism
- Error Handling: Returns `False` on API failure
- Temperature: Default (not explicitly set)

Implementation:
```python
def determine_metrics(self, question_type, question, answer):
    if question_type == "type2":
        return fixed_metric_set_type2
    elif question_type == "type3":
        return self.evaluate_type3_metrics(question, answer)
    elif question_type == "type4":
        return self.evaluate_type4_metrics(question, answer)

def get_llm_judgment(prompt: str) -> bool:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = self.generator.generate(messages)
        return response.strip().upper() == "YES"
    except Exception as e:
        print(f"Error getting LLM judgment: {e}")
        return False
```

### Phase 2: Parallel Scoring

Agents: Seven `CategoryScoringAgent` instances, one per category

Input (per metric):
- `question`: Question text (string)
- `answer`: Answer text (string)
- `metric`: Metric name (string)

Output (per metric):
```python
{
    "metric": str,      # Metric name
    "score": int,       # Score from 0-5
    "reasoning": str    # Brief explanation of the score
}
```

Coordination Protocol:

1. Agent Selection:
   - ComprehensiveEvaluator iterates through `selected_metrics` dictionary
   - For each category, retrieves corresponding `CategoryScoringAgent` from `scoring_agents` dictionary
   - Only evaluates metrics in categories that have corresponding agents

2. Sequential Metric Evaluation:
   - For each category in `selected_metrics`:
     - For each metric in the category's metric list:
       - Calls `category_agent.evaluate_metric(question, answer, metric)`
       - Collects result: `{metric, score, reasoning}`
   - Metrics within a category are evaluated sequentially
   - Categories are evaluated sequentially (can be parallelized)

3. Prompt Construction:
   Each scoring agent constructs a metric-specific prompt:
   ```python
   prompt = f"""
   Evaluate the following answer based on the given question using the {metric} metric.
   
   Question: {question}
   
   Answer: {answer}
   
   Metric Definition:
   {metric}: {METRIC_DEFINITIONS.get(metric, 'No definition available')}
   
   Evaluation Criteria:
   {criteria}
   
   Provide evaluation in JSON format:
   {{
       "Score": <score 0-5>,
       "Reasoning": "<brief explanation of the score>"
   }}
   
   Output format must be JSON only.
   """
   ```
   - `criteria` is obtained by calling `get_eval_criteria_{metric_key}()` function
   - Metric key is generated by: `metric.lower().replace(' ', '_').replace('-', '_')`

4. LLM Interaction:
   ```python
   messages = [
       {
           "role": "system", 
           "content": "You are an evaluation expert. Provide a numeric score and reasoning based on the evaluation criteria."
       },
       {
           "role": "user", 
           "content": prompt
       }
   ]
   llm_response = self.generator.generate(messages)
   ```

5. Response Parsing (Fallback Chain):
   Three-tier fallback strategy:
   
   Tier 1: JSON Parsing (with quotes)
   ```python
   score_match = re.search(r'"Score"\s*:\s*(\d+)', content, re.IGNORECASE)
   reason_match = re.search(r'"Reasoning"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
   ```
   
   Tier 2: Regex Extraction (without quotes)
   ```python
   if not score_match:
       score_match = re.search(r'Score\s*:\s*(\d+)', content, re.IGNORECASE)
   if not reason_match:
       reason_match = re.search(r'Reasoning\s*:\s*(.+)', content, re.IGNORECASE)
   ```
   
   Tier 3: Default Fallback
   ```python
   score = int(score_match.group(1)) if score_match else 0
   reasoning = reason_match.group(1).strip() if reason_match else content
   ```

6. Error Handling:
   - API failures: Returns `{"metric": metric, "score": 0, "reasoning": f"API error: {e}"}`
   - Missing evaluation criteria: Returns `{"metric": metric, "score": 0, "reasoning": "No evaluation criteria available"}`
   - Parsing failures: Defaults to score=0, uses full response as reasoning

Tool Use:
- LLM Model: Same as discriminator (configurable via generator)
- API Call: Single-shot generation per metric
- Temperature: Default (not explicitly set)
- Max Tokens: Not explicitly set (uses model default)
- Timeout: 180 seconds (for OpenRouterGenerator)

Implementation:
```python
def evaluate_metric(self, question, answer, metric):
    # Get evaluation criteria
    metric_key = metric.lower().replace(' ', '_').replace('-', '_')
    criteria_func = globals().get(f"get_eval_criteria_{metric_key}")
    criteria = criteria_func() if criteria_func else None
    
    # Construct prompt
    prompt = construct_evaluation_prompt(question, answer, metric, criteria)
    
    # LLM call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    content = self.generator.generate(messages)
    
    # Parse response with fallback
    score, reasoning = parse_score_and_reasoning(content)
    
    return {"metric": metric, "score": score, "reasoning": reasoning}
```

### Phase 3: Result Aggregation

Agent: `ComprehensiveEvaluator`

Input: All results from Phase 2

Output: Nested dictionary structure:
```python
{
    "Structure": {
        "Argument Soundness": {"metric": "...", "score": 4, "reasoning": "..."},
        "Logical Coherence": {"metric": "...", "score": 5, "reasoning": "..."},
        "Verbosity": {"metric": "...", "score": 3, "reasoning": "..."}
    },
    "Factuality": {
        "External Information Accuracy": {"metric": "...", "score": 4, "reasoning": "..."}
    },
    ...
}
```

Coordination Protocol:

1. Result Collection:
   - Iterates through all categories in `selected_metrics`
   - For each category, collects all metric evaluation results
   - Organizes results in nested dictionary: `evaluation_results[category][metric] = result`

2. Structure Validation:
   - Only includes categories that exist in `scoring_agents` dictionary
   - Only includes metrics that were selected by discriminator agent
   - Skips empty metric lists

3. Metadata Addition (in evaluation pipeline):
   - Adds question metadata: `question_number`, `question`, `answer`, `question_type`
   - Adds evaluator metadata: `evaluator_model`, `answer_model`, `dataset`
   - Flattens nested structure for CSV export

Implementation:
```python
def evaluate_question_answer(self, question, answer, question_type):
    # Phase 1: Metric selection
    selected_metrics = self.discriminator.determine_metrics(
        question_type, question, answer
    )
    
    # Phase 2: Scoring
    evaluation_results = {}
    for category, metrics in selected_metrics.items():
        if category in self.scoring_agents:
            category_results = {}
            for metric in metrics:
                result = self.scoring_agents[category].evaluate_metric(
                    question, answer, metric
                )
                category_results[metric] = result
            evaluation_results[category] = category_results
    
    # Phase 3: Return aggregated results
    return evaluation_results
```

## Communication Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  ComprehensiveEvaluator                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ evaluate_question_answer(question, answer, question_type)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────────┐         ┌──────────────────────┐
│  Phase 1:         │         │  Phase 2:            │
│  Metric Selection │         │  Parallel Scoring    │
└─────────┬─────────┘         └──────────┬───────────┘
          │                               │
          │ determine_metrics()           │
          │                               │
          ▼                               │
┌─────────────────────────────┐          │
│   DiscriminatorAgent        │          │
│                             │          │
│  - Type2: Fixed set         │          │
│  - Type3: Base + LLM        │          │
│    judgment (1 call)        │          │
│  - Type4: Base + LLM        │          │
│    judgment (1 call)        │          │
└───────────┬─────────────────┘          │
            │                             │
            │ Returns:                    │
            │ Dict[Category, List[Metric]]│
            │                             │
            └──────────────┬──────────────┘
                           │
                           │ For each (category, metrics) pair:
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Structure   │  │  Factuality  │  │  Data Sense  │
│  Agent       │  │  Agent       │  │  Agent       │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       │ evaluate_metric() for each metric in category
       │
       ▼
┌─────────────────────────────────────┐
│  LLM Call per Metric:               │
│  - Construct prompt                 │
│  - Include metric definition        │
│  - Include evaluation criteria      │
│  - Request JSON output              │
│  - Parse with fallback chain        │
└───────────┬─────────────────────────┘
            │
            │ Returns: {metric, score, reasoning}
            │
            ▼
┌─────────────────────────────────────┐
│  Phase 3: Result Aggregation        │
│  - Collect all results              │
│  - Organize by category             │
│  - Return nested structure          │
└─────────────────────────────────────┘
```

## Agent Initialization and Configuration

### Discriminator Agent Initialization

```python
# Generator initialization (shared across all agents)
generator = initialize_generator(model_key="gpt5", use_openrouter=True)
# or
generator = GCPGenerator(api_key=api_key, model="gemini-2.5-flash-lite")

# Discriminator agent creation
discriminator = DiscriminatorAgent(generator)
```

### Scoring Agents Initialization

```python
def create_scoring_agents_with_generator(generator):
    agents = {}
    
    agents["Structure"] = CategoryScoringAgent(
        "Structure", 
        ["Argument Soundness", "Logical Coherence", "Verbosity"], 
        generator
    )
    
    agents["Factuality"] = CategoryScoringAgent(
        "Factuality", 
        ["External Information Accuracy"],
        generator
    )
    
    agents["Data Sense"] = CategoryScoringAgent(
        "Data Sense", 
        ["Information Adequacy", "Trend Awareness", "Model Selection rationale"], 
        generator
    )
    
    agents["Insightfulness"] = CategoryScoringAgent(
        "Insightfulness", 
        ["Out-of-the-box Thinking", "Root Cause Depth", "Assumption Appropriateness"], 
        generator
    )
    
    agents["Operational Implementability"] = CategoryScoringAgent(
        "Operational Implementability", 
        ["Actionability", "Time-Based Planning"], 
        generator
    )
    
    agents["Purpose Alignment"] = CategoryScoringAgent(
        "Purpose Alignment", 
        ["Goal Orientation", "Stakeholder Orientation"], 
        generator
    )
    
    agents["Compliance"] = CategoryScoringAgent(
        "Compliance", 
        ["Risk Management", "Regulatory Compliance", "Ethical Responsibility"], 
        generator
    )
    
    return agents

scoring_agents = create_scoring_agents_with_generator(generator)
```

### Comprehensive Evaluator Initialization

```python
comprehensive_evaluator = ComprehensiveEvaluator(
    discriminator_agent=discriminator,
    scoring_agents=scoring_agents
)
```

## Tool Use Specifications

### LLM API Configuration

OpenRouter Generator (for GPT-5, Gemini 2.5 Pro, Llama 4):
- Base URL: `https://openrouter.ai/api/v1/chat/completions`
- Headers:
  ```python
  {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost",
      "X-Title": "Evaluation System"
  }
  ```
- Request Format:
  ```python
  {
      "model": model_id,  # e.g., "openai/gpt-5"
      "messages": messages,
      "temperature": temperature,  # Optional
      "max_tokens": max_tokens     # Optional
  }
  ```
- Timeout: 180 seconds
- Retry Strategy: Exponential backoff (2^attempt seconds), max 3 retries

GCP Generator (for Gemini 2.5 Flash Lite):
- Model: `gemini-2.5-flash-lite`
- API Key: From environment variable `GCP_KEY`
- Implementation: Uses `text2sql.engine.generation.GCPGenerator`

### Prompt Engineering

Discriminator Agent Prompts:
- Format: Simple question-answer format
- Output Constraint: "YES" or "NO" only
- Example:
  ```
  Analyze the following question and answer to determine if the question or answer involves numerical prediction.
  
  Question: {question}
  Answer: {answer}
  
  Does this question or answer involve numerical prediction or modeling that would require model selection rationale? Respond with only "YES" or "NO".
  ```

Scoring Agent Prompts:
- System Message: Role definition ("You are an evaluation expert with 10+ years of experience...")
- User Message Structure:
  1. Task description
  2. Question text
  3. Answer text
  4. Metric definition (from `METRIC_DEFINITIONS`)
  5. Evaluation criteria (from `get_eval_criteria_*` functions)
  6. Output format specification (JSON)
  7. Instructions for precision

### Response Parsing

Complete Parsing Function:
```python
def parse_score_and_reasoning(content: str) -> Tuple[int, str]:
    """
    Parse LLM response to extract score and reasoning.
    Uses three-tier fallback strategy.
    """
    # Tier 1: JSON format with quotes
    score_match = re.search(r'"Score"\s*:\s*(\d+)', content, re.IGNORECASE)
    reason_match = re.search(r'"Reasoning"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
    
    # Tier 2: Without quotes
    if not score_match:
        score_match = re.search(r'Score\s*:\s*(\d+)', content, re.IGNORECASE)
    if not reason_match:
        reason_match = re.search(r'Reasoning\s*:\s*(.+)', content, re.IGNORECASE)
    
    # Tier 3: Default fallback
    score = int(score_match.group(1)) if score_match else 0
    reasoning = reason_match.group(1).strip() if reason_match else content
    
    return score, reasoning
```

## State Management and Concurrency

### Stateless Design

- No Shared State: Each evaluation is independent
- Stateless Agents: Agents do not maintain state between evaluations
- Result Collection: Results are collected and returned, not stored internally
- Thread Safety: Agents can be used concurrently (each has its own generator instance)

### Sequential vs. Parallel Execution

Current Implementation: Sequential
- Categories evaluated sequentially
- Metrics within a category evaluated sequentially
- One LLM call at a time per agent

Potential Parallelization:
```python
# Parallel category evaluation (example)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=7) as executor:
    futures = {
        category: executor.submit(
            evaluate_category, 
            category, 
            metrics, 
            question, 
            answer
        )
        for category, metrics in selected_metrics.items()
    }
    results = {cat: future.result() for cat, future in futures.items()}
```

## Error Handling and Robustness

### Discriminator Agent Error Handling

- LLM API Failure: Returns `False` (defaults to not including optional metric)
- Invalid Question Type: Raises `ValueError`
- Missing Generator: Raises `AttributeError` on initialization

### Scoring Agent Error Handling

- LLM API Failure: Returns `{"metric": metric, "score": 0, "reasoning": f"API error: {e}"}`
- Missing Evaluation Criteria: Returns `{"metric": metric, "score": 0, "reasoning": "No evaluation criteria available"}`
- Parsing Failure: Defaults to score=0, uses full response as reasoning
- Invalid Metric: Returns score=0 with error message

### Comprehensive Evaluator Error Handling

- Missing Category Agent: Skips category
- Empty Metric List: Skips category
- Evaluation Failure: Continues with next metric/category

## Reproducibility Guarantees

### Deterministic Components

1. Fixed Metric Sets: Type2 questions always use the same metric set
2. Deterministic Parsing: Regex patterns are fixed and deterministic
3. Fixed Prompts: Prompt templates are static (except for question/answer substitution)

### Non-Deterministic Components

1. LLM Responses: Inherently non-deterministic (can be controlled with temperature=0)
2. Dynamic Metric Selection: Depends on LLM judgment (may vary between runs)

### Reproducibility Recommendations

1. Set Temperature to 0: For deterministic LLM outputs
2. Use Same Model: Ensure same model version across runs
3. Fix Random Seeds: If using any random components
4. Log All LLM Calls: Store prompts and responses for debugging
5. Version Control: Track model versions, prompt templates, and parsing logic

## Performance Characteristics

### Time Complexity

- Type2: O(1) metric selection + O(M) scoring, where M = number of metrics
- Type3/Type4: O(K) metric selection (K LLM calls) + O(M) scoring
- Total: O(K + M) LLM calls per question-answer pair

### Typical Metrics per Question Type

- Type2: ~8-10 metrics
- Type3: ~6-8 metrics (depending on LLM judgment)
- Type4: ~12-14 metrics (depending on LLM judgment)

### Estimated Time per Evaluation

- LLM Call Latency: ~2-5 seconds per call
- Type2 Evaluation: ~20-50 seconds (8-10 metrics × 2-5 seconds)
- Type3 Evaluation: ~15-45 seconds (1 discriminator call + 6-8 scoring calls)
- Type4 Evaluation: ~25-75 seconds (1 discriminator call + 12-14 scoring calls)

## Extension Points

### Adding New Categories

1. Add category to `create_scoring_agents_with_generator()`
2. Define metrics in `METRIC_CATEGORIES`
3. Add metric definitions to `METRIC_DEFINITIONS`
4. Create evaluation criteria function: `get_eval_criteria_{metric_key}()`
5. Update discriminator agent if category has optional metrics

### Adding New Metrics

1. Add metric to appropriate category in `METRIC_CATEGORIES`
2. Add definition to `METRIC_DEFINITIONS`
3. Create evaluation criteria function: `get_eval_criteria_{metric_key}()`
4. Update discriminator agent if metric is optional

### Customizing Coordination Protocol

1. Modify `ComprehensiveEvaluator.evaluate_question_answer()` for different coordination patterns
2. Implement parallel execution in Phase 2
3. Add caching layer for repeated evaluations
4. Implement result validation and quality checks

## Summary

The multi-agent coordination protocol provides separation of concerns:
- Discriminator Agent: Metric selection
- Scoring Agents: Domain-specific evaluation
- Comprehensive Evaluator: Orchestration and aggregation

The three-phase architecture ensures reproducibility through:
- Explicit coordination protocol
- Standardized communication interfaces
- Error handling
- Tool use specifications
