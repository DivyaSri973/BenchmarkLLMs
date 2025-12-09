'''
    To benchmark multiple LLM providers (OpenAI, Groq, Gemini) on sentiment analysis of customer feedback.
'''

import json
import os
import time
import pandas as pd
from openai import OpenAI
from groq import Groq
from google import genai
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

# Configuring logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark")

SYSTEM_PROMPT = """You are an expert sentiment analysis system specializing in customer feedback classification.

Your task is to analyze customer feedback and classify it into exactly ONE of three categories:
- Positive: Feedback expressing satisfaction, praise, appreciation, or positive experiences
- Negative: Feedback expressing dissatisfaction, complaints, frustration, or negative experiences  
- Neutral: Feedback that is informational, asking questions, or neither clearly positive nor negative

ANALYSIS GUIDELINES:
1. Consider both explicit sentiment (direct statements) and implicit sentiment (tone, context)
2. Look for sentiment indicators like:
   - Positive: "great", "love", "excellent", "fantastic", "helpful", "resolved", "saved", "thank you"
   - Negative: "terrible", "awful", "frustrated", "crashed", "lost", "unacceptable", "horrible", "worst"
   - Neutral: questions, factual statements, feature requests without emotion
3. Weigh the overall message - a complaint with a positive resolution is still positive
4. Sarcasm like "just great" with negative context should be classified as Negative
5. Mixed feedback should be classified based on the dominant sentiment

RESPONSE FORMAT:
Respond with ONLY the classification word: "Positive", "Negative", or "Neutral"
Do not include explanations, punctuation, or additional text.

EXAMPLES:
- "The dashboard is incredibly fast!" -> Positive
- "I've been on hold for 45 minutes." -> Negative  
- "How do I reset my password?" -> Neutral
- "The app crashed and I lost my work." -> Negative
- "Thank you for resolving my issue quickly!" -> Positive
"""

@dataclass
class ModelConfig:
    provider: str
    model_name: str
    pricing: Dict[str, float]

@dataclass
class AppConfig:
    csv_path: str = "customer_feedback.csv"
    output_json: str = "benchmark_results.json"
    sleep_between_requests: float = 0.5 
    max_tokens: int = 10
    temperature: float = 0.0
    max_retries: int = 3  # Maximum retry attempts
    retry_delay: float = 2.0  # Initial retry delay in seconds
    batch_size: int = 10  # Requests before break
    batch_break: float = 5.0  # Break duration after batch

# utility functions
def load_csv(path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading CSV file from: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV file is empty")
        logger.info(f"Successfully loaded {len(df)} rows from CSV")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or has no data")
        raise ValueError("CSV file is empty")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}", exc_info=True)
        raise RuntimeError(f"Error loading CSV file: {e}")

class SentimentParser:    
    @staticmethod
    def parse(text: str) -> str:
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text received for parsing: {type(text)}")
            return "Unknown"
            
        t = text.strip().lower()
        if "positive" in t:
            return "Positive"
        if "negative" in t:
            return "Negative"
        if "neutral" in t:
            return "Neutral"
            
        # Fallback heuristics
        for w in ["great", "love", "excellent", "fantastic", "helpful", "resolved", "thank"]:
            if w in t:
                logger.debug(f"Matched fallback positive keyword '{w}'")
                return "Positive"
                
        for w in ["terrible", "awful", "frustrat", "crash", "lost", "unaccept", "horrible", "worst"]:
            if w in t:
                logger.debug(f"Matched fallback negative keyword '{w}'")
                return "Negative"
                
        # No match found
        logger.warning(f"Could not parse sentiment from: {text[:100]}")
        return "Unknown"

# Base model class for LLMs with retry logic
class BaseModel:
    def __init__(self, client: Any, config: ModelConfig, app_config: AppConfig):
        self.client = client
        self.config = config
        self.model_name = config.model_name
        self.app_config = app_config
        logger.info(f"Initialized {config.provider} model: {config.model_name}")

    # To infer the llms with custom retry logic
    def infer_with_retry(self, prompt: str) -> Tuple[str, int, int, float]:
        last_exception = None
        
        for attempt in range(self.app_config.max_retries):
            try:
                # Attempt inference
                return self.infer(prompt)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.app_config.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = self.app_config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Inference failed (attempt {attempt + 1}/{self.app_config.max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Inference failed after {self.app_config.max_retries} attempts: {e}"
                    )
        
        # All retries exhausted
        raise last_exception

    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        #Subclasses will implement actual inference
        raise NotImplementedError("Subclasses must implement this method.")

# OpenAI model class for inference
class OpenAIModel(BaseModel):    
    # To perform inference using OpenAI API
    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        logger.debug(f"Starting OpenAI inference")
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.app_config.max_tokens,
                temperature=self.app_config.temperature
            )
            latency = (time.time() - start) * 1000.0
            text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            logger.debug(f"OpenAI inference completed: {latency:.2f}ms")
            return text, int(input_tokens), int(output_tokens), latency
        except Exception as e:
            logger.error(f"OpenAI inference failed: {e}")
            raise

# Groq model class for inference
class GroqModel(BaseModel):    
    # To infer from LLama using Groq API
    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        logger.debug(f"Starting Groq inference")
        start = time.time()
    
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.app_config.max_tokens,
                temperature=self.app_config.temperature
            )
            latency = (time.time() - start) * 1000.0
            text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            logger.debug(f"Groq inference completed: {latency:.2f}ms")
            return text, int(input_tokens), int(output_tokens), latency
        except Exception as e:
            logger.error(f"Groq inference failed: {e}")
            raise

# Gemini model class for inference
class GeminiModel(BaseModel):
    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        logger.debug(f"Starting Gemini inference")
        start = time.time()

        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.app_config.temperature,
                    max_output_tokens=self.app_config.max_tokens
                )
            )
            
            latency = (time.time() - start) * 1000.0
            
            # Extract response data
            text = getattr(response, "text", "")
            usage = getattr(response, "usage_metadata", None)
            input_tokens = 0
            output_tokens = 0
            
            if usage:
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            
            logger.debug(f"Gemini inference completed: {latency:.2f}ms")
            return text, int(input_tokens), int(output_tokens), latency
            
        except Exception as e:
            logger.error(f"Gemini inference failed: {e}")
            raise

# This class will run the benchmark process for given model
class BenchmarkRunner:    
    def __init__(self, app_config: AppConfig, parser: SentimentParser = None):
        self.app_config = app_config
        self.parser = parser or SentimentParser()
        logger.info("BenchmarkRunner initialized")

    def run_benchmark(self, df: pd.DataFrame, llm: BaseModel) -> Dict[str, Any]:
        #runs benchmark for a given model and return metrics
        model_id = f"{llm.config.provider}:{llm.config.model_name}"
        logger.info(f"Starting benchmark for {model_id}")
        logger.info(
            f"Batch settings: {self.app_config.batch_size} requests per batch, "
            f"{self.app_config.batch_break}s break between batches"
        )

        metrics = self._init_metrics()
        total_samples = len(df)
        requests_in_batch = 0

        for idx, row in df.iterrows():
            if not self._validate_row(row):
                logger.warning(f"Skipping row {idx + 1}: empty feedback_text")
                metrics["errors"] += 1
                continue
            prompt = self._build_prompt(row)

            try:
                text, in_toks, out_toks, latency = self._run_llm(llm, prompt, metrics)
                self._update_metrics(metrics, text, row, in_toks, out_toks, latency)
                self._log_progress(metrics, total_samples, latency)
                requests_in_batch = self._batch_sleep_logic(
                    requests_in_batch, idx, total_samples
                )
            except Exception as e:
                logger.error(f"Error on sample {idx + 1}: {e}")
                metrics["errors"] += 1
                continue
        return self._finalize_metrics(metrics, llm, model_id)
    
    def _init_metrics(self) -> Dict[str, Any]:
        # Initializing metrics for current benchmark model run
        labels = ["Positive", "Negative", "Neutral"]
        return {
            "correct": 0,
            "total": 0,
            "latencies": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "errors": 0,
            "retries": 0,
            "confusion_matrix": {t: {p: 0 for p in labels} for t in labels},
            "predicted": None,
            "true": None
        }

    def _validate_row(self, row: pd.Series) -> bool:
        # if feedback_text exists and valid
        feedback = row.get("feedback_text", "")
        return feedback and not pd.isna(feedback)

    def _build_prompt(self, row: pd.Series) -> str:
        #Construct LLM prompt from row data
        return f"Feedback: {row.get('feedback_text')}"

    def _run_llm(self, llm: BaseModel, prompt: str, metrics: Dict[str, Any]) -> Tuple[str, int, int, float]:
        #calls LLM with retry logic and track retries
        text, in_toks, out_toks, latency = llm.infer_with_retry(prompt)
        if latency > (self.app_config.retry_delay * 1000):
            metrics["retries"] += 1

        return text, in_toks, out_toks, latency

    def _update_metrics(
        self, 
        metrics: Dict[str, Any], 
        text: str, 
        row: pd.Series, 
        in_toks: int, 
        out_toks: int, 
        latency: float
    ):
        predicted = self.parser.parse(text)
        true_sentiment = row.get("true_sentiment", "")
        metrics["predicted"] = predicted
        metrics["true"] = true_sentiment

        metrics["total"] += 1
        metrics["input_tokens"] += in_toks
        metrics["output_tokens"] += out_toks
        metrics["latencies"].append(latency)
        if predicted == true_sentiment:
            metrics["correct"] += 1

        # Update confusion matrix
        if true_sentiment in metrics["confusion_matrix"]:
            if predicted in metrics["confusion_matrix"][true_sentiment]:
                metrics["confusion_matrix"][true_sentiment][predicted] += 1

    def _log_progress(self, metrics: Dict[str, Any], total_samples: int, latency: float):
        """Log progress for current prediction."""
        match_symbol = "✓" if metrics["predicted"] == metrics["true"] else "✗"
        logger.info(
            f"  [{metrics['total']}/{total_samples}] {match_symbol} "
            f"Predicted: {metrics['predicted']}, True: {metrics['true']}, "
            f"Latency: {latency:.0f}ms"
        )

    def _batch_sleep_logic(self, requests_in_batch: int, idx: int, total_samples: int) -> int:
        #sleeps between batches and requests to avoid throttling
        requests_in_batch += 1
        if requests_in_batch >= self.app_config.batch_size:
            logger.info(
                f"Completed batch of {self.app_config.batch_size} requests. "
                f"Taking {self.app_config.batch_break}s break..."
            )
            time.sleep(self.app_config.batch_break)
            return 0
        elif idx < total_samples - 1 and self.app_config.sleep_between_requests > 0:
            time.sleep(self.app_config.sleep_between_requests)

        return requests_in_batch

    def _finalize_metrics(
        self, 
        metrics: Dict[str, Any], 
        llm: BaseModel, 
        model_id: str
    ) -> Dict[str, Any]:
        #Computes accuracy, cost, latency
        if metrics["total"] == 0:
            logger.warning(f"No predictions recorded for {model_id}")
            return {
                "model_name": model_id,
                "accuracy": 0.0,
                "average_latency_ms": 0.0,
                "estimated_total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "confusion_matrix": {},
                "total_predictions": 0,
                "errors": metrics["errors"],
                "retries": metrics["retries"]
            }

        accuracy = (metrics["correct"] / metrics["total"] * 100)
        avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0.0
        total_tokens = metrics["input_tokens"] + metrics["output_tokens"]
        cost = (
            metrics["input_tokens"] * llm.config.pricing.get("input", 0.0) +
            metrics["output_tokens"] * llm.config.pricing.get("output", 0.0)
        ) / 1_000_000

        logger.info(
            f"Metrics for {model_id}: "
            f"accuracy={accuracy:.2f}%, "
            f"avg_latency={avg_latency:.2f}ms, "
            f"cost=${cost:.6f}, "
            f"errors={metrics['errors']}, "
            f"retries={metrics['retries']}"
        )

        return {
            "model_name": model_id,
            "accuracy": round(accuracy, 2),
            "average_latency_ms": round(avg_latency, 2),
            "estimated_total_tokens": int(total_tokens),
            "estimated_cost_usd": round(cost, 6),
            "confusion_matrix": metrics["confusion_matrix"],
            "total_predictions": metrics["total"],
            "errors": metrics["errors"],
            "retries": metrics["retries"]
        }



# This is the main class for the application this will tie everything together
class BenchmarkApp:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.models: List[BaseModel] = []
        self.runner = BenchmarkRunner(app_config)
        logger.info("BenchmarkApp initialized")

    def register_model(self, model: BaseModel):
        #Registering models for benchmarking
        self.models.append(model)
        logger.info(f"Registered model: {model.config.provider}:{model.config.model_name}")

    def run(self):
        #Running benchmark on registered models
        logger.info("Starting benchmark run")
        try:
            df = load_csv(self.app_config.csv_path)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            print(f"Error: {e}")
            return
        
        if "feedback_text" not in df.columns:
            logger.error("CSV missing 'feedback_text' column")
            raise ValueError("CSV must contain 'feedback_text' column.")
            
        if "true_sentiment" not in df.columns:
            logger.warning("'true_sentiment' column not found in CSV")
            print("Warning: 'true_sentiment' column not found. Accuracy metrics will be limited.")

        results = []
        for model in self.models:
            logger.info("=" * 70)
            logger.info(f"BENCHMARKING: {model.config.provider} - {model.config.model_name}")
            logger.info("=" * 70)
            
            try:
                result = self.runner.run_benchmark(df, model)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed for {model.config.provider}: {e}")
                continue

        if not results:
            logger.error("No models completed successfully")
            print("No models completed successfully.")
            return

        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)

    def _save_results(self, results: List[Dict[str, Any]]):
        #Save results to JSON file
        try:
            logger.info(f"Saving results to {self.app_config.output_json}")
            with open(self.app_config.output_json, "w") as f:
                json.dump({"models": results}, f, indent=2)
            logger.info("Results saved successfully")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def _print_summary(self, results: List[Dict[str, Any]]):
        logger.info("=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70)
        
        for r in results:
            summary = (
                f"\n{r['model_name']}:\n"
                f"  Accuracy:       {r['accuracy']}%\n"
                f"  Avg Latency:    {r['average_latency_ms']}ms\n"
                f"  Total Tokens:   {r['estimated_total_tokens']}\n"
                f"  Cost:           ${r['estimated_cost_usd']}\n"
                f"  Errors:         {r['errors']}\n"
                f"  Retries:        {r['retries']}"
            )
            logger.info(summary)


# To build app with llm models and to configure the models
def build_app(app_config: AppConfig) -> BenchmarkApp:
    logger.info("Building application with configured models")
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    app = BenchmarkApp(app_config)
    
    # OpenAI
    if openai_key:
        try:
            logger.info("Initializing OpenAI client")
            openai_client = OpenAI(api_key=openai_key)
            openai_config = ModelConfig(
                provider="OpenAI",
                model_name="gpt-4o-mini",
                pricing={"input": 0.15, "output": 0.60}
            )
            app.register_model(OpenAIModel(openai_client, openai_config, app_config))
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI: {e}")

    # Groq
    if groq_key:
        try:
            logger.info("Initializing Groq client")
            groq_client = Groq(api_key=groq_key)
            groq_config = ModelConfig(
                provider="Groq",
                model_name="llama-3.1-8b-instant",
                pricing={"input": 0.05, "output": 0.08}
            )
            app.register_model(GroqModel(groq_client, groq_config, app_config))
        except Exception as e:
            logger.warning(f"Could not initialize Groq: {e}")

    # Gemini 
    # if gemini_key:
    #     try:
    #         logger.info("Initializing Gemini client")
    #         gemini_client = genai.Client(api_key=gemini_key)
    #         gemini_config = ModelConfig(
    #             provider="Gemini",
    #             model_name="gemini-1.5-flash",
    #             pricing={"input": 0.0, "output": 0.0}
    #         )
    #         app.register_model(GeminiModel(gemini_client, gemini_config, app_config))
    #     except Exception as e:
    #         logger.warning(f"Could not initialize Gemini: {e}")

    if not app.models:
        raise RuntimeError(
            "No models registered. Please set at least one API key:\n"
            "  - OPENAI_API_KEY\n"
            "  - GROQ_API_KEY\n"
            "  - GEMINI_API_KEY"
        )
        
    logger.info(f"Successfully registered {len(app.models)} model(s)")
    return app


def main():
    logger.info("Starting LLM Benchmark application")
    
    try:
        # Load configuration
        app_config = AppConfig(
            csv_path=os.getenv("CSV_PATH", "customer_feedback.csv"),
            output_json=os.getenv("OUTPUT_JSON", "benchmark_results.json"),
            sleep_between_requests=float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "10")),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "2.0")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            batch_break=float(os.getenv("BATCH_BREAK", "5.0"))
        )
        
        logger.info(f"Configuration loaded: csv_path={app_config.csv_path}")
        logger.info(f"Retry settings: max_retries={app_config.max_retries}, retry_delay={app_config.retry_delay}s")
        logger.info(f"Batch settings: batch_size={app_config.batch_size}, batch_break={app_config.batch_break}s")
        
        app = build_app(app_config)
        app.run()
        
        logger.info("Benchmark completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()