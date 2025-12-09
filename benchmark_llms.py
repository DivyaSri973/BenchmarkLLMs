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

# Configure logging
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

# UTILITY FUNCTIONS
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

class ConfusionMatrix:
    def __init__(self, labels: List[str] = None):
        if labels is None:
            labels = ["Positive", "Negative", "Neutral"]
        self.labels = labels
        self.matrix = {t: {p: 0 for p in labels} for t in labels}
        logger.debug(f"Initialized confusion matrix with labels: {labels}")

    def update(self, true_label: str, pred_label: str):
        if true_label in self.matrix and pred_label in self.matrix[true_label]:
            self.matrix[true_label][pred_label] += 1
            logger.debug(f"Updated confusion matrix: true={true_label}, pred={pred_label}")
        else:
            logger.warning(f"Invalid labels for confusion matrix: true={true_label}, pred={pred_label}")

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return self.matrix

class MetricsCollector:
    """Collect and calculate benchmark metrics."""
    
    def __init__(self, pricing: Dict[str, float]):
        self.pricing = pricing
        self.correct = 0
        self.total = 0
        self.latencies: List[float] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.confusion_matrix = ConfusionMatrix()
        self.errors = 0
        logger.debug("MetricsCollector initialized")
    
    def record_prediction(
        self, 
        predicted: str, 
        true_sentiment: str, 
        input_toks: int, 
        output_toks: int, 
        latency: float
    ):
        """Record a single prediction."""
        self.input_tokens += input_toks
        self.output_tokens += output_toks
        self.latencies.append(latency)
        self.total += 1
        
        if predicted == true_sentiment:
            self.correct += 1
        
        if true_sentiment in self.confusion_matrix.matrix:
            self.confusion_matrix.update(true_sentiment, predicted)
    
    def record_error(self):
        """Record an error occurrence."""
        self.errors += 1
    
    def calculate_metrics(self, model_name: str) -> Dict[str, Any]:
        """Calculate final metrics."""
        if self.total == 0:
            logger.warning(f"No predictions recorded for {model_name}")
            return {
                "model_name": model_name,
                "accuracy": 0.0,
                "average_latency_ms": 0.0,
                "estimated_total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "confusion_matrix": {},
                "total_predictions": 0,
                "errors": self.errors
            }
        
        accuracy = (self.correct / self.total * 100)
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        total_tokens = self.input_tokens + self.output_tokens
        cost = (
            self.input_tokens * self.pricing.get("input", 0.0) + 
            self.output_tokens * self.pricing.get("output", 0.0)
        ) / 1_000_000
        
        logger.info(
            f"Metrics for {model_name}: "
            f"accuracy={accuracy:.2f}%, "
            f"avg_latency={avg_latency:.2f}ms, "
            f"cost=${cost:.6f}, "
            f"errors={self.errors}"
        )
        
        return {
            "model_name": model_name,
            "accuracy": round(accuracy, 2),
            "average_latency_ms": round(avg_latency, 2),
            "estimated_total_tokens": int(total_tokens),
            "estimated_cost_usd": round(cost, 6),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "total_predictions": self.total,
            "errors": self.errors
        }


class BaseModel:
    def __init__(self, client: Any, config: ModelConfig, app_config: AppConfig):
        self.client = client
        self.config = config
        self.model_name = config.model_name
        self.app_config = app_config
        logger.info(f"Initialized {config.provider} model: {config.model_name}")

    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        raise NotImplementedError("Subclasses must implement this method.")
    
class OpenAIModel(BaseModel):    
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

class GroqModel(BaseModel):    
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

class BenchmarkRunner:    
    def __init__(self, app_config: AppConfig, parser: SentimentParser = None):
        self.app_config = app_config
        self.parser = parser or SentimentParser()
        logger.info("BenchmarkRunner initialized")

    def run_benchmark(self, df: pd.DataFrame, llm: BaseModel) -> Dict[str, Any]:
        model_identifier = f"{llm.config.provider}:{llm.config.model_name}"
        logger.info(f"Starting benchmark for {model_identifier}")
        
        metrics_collector = MetricsCollector(llm.config.pricing)
        total_samples = len(df)

        for idx, row in df.iterrows():
            feedback_text = row.get('feedback_text', '')
            if not feedback_text or pd.isna(feedback_text):
                logger.warning(f"Skipping row {idx + 1}: empty feedback_text")
                metrics_collector.record_error()
                continue
            
            prompt = f"Feedback: {feedback_text}"
            
            try:
                text, in_toks, out_toks, latency = llm.infer(prompt)
                predicted = self.parser.parse(text)
                true_sentiment = row.get("true_sentiment", "")
                metrics_collector.record_prediction(
                    predicted, 
                    true_sentiment, 
                    in_toks, 
                    out_toks, 
                    latency
                )
                is_correct = predicted == true_sentiment
                match_symbol = "✓" if is_correct else "✗"
                logger.info(
                    f"  [{metrics_collector.total}/{total_samples}] {match_symbol} "
                    f"Predicted: {predicted}, True: {true_sentiment}, "
                    f"Latency: {latency:.0f}ms"
                )

                # Sleep between requests
                if idx < total_samples - 1 and self.app_config.sleep_between_requests > 0:
                    time.sleep(self.app_config.sleep_between_requests)

            except Exception as e:
                logger.error(f"Error on sample {idx + 1}: {e}")
                metrics_collector.record_error()
                continue
        return metrics_collector.calculate_metrics(model_identifier)

class BenchmarkApp:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.models: List[BaseModel] = []
        self.runner = BenchmarkRunner(app_config)
        logger.info("BenchmarkApp initialized")

    def register_model(self, model: BaseModel):
        #Registering a model for benchmarking.
        self.models.append(model)
        logger.info(f"Registered model: {model.config.provider}:{model.config.model_name}")

    def run(self):
        #Running benchmarks for all registered models.
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
        """Save benchmark results to JSON file."""
        try:
            logger.info(f"Saving results to {self.app_config.output_json}")
            with open(self.app_config.output_json, "w") as f:
                json.dump({"models": results}, f, indent=2)
            logger.info("Results saved successfully")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print benchmark summary."""
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
                f"  Errors:         {r['errors']}"
            )
            logger.info(summary)

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

    # Gemini (commented out by default)
    # if gemini_key:
    #     try:
    #         logger.info("Initializing Gemini client")
    #         gemini_client = genai.Client(api_key=gemini_key)
    #         config = ModelConfig(
    #             provider="Gemini",
    #             model_name="gemini-1.5-flash",
    #             pricing={"input": 0.0, "output": 0.0}
    #         )
    #         app.register_model(GeminiModel(client, config, app_config))
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
            temperature=float(os.getenv("TEMPERATURE", "0.0"))
        )
        
        logger.info(f"Configuration loaded: csv_path={app_config.csv_path}")
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