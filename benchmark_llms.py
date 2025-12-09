import json
import os
import time
import pandas as pd
from openai import OpenAI
from groq import Groq
from google import genai
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

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

def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")

class SentimentParser:
    @staticmethod
    def parse(text: str) -> str:
        if not text:
            return "Unknown"
        t = text.strip().lower()
        # Look for exact keywords or single-word responses
        if "positive" in t:
            return "Positive"
        if "negative" in t:
            return "Negative"
        if "neutral" in t:
            return "Neutral"
        # fallback heuristics
        for w in ["great", "love", "excellent", "fantastic", "helpful", "resolved", "thank"]:
            if w in t:
                return "Positive"
        for w in ["terrible", "awful", "frustrat", "crash", "lost", "unaccept", "horrible", "worst"]:
            if w in t:
                return "Negative"
        # If nothing matches, Unknown
        return "Unknown"
    
class ConfusionMatrix:
    def __init__(self, labels: List[str] = ["Positive", "Negative", "Neutral"]):
        self.labels = labels
        self.matrix = {t: {p: 0 for p in labels} for t in labels}

    def update(self, true_label: str, pred_label: str):
        if true_label in self.matrix and pred_label in self.matrix[true_label]:
            self.matrix[true_label][pred_label] += 1

    def to_dict(self):
        return self.matrix

class BaseModel:
    def __init__(self, client: Any, config: ModelConfig, app_config: AppConfig):
        self.client = client
        self.config = config
        self.model_name = config.model_name
        self.app_config = app_config

    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        raise NotImplementedError("Subclasses must implement this method.")
    
class OpenAIModel(BaseModel):
    def infer(self, prompt):
        start = time.time()
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
        text = ""
        input_tokens = 0
        output_tokens = 0
        try:
            text = response.choices[0].message.content
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or response.usage.get("prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0) or response.usage.get("completion_tokens", 0)
        except Exception:
            try:
                text = response["choices"][0]["message"]["content"]
                input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
                output_tokens = response.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                text = str(response)
        return text, int(input_tokens), int(output_tokens), latency
    
class GroqModel(BaseModel):
    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        start = time.time()
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
        text = ""
        input_tokens = 0
        output_tokens = 0
        try:
            text = response.choices[0].message.content
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or response.usage.get("prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0) or response.usage.get("completion_tokens", 0)
        except Exception:
            try:
                text = response["choices"][0]["message"]["content"]
                input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
                output_tokens = response.get("usage", {}).get("completion_tokens", 0)
            except Exception:
                text = str(response)
        return text, int(input_tokens), int(output_tokens), latency
    
class GeminiModel(BaseModel):
    def infer(self, prompt: str) -> Tuple[str, int, int, float]:
        start = time.time()
        # local import to avoid mandatory dependency if user doesn't use Gemini
        from google.genai import types  # type: ignore
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
        text = getattr(response, "text", str(response))
        # token usage metadata on genai
        input_tokens = getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0) or \
                       getattr(getattr(response, "usage_metadata", None), "input_token_count", 0) or 0
        output_tokens = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0) or \
                        getattr(getattr(response, "usage_metadata", None), "output_token_count", 0) or 0
        return text, int(input_tokens), int(output_tokens), latency
    
class BenchmarkEngine:
    def __init__(self, app_config: AppConfig, parser: SentimentParser = SentimentParser()):
        self.app_config = app_config
        self.parser = parser

    def benchmark_model(self, df: pd.DataFrame, llm: BaseModel) -> Dict[str, Any]:
        results = {
            "model_name": f"{llm.config.provider}:{llm.config.model_name}",
            "correct": 0,
            "total": 0,
            "latencies": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "confusion_matrix": ConfusionMatrix().to_dict()
        }

        cm = ConfusionMatrix()

        for idx, row in df.iterrows():
            prompt = f"Feedback: {row['feedback_text']}"
            try:
                text, in_toks, out_toks, latency = llm.infer(prompt)
                predicted = self.parser.parse(text)
                true_sentiment = row.get("true_sentiment", "")

                results["input_tokens"] += in_toks
                results["output_tokens"] += out_toks
                results["latencies"].append(latency)
                results["total"] += 1
                if predicted == true_sentiment:
                    results["correct"] += 1

                if true_sentiment in cm.matrix:
                    cm.update(true_sentiment, predicted)

                print(f"  {results['total']}/{len(df)}: {predicted} (true: {true_sentiment}) - {latency:.0f}ms")

                if idx < len(df) - 1 and self.app_config.sleep_between_requests > 0:
                    time.sleep(self.app_config.sleep_between_requests)

            except Exception as e:
                print(f"  Error on sample {results['total'] + 1}: {e}")
                continue

        accuracy = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0.0
        avg_latency = (sum(results["latencies"]) / len(results["latencies"])) if results["latencies"] else 0.0
        total_tokens = results["input_tokens"] + results["output_tokens"] or 1
        pricing = llm.config.pricing or {"input": 0.0, "output": 0.0}
        cost = (results["input_tokens"] * pricing.get("input", 0.0) + results["output_tokens"] * pricing.get("output", 0.0)) / 1_000_000
        results["confusion_matrix"] = cm.to_dict()

        return {
            "model_name": results["model_name"],
            "accuracy": round(accuracy, 2),
            "average_latency_ms": round(avg_latency, 2),
            "estimated_total_tokens": int(total_tokens),
            "estimated_cost_usd": round(cost, 6),
            "confusion_matrix": cm.to_dict()
        }

class BenchmarkApp:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.models: List[BaseModel] = []
        self.engine = BenchmarkEngine(app_config)

    def register_model(self, model: BaseModel):
        self.models.append(model)

    def run(self):
        df = load_csv(self.app_config.csv_path)
        if df.empty:
            print("No data found in CSV. Exiting.")
            return
        if "feedback_text" not in df.columns:
            raise ValueError("CSV must contain 'feedback_text' column.")
        if "true_sentiment" not in df.columns:
            print("Warning: 'true_sentiment' column not found. Accuracy metrics will be skipped.")

        results = []
        for model in self.models:
            print("=" * 70)
            print(f"BENCHMARKING: {model.config.provider} - {model.config.model_name}")
            print("=" * 70)
            r = self.engine.benchmark_model(df, model)
            results.append(r)

        # Save results
        with open(self.app_config.output_json, "w") as f:
            json.dump({"models": results}, f, indent=2)

        print("\nBENCHMARK SUMMARY")
        for r in results:
            print(f"\n{r['model_name']}:")
            print(f"  Accuracy:      {r['accuracy']}%")
            print(f"  Avg Latency:   {r['average_latency_ms']}ms")
            print(f"  Total Tokens:  {r['estimated_total_tokens']}")
            print(f"  Cost:          ${r['estimated_cost_usd']}")

def build_app(app_config: AppConfig) -> BenchmarkApp:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    app = BenchmarkApp(app_config)

    if openai_api_key:
        try:
            from openai import OpenAI 
            openai_client = OpenAI(api_key=openai_api_key)
            openai_config = ModelConfig(provider="OpenAI", model_name="gpt-4o-mini", pricing={"input": 0.15, "output": 0.60})
            app.register_model(OpenAIModel(openai_client, openai_config, app_config))
        except Exception as e:
            print("Warning: Could not initialize OpenAI client:", e)

    if groq_api_key:
        try:
            from groq import Groq 
            groq_client = Groq(api_key=groq_api_key)
            groq_config = ModelConfig(provider="Groq", model_name="llama-3.1-8b-instant", pricing={"input": 0.05, "output": 0.08})
            app.register_model(GroqModel(groq_client, groq_config, app_config))
        except Exception as e:
            print("Warning: Could not initialize Groq client:", e)

    # if gemini_api_key:
    #     try:
    #         from google import genai  # type: ignore
    #         gemini_client = genai.Client(api_key=gemini_api_key)
    #         cfg = ModelConfig(provider="Gemini", model_name="gemini-1.5-flash", pricing={"input": 0.0, "output": 0.0})
    #         app.register_model(GeminiModel(gemini_client, cfg, app_config))
    #     except Exception as e:
    #         print("Warning: Could not initialize Gemini client:", e)

    if not app.models:
        print("No models registered (no API keys found). Exiting.")
    return app

def main():
    app_config = AppConfig(
        csv_path=os.getenv("CSV_PATH", "customer_feedback.csv"),
        output_json=os.getenv("OUTPUT_JSON", "benchmark_results.json"),
        sleep_between_requests=float(os.getenv("SLEEP_BETWEEN_REQUESTS", 2)),
        max_tokens=int(os.getenv("MAX_TOKENS", 10)),
        temperature=float(os.getenv("TEMPERATURE", 0.0))
    )

    app = build_app(app_config)
    if app.models:
        app.run()

if __name__ == "__main__":
    main()