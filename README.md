# LLM Sentiment Analysis Benchmark

A benchmarking system for comparing Large Language Model (LLM) performance on sentiment analysis tasks. This tool helps evaluate multiple LLM providers (OpenAI, Groq, Gemini) across key metrics: accuracy, latency, and cost.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Format](#output-format)
- [Architecture & Design](#architecture--design)
- [Business Recommendations](#business-recommendations)

---

## Overview

This benchmarking system evaluates LLMs for sentiment classification of customer feedback into three categories:
- **Positive**: Satisfaction, praise, appreciation
- **Negative**: Dissatisfaction, complaints, frustration
- **Neutral**: Informational queries, factual statements

### Key Capabilities

- **Multi-Provider Support**: OpenAI, Groq, Google Gemini
- **Comprehensive Metrics**: Accuracy, latency, cost, confusion matrix
- **Error Resilience**: Automatic retry with exponential backoff
- **Rate Limit Management**: Batch processing with configurable breaks
- **Production Logging**: Detailed logs for debugging and monitoring

---

## Features

### Core Functionality
- Compare multiple LLM providers simultaneously
- Track accuracy, latency, and cost per model
- Generate confusion matrices for detailed analysis
- Automatic retry mechanism for transient failures
- Batch processing with rate limit management

### Design
- Configurable via environment variables
- Structured logging to file and console
- Comprehensive error handling
- JSON output for easy integration
- Detailed metrics tracking

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for at least one LLM provider

### Step 1: Clone the Repository

```bash
git clone https://github.com/DivyaSri973/BenchmarkLLMs.git
cd BENCHMARKSLLMS
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
```txt
pandas>=2.0.0
openai>=1.0.0
groq>=0.4.0
google-genai>=0.2.0
python-dotenv>=1.0.0
```

---

## Configuration

### Step 1: Create `.env` File

Create a `.env` file in the project root:

```bash
# API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Input/Output Configuration
CSV_PATH=customer_feedback.csv
OUTPUT_JSON=benchmark_results.json

# Request Settings
SLEEP_BETWEEN_REQUESTS=0.5
MAX_TOKENS=10
TEMPERATURE=0.0

# Retry Configuration
MAX_RETRIES=3
RETRY_DELAY=2.0

# Batch Processing
BATCH_SIZE=10
BATCH_BREAK=5.0
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CSV_PATH` | `customer_feedback.csv` | Input CSV file path |
| `OUTPUT_JSON` | `benchmark_results.json` | Output JSON file path |
| `SLEEP_BETWEEN_REQUESTS` | `0.5` | Delay between requests (seconds) |
| `MAX_TOKENS` | `10` | Maximum tokens for response |
| `TEMPERATURE` | `0.0` | Model temperature (0-2) |
| `MAX_RETRIES` | `3` | Maximum retry attempts on failure |
| `RETRY_DELAY` | `2.0` | Initial retry delay (exponential backoff) |
| `BATCH_SIZE` | `10` | Number of requests before break |
| `BATCH_BREAK` | `5.0` | Break duration after batch (seconds) |

### Step 2: Prepare Input Data

Create `customer_feedback.csv` with the following format:

```csv
feedback_text,true_sentiment
"The new dashboard is incredibly fast and intuitive. Great job, team!","Positive"
"Your customer support was fantastic!","Positive"
"The app crashed three times today.","Negative"
"How do I change my payment information?","Neutral"
```

**Required Columns:**
- `feedback_text`: Customer feedback text
- `true_sentiment`: Ground truth label (Positive, Negative, or Neutral)

---

## Usage

### Run the code

```bash
python benchmark_llms.py
```

### Custom Configuration

Override default settings using environment variables:

```bash
CSV_PATH=my_feedback.csv OUTPUT_JSON=my_results.json python benchmark_llms.py
```

### Expected Output

The script will:

1. **Load Configuration**: Read environment variables and validate settings
2. **Initialize Models**: Connect to available LLM providers
3. **Run Benchmark**: Process each feedback sample
4. **Display Progress**: Show real-time results in console
5. **Save Results**: Generate `benchmark_results.json`
6. **Print Summary**: Display final metrics

## Output Format

### benchmark_results.json

```json
{
  "models": [
    {
      "model_name": "OpenAI:gpt-4o-mini",
      "accuracy": 96.0,
      "average_latency_ms": 684.67,
      "estimated_total_tokens": 9243,
      "estimated_cost_usd": 0.001398,
      "total_predictions": 25,
      "errors": 0,
      "retries": 2,
      "confusion_matrix": {
        "Positive": {
          "Positive": 8,
          "Negative": 0,
          "Neutral": 0
        },
        "Negative": {
          "Positive": 0,
          "Negative": 8,
          "Neutral": 1
        },
        "Neutral": {
          "Positive": 0,
          "Negative": 0,
          "Neutral": 8
        }
      }
    }
  ]
}
```

### Metrics Explained

- **accuracy**: Percentage of correct predictions
- **average_latency_ms**: Mean response time in milliseconds
- **estimated_total_tokens**: Sum of input and output tokens
- **estimated_cost_usd**: Estimated cost based on provider pricing
- **total_predictions**: Number of samples processed
- **errors**: Number of failed predictions
- **retries**: Number of retry attempts
- **confusion_matrix**: Breakdown of predictions vs. ground truth

### benchmark.log

Detailed execution logs saved to `benchmark.log`:
- API call details
- Error traces
- Retry attempts
- Batch processing checkpoints

---

## Architecture & Design

### Design Principles

This system follows **SOLID principles** and software engineering best practices:

#### 1. Single Responsibility Principle (SRP)
Each class has one clear purpose:
- **`SentimentParser`**: Parses model responses into sentiment labels
- **`MetricsCollector`**: Collects and calculates performance metrics
- **`ConfusionMatrix`**: Tracks prediction accuracy per class
- **`BenchmarkRunner`**: Orchestrates benchmark execution
- **`BenchmarkApp`**: Coordinates overall application flow

#### 2. Open/Closed Principle (OCP)
- **`BaseModel`** abstract class allows adding new providers without modifying existing code
- Extend functionality by creating new model classes (e.g., `ClaudeModel`, `MistralModel`)

#### 3. Dependency Inversion Principle (DIP)
- High-level modules depend on abstractions, not concrete implementations
- Models are injected via configuration, not hard-coded

### System Architecture

```
                      BenchmarkApp                            
  (Application Orchestrator)                                  
  - Registers models                                          
  - Coordinates execution                                     
                 │ uses
                 ▼
                    BenchmarkRunner                           
  - Iterates through dataset                                  
  - Handles batch processing                                  
  - Manages retries                                           
           │ uses                     │ uses
           ▼                          ▼
  MetricsCollector             BaseModel (Abstract)         
  - Tracks metrics         - Retry logic                    
  - Calculates stats       - Error handling                 
  - Confusion matrix       - Rate limiting                  
                                  │ implements
                                  ▼
                     OpenAIModel  GroqModel  GeminiModel    
                     - Provider-specific implementations     
```

### Key Design Decisions

#### Why OpenAI 4.0 Mini and Groq LLaMA
- OpenAI no longer offers a free tier; Gemini free tier is not available
- GPT-4o Mini is faster than other free-tier LLMs and accuracy is close to full GPT-4
- Groq LLaMA 3.1 8B Instant is very fast and available on free tier

#### Why Retry Logic ?
- Handles temporary API failures (timeouts, rate limits)
- Automatically retries with exponential backoff
- Ensures benchmark completes reliably even on free-tier APIs

#### Why Batch Processing?
-  Avoids hitting API rate limits
- Built-in pauses reduce unexpected costs
- Gives APIs time to recover between batches to reduce the throttling effect especially when using free tier

#### Future Implementations

- Prompt cache to reuse repeated inputs and reduce API calls
- Confidence scoring to flag uncertain predictions for review
- Parallel API requests for faster benchmarking

---

## Business Recommendations

### Results Summary

Based on 25 customer feedback samples:

| Metric | OpenAI (gpt-4o-mini) | Groq (llama-3.1-8b-instant) |
|--------|---------------------|------------------------------|
| **Accuracy** | 96.0% | 92.0% |
| **Avg Latency** | 684.67ms | 230.47ms |
| **Total Tokens** | 9,243 | 9,870 |
| **Estimated Cost** | $0.001398 | $0.000495 |
| **Cost per 1K samples** | $0.05592 | $0.0198 |

**Key Findings:**
- OpenAI provides **4% higher accuracy** but at **2.97x slower** response time
- Groq is **2.82x cheaper** while maintaining **92% accuracy**
- Both models handle Positive/Negative well; most errors on Neutral classification

---

### Decision Framework

Choosing the right model depends on multiple factors:

```
Can be based on primary requirement?
│
├─ ACCURACY is critical -> Consider OpenAI
│   ├─ False negatives costly? -> OpenAI (96% accuracy)
│   └─ Moderate accuracy OK? -> Groq (92% still strong)
│
├─ SPEED is critical -> Consider Groq
│   ├─ Sub-second response needed? -> Groq (230ms)
│   └─ Can tolerate delay?-> OpenAI (685ms acceptable)
│
└─ COST is critical -> Consider Groq
    ├─ High volume (>100K/day)? -> Groq (65% cost savings)
    └─ Low volume (<10K/day)? -> Either (cost difference minimal)
```

---

### Scenario Analysis

#### **Scenario 1: Real-Time Chatbot**

**Context:**
- User is actively waiting for response
- Needs to react instantly to user mood
- High volume of concurrent conversations
- User experience degraded by delays

**Critical Factors:**
1. **Latency** (Highest Priority): Users expect <1s response
2. **Accuracy** (High Priority): Must correctly identify user mood
3. **Cost** (Medium Priority): High volume requires cost efficiency

**Recommendation: Depends on Budget and Scale**

**Choose Groq if:**
- Handling >10,000 conversations/day
- Budget is constrained
- 92% accuracy is acceptable
- User experience requires <500ms response

**Cost-Benefit Analysis (10,000 chats/day):**
- **Groq**: $0.20/day = **$5.94/month** (230ms avg response)
- **OpenAI**: $0.56/day = **$16.78/month** (685ms avg response)
- **Savings: $10.84/month with Groq (65% cheaper)**

**Choose OpenAI if:**
- Handling sensitive customer support
- False negatives are costly (e.g., missing angry customers)
- Budget allows for 2.82x higher cost
- Can optimize other latency factors (caching, CDN)

**Hybrid Approach:**
```
User Message
    ↓
Fast Triage (Groq - 230ms)
    ├─ High Confidence (>90%) → Using Groq Result
    └─ Low Confidence (<90%) → Escalate to OpenAI
```
- Groq handles 80% of clear cases fast
- OpenAI handles ambiguous 20% accurately
- **Cost: ~$8.10/month, Avg Latency: ~321ms**

---

#### **Scenario 2: Overnight Batch Processing**

**Context:**
- Analyzing all previous day's feedback
- Runs overnight (8-hour window)
- No real-time user waiting
- Results used for business intelligence

**Critical Factors:**
1. **Accuracy** (Highest Priority): Drives business decisions
2. **Cost** (High Priority): Daily recurring expense
3. **Latency** (Low Priority): 8-hour window is ample

**Recommendation: Depends on Volume and Budget**

**Volume Scenarios:**

**Low Volume (<=100K feedbacks/day):**
- **OpenAI**: $168/month for 96% accuracy
- **Groq**: $59.4/month for 92% accuracy
- **Decision**: OpenAI - Cost difference minimal ($10.84/month), accuracy improvement justifies cost for business insights

**Medium Volume (1M/day):**
- **OpenAI**: $1680/month
- **Groq**: $600/month
- **Decision**: Depends on budget and accuracy requirements
  - If budget allows → OpenAI for better insights
  - If cost-conscious → Groq, manually review edge cases

**High Volume (>1M/day):**
- **Decision**: Groq - Cost savings/month justify slight accuracy drop

**Smart Batch Strategy:**
- Process with Groq first (fast & cheap)
- Filter low-confidence predictions
- Re-process uncertain ones with OpenAI
- Combine results

**Benefits:**
- 90% processed by Groq
- 10% validated by OpenAI
- **Effective accuracy: ~95%**

---

### Trade-Off Analysis

#### Accuracy vs. Cost

```
                 High Accuracy (96%)
                        ↑
                   OpenAI
                        │
                        │ +4% accuracy
                        │ +182% cost
                        │
                     Groq
                        │
                        ↓
              Low Cost ($0.000495/request)
```

**When 4% accuracy gain matters:**
- Financial fraud detection
- Medical triage systems
- Legal compliance scenarios
- Brand reputation monitoring

**When 4% accuracy gap acceptable:**
- General customer feedback analysis
- Social media sentiment monitoring
- Non-critical product reviews
- Marketing campaign analysis

#### Speed vs. Accuracy

```
              Fast Response (230ms)
                        ↑
                     Groq
                        │
                        │ -3x latency
                        │ -4% accuracy
                        │
                   OpenAI
                        │
                        ↓
            High Accuracy (96%)
```

**When speed is critical:**
- Real-time chat support
- Live event monitoring
- Interactive applications
- User-facing features

**When accuracy is critical:**
- Business intelligence
- Strategic decision-making
- Regulatory reporting
- Quality assurance

---

### Recommendations Summary

| Use Case | Primary Concern | Recommended Model | Rationale |
|----------|----------------|-------------------|-----------|
| **Real-time Chat (Low Volume)** | User Experience | OpenAI | Better accuracy, cost difference minimal |
| **Real-time Chat (High Volume)** | Cost + Speed | Groq | 3x faster, 65% cheaper, 92% still strong |
| **Batch Processing (Low Volume)** | Accuracy | OpenAI | Cost difference negligible, insights quality matters |
| **Batch Processing (High Volume)** | Cost | Groq | Significant savings, slight accuracy drop acceptable |
| **Hybrid (Best Balance)** | All Factors | Groq + OpenAI | Fast triage + accurate validation |

---

### Implementation Recommendations

#### Phase 1: Start with Groq
- Use Groq first because it’s fast and cheap
- Get a baseline of how well the model performs
- Keep an eye on accuracy while it runs in production

#### Phase 2: A/B Test OpenAI
- Try OpenAI on some cases to see real-world performance
- Check the business impact of better accuracy
- Figure out if the extra cost is worth it

#### Phase 3: Optimize Hybrid
- Let Groq handle predictions it’s confident about
- Send tricky cases to OpenAI for higher accuracy
- Find a balance between speed, cost, and accuracy

#### Monitoring Metrics
Keep track of these to make sure setup works well:
- **Accuracy Drift**: Are predictions staying accurate over time?
- **Cost per Prediction**: Are we spending what we expected?
- **Latency P95**: Are users experiencing delays?
- **False Negative Rate**: Are we missing critical issues?

---

## Troubleshooting

### Common Issues

**1. No models registered**
```
Error: No models registered. Please set at least one API key
```
**Solution**: Set API keys in `.env` file

**2. Rate limit errors**
```
Error: Rate limit exceeded
```
**Solution**: Increase `BATCH_BREAK` or decrease `BATCH_SIZE`

**3. CSV format errors**
```
Error: CSV must contain 'feedback_text' column
```
**Solution**: Ensure CSV has required columns: `feedback_text`, `true_sentiment`
