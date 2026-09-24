# Support Assistant Module

This module implements a customer support assistant pipeline equipped with tool execution and sentiment analysis.

## Pipeline Architecture
1. **Sentiment Analysis**: Evaluates customer message polarity using `TextBlob` to assign an appropriate response tone and set escalation flags for frustrated users.
2. **Tool Execution Engine**:
   - `check_inventory`: Connects to SQLite (`data_pipeline/zepto_catalog.db`) to verify product stock.
   - `calculate_discount`: Computes percentage discount reductions on product pricing.
3. **Orchestrator**: Processes incoming queries, evaluates sentiment, triggers required tools, and structures output.

## Sample Execution Call Transcripts (MOCK_LLM Default)

```text
==================================================
User Message: 'Hello, do you have any books about Sapiens in stock?'

[Sentiment Analysis]
Detected Sentiment: Neutral (Score: 0.0)
Tone Strategy: Professional, concise, and direct.
Escalation Flag: False

[Tool Execution]
Inventory Check Tool Result:
No products found matching 'Sapiens'.
==================================================

==================================================
User Message: 'My order is taking forever! Can I get a 15% discount on this 500 rupees book?'

[Sentiment Analysis]
Detected Sentiment: Neutral (Score: 0.0)
Tone Strategy: Professional, concise, and direct.
Escalation Flag: False

[Tool Execution]
Discount Tool Result: Original ₹500.0 with 15.0% off -> Final: ₹425.0
==================================================
```