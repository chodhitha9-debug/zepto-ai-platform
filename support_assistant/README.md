# Zepto AI Platform

A complete AI/data analytics platform containing a web-scraping data pipeline, Titanic analytics and machine-learning models, and a policy-based customer support assistant.

## Project Structure

```text
zepto-ai-platform/
│
├── data_pipeline/
│   ├── data_pipeline.py
│   ├── books_raw.csv
│   ├── books_clean.csv
│   └── books.db
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   ├── decision_tree.png
│   ├── regression_residuals.png
│   ├── pipeline.joblib
│   └── charts/
│
├── support_assistant/
│   ├── assistant.py
│   ├── Dockerfile
│   ├── README.md
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   └── chroma_db/
│
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

The main dependencies include Requests, BeautifulSoup, Pandas, NumPy, Scikit-learn, imbalanced-learn, Seaborn, Matplotlib, Joblib, ChromaDB, Sentence Transformers, LangGraph, FastAPI, Uvicorn, and Pydantic.

## 1. Data Pipeline

The data pipeline is implemented in:

```text
data_pipeline/data_pipeline.py
```

It scrapes book information using `requests` and `BeautifulSoup`.

The pipeline collects at least 60 books from multiple categories and stores the following raw fields:

* title
* price
* star_rating
* availability
* category

The cleaned dataset contains:

* title
* category
* price_gbp
* rating
* in_stock
* price_inr

Star ratings are converted from words such as `One`, `Two`, `Three`, `Four`, and `Five` into numeric values from 1 to 5.

Availability is converted into a Boolean in-stock field.

The fixed exchange rate used for conversion is:

```text
1 GBP = 105.50 INR
```

The pipeline also creates a normalized SQLite database containing categories and books with a foreign-key relationship.

Example SQL operations demonstrate:

* SELECT
* WHERE
* ORDER BY
* LIMIT
* DISTINCT
* IN
* BETWEEN
* JOIN

SQL results are reproduced using both `pandas.read_sql()` and `pandas.merge()`.

### Run the Data Pipeline

```powershell
python .\data_pipeline\data_pipeline.py
```

Outputs:

```text
data_pipeline/books_raw.csv
data_pipeline/books_clean.csv
data_pipeline/books.db
```

## 2. Analytics

The analytics module contains two scripts.

```text
analytics/01_eda.py
analytics/02_modeling.py
```

### Exploratory Data Analysis

`01_eda.py` loads the Titanic dataset, saves a local copy, and performs exploratory data analysis.

The analysis includes:

* dataset information
* descriptive statistics
* dataset shape
* missing-value percentages
* missing-value treatment
* age and fare histograms
* age and fare boxplots
* IQR outlier counts
* fare mean, median, mode, and skewness
* survival by sex
* survival by passenger class
* survival by sex and passenger class
* a 6x6 correlation analysis
* correlation heatmap
* strongest absolute correlations
* age and fare standardization
* multivariate visualizations

The six variables used for the required correlation matrix are:

```text
survived
pclass
age
sibsp
parch
fare
```

The cleaned Titanic dataset is saved locally so that modeling can use the saved dataset rather than repeatedly downloading it.

Run:

```powershell
python .\analytics\01_eda.py
```

Outputs include:

```text
analytics/titanic.csv
analytics/titanic_cleaned.csv
analytics/charts/
```

### Machine Learning Modeling

`02_modeling.py` performs classification and regression.

The classification workflow includes:

* stratified train/test split
* preprocessing with imputation, scaling, and one-hot encoding
* Logistic Regression
* Decision Tree
* Random Forest
* accuracy
* precision
* recall
* F1 score
* ROC AUC
* confusion matrices
* decision-tree visualization
* class-weight-balanced models
* SMOTE comparison
* Random Forest hyperparameter tuning

The Random Forest grid search evaluates:

```text
n_estimators
max_depth
max_features
```

SMOTE is applied only to the training data through the modeling pipeline.

The final Random Forest uses out-of-bag evaluation with:

```text
oob_score=True
```

The regression task predicts passenger fare and reports:

* MAE
* RMSE
* R²
* Adjusted R²
* residual visualization
* heteroscedasticity assessment

The complete fitted preprocessing and Random Forest pipeline is saved with Joblib and subsequently reloaded to demonstrate prediction from raw input.

Run:

```powershell
python .\analytics\02_modeling.py
```

Important generated outputs include:

```text
analytics/decision_tree.png
analytics/regression_residuals.png
analytics/pipeline.joblib
```

## 3. Support Assistant

The support assistant is implemented in:

```text
support_assistant/assistant.py
```

It provides a FastAPI customer-support API backed by eight policy documents.

The policy documents are:

```text
doc_01.txt
doc_02.txt
doc_03.txt
doc_04.txt
doc_05.txt
doc_06.txt
doc_07.txt
doc_08.txt
```

The system uses:

* `all-MiniLM-L6-v2` embeddings
* ChromaDB
* LangGraph
* TypedDict graph state
* Pydantic response validation
* FastAPI
* Uvicorn

### LangGraph Flow

The graph contains the following named nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The flow is:

```text
User Query
    |
    v
classify_intent
    |
    +----------------------+
    |                      |
    v                      v
policy_question       general_question
    |                      |
    v                      v
retrieve_and_answer   direct_answer
    |                      |
    +----------+-----------+
               |
              END
```

Policy questions retrieve the top three relevant policy chunks from ChromaDB.

General questions use the direct-answer route.

### Structured Prompt

The policy-answering prompt contains:

* Role
* Context
* Task
* Format
* Length
* Negative constraint
* Few-shot example

The negative constraint prevents the assistant from inventing policy information outside the retrieved context.

### Response Schema

API responses follow the Pydantic structure:

```json
{
  "answer": "string",
  "sources": [],
  "confidence": 0.9
}
```

`confidence` is constrained to the range 0 to 1.

General questions return an empty `sources` list.

### Mock LLM Mode

The application supports:

```text
MOCK_LLM=1
```

This mode does not require an external LLM or external network call for answer generation.

On Windows PowerShell:

```powershell
$env:MOCK_LLM="1"
python .\support_assistant\assistant.py
```

The API runs on:

```text
http://127.0.0.1:7860
```

### API Endpoint

The application exposes:

```text
POST /ask
```

Example request:

```json
{
  "query": "What is the delivery fee for orders under INR 500?"
}
```

Example response:

```json
{
  "answer": "Based on the provided policy context: ...",
  "sources": [
    "doc_01_chunk_0",
    "doc_03_chunk_0",
    "doc_07_chunk_0"
  ],
  "confidence": 0.9
}
```

Example PowerShell request:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the delivery fee for orders under INR 500?"}'
```

## 4. Docker

The support assistant includes a Dockerfile.

Build the image from the repository root:

```powershell
docker build -f .\support_assistant\Dockerfile -t zepto-support-assistant .
```

Run the container:

```powershell
docker run -p 7860:7860 zepto-support-assistant
```

The API is then available at:

```text
http://127.0.0.1:7860
```

## 5. Generated Outputs

### Data Pipeline

```text
data_pipeline/books_raw.csv
data_pipeline/books_clean.csv
data_pipeline/books.db
```

### Analytics

```text
analytics/titanic.csv
analytics/titanic_cleaned.csv
analytics/decision_tree.png
analytics/regression_residuals.png
analytics/pipeline.joblib
analytics/charts/
```

### Support Assistant

```text
support_assistant/chroma_db/
```

The ChromaDB directory stores the local vector database used for policy retrieval.

## 6. End-to-End Execution

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the data pipeline:

```powershell
python .\data_pipeline\data_pipeline.py
```

Run exploratory analysis:

```powershell
python .\analytics\01_eda.py
```

Run modeling:

```powershell
python .\analytics\02_modeling.py
```

Run the support assistant:

```powershell
$env:MOCK_LLM="1"
python .\support_assistant\assistant.py
```

## 7. Design Summary

The project is divided into three independent modules.

### Data Pipeline

Responsible for acquiring, cleaning, transforming, validating, and storing book data.

### Analytics

Responsible for exploratory analysis, visualization, preprocessing, classification, model comparison, hyperparameter tuning, and fare regression.

### Support Assistant

Responsible for policy-document ingestion, embedding, vector retrieval, intent routing, response validation, and API serving.

This modular design keeps the scraping, analytics, and support-assistant workflows separated while allowing the complete project to be executed from a single repository.
