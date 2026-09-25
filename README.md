\# Zepto AI Platform



A complete AI/data analytics platform containing a web-scraping data pipeline, Titanic analytics and machine-learning models, and a policy-based customer support assistant.



\## Project Structure



```text

zepto-ai-platform/

│

├── data\_pipeline/

│   ├── data\_pipeline.py

│   ├── books\_raw.csv

│   ├── books\_clean.csv

│   └── books.db

│

├── analytics/

│   ├── 01\_eda.py

│   ├── 02\_modeling.py

│   ├── titanic.csv

│   ├── titanic\_cleaned.csv

│   ├── decision\_tree.png

│   ├── regression\_residuals.png

│   ├── pipeline.joblib

│   └── charts/

│

├── support\_assistant/

│   ├── assistant.py

│   ├── Dockerfile

│   ├── README.md

│   ├── docs/

│   │   ├── doc\_01.txt

│   │   ├── doc\_02.txt

│   │   ├── doc\_03.txt

│   │   ├── doc\_04.txt

│   │   ├── doc\_05.txt

│   │   ├── doc\_06.txt

│   │   ├── doc\_07.txt

│   │   └── doc\_08.txt

│   └── chroma\_db/

│

├── requirements.txt

└── README.md

```



\## Installation



Create and activate a Python virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install the project dependencies:



```powershell

pip install -r requirements.txt

```



The main dependencies include Requests, BeautifulSoup, Pandas, NumPy, Scikit-learn, imbalanced-learn, Seaborn, Matplotlib, Joblib, ChromaDB, Sentence Transformers, LangGraph, FastAPI, Uvicorn, and Pydantic.



\## 1. Data Pipeline



The data pipeline is implemented in:



```text

data\_pipeline/data\_pipeline.py

```



It scrapes book information using `requests` and `BeautifulSoup`.



The pipeline collects at least 60 books from multiple categories and stores the following raw fields:



\* title

\* price

\* star\_rating

\* availability

\* category



The cleaned dataset contains:



\* title

\* category

\* price\_gbp

\* rating

\* in\_stock

\* price\_inr



Star ratings are converted from words such as `One`, `Two`, `Three`, `Four`, and `Five` into numeric values from 1 to 5.



Availability is converted into a Boolean in-stock field.



The fixed exchange rate used for conversion is:



```text

1 GBP = 105.50 INR

```



The pipeline also creates a normalized SQLite database containing categories and books with a foreign-key relationship.



Example SQL operations demonstrate:



\* SELECT

\* WHERE

\* ORDER BY

\* LIMIT

\* DISTINCT

\* IN

\* BETWEEN

\* JOIN



SQL results are reproduced using both `pandas.read\_sql()` and `pandas.merge()`.



\### Run the Data Pipeline



```powershell

python .\\data\_pipeline\\data\_pipeline.py

```



Outputs:



```text

data\_pipeline/books\_raw.csv

data\_pipeline/books\_clean.csv

data\_pipeline/books.db

```



\## 2. Analytics



The analytics module contains two scripts.



```text

analytics/01\_eda.py

analytics/02\_modeling.py

```



\### Exploratory Data Analysis



`01\_eda.py` loads the Titanic dataset, saves a local copy, and performs exploratory data analysis.



The analysis includes:



\* dataset information

\* descriptive statistics

\* dataset shape

\* missing-value percentages

\* missing-value treatment

\* age and fare histograms

\* age and fare boxplots

\* IQR outlier counts

\* fare mean, median, mode, and skewness

\* survival by sex

\* survival by passenger class

\* survival by sex and passenger class

\* a 6x6 correlation analysis

\* correlation heatmap

\* strongest absolute correlations

\* age and fare standardization

\* multivariate visualizations



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

python .\\analytics\\01\_eda.py

```



Outputs include:



```text

analytics/titanic.csv

analytics/titanic\_cleaned.csv

analytics/charts/

```



\### Machine Learning Modeling



`02\_modeling.py` performs classification and regression.



The classification workflow includes:



\* stratified train/test split

\* preprocessing with imputation, scaling, and one-hot encoding

\* Logistic Regression

\* Decision Tree

\* Random Forest

\* accuracy

\* precision

\* recall

\* F1 score

\* ROC AUC

\* confusion matrices

\* decision-tree visualization

\* class-weight-balanced models

\* SMOTE comparison

\* Random Forest hyperparameter tuning



The Random Forest grid search evaluates:



```text

n\_estimators

max\_depth

max\_features

```



SMOTE is applied only to the training data through the modeling pipeline.



The final Random Forest uses out-of-bag evaluation with:



```text

oob\_score=True

```



The regression task predicts passenger fare and reports:



\* MAE

\* RMSE

\* R²

\* Adjusted R²

\* residual visualization

\* heteroscedasticity assessment



The complete fitted preprocessing and Random Forest pipeline is saved with Joblib and subsequently reloaded to demonstrate prediction from raw input.



Run:



```powershell

python .\\analytics\\02\_modeling.py

```



Important generated outputs include:



```text

analytics/decision\_tree.png

analytics/regression\_residuals.png

analytics/pipeline.joblib

```



\## 3. Support Assistant



The support assistant is implemented in:



```text

support\_assistant/assistant.py

```



It provides a FastAPI customer-support API backed by eight policy documents.



The policy documents are:



```text

doc\_01.txt

doc\_02.txt

doc\_03.txt

doc\_04.txt

doc\_05.txt

doc\_06.txt

doc\_07.txt

doc\_08.txt

```



The system uses:



\* `all-MiniLM-L6-v2` embeddings

\* ChromaDB

\* LangGraph

\* TypedDict graph state

\* Pydantic response validation

\* FastAPI

\* Uvicorn



\### LangGraph Flow



The graph contains the following named nodes:



```text

classify\_intent

retrieve\_and\_answer

direct\_answer

```



The flow is:



```text

User Query

&#x20;   |

&#x20;   v

classify\_intent

&#x20;   |

&#x20;   +----------------------+

&#x20;   |                      |

&#x20;   v                      v

policy\_question       general\_question

&#x20;   |                      |

&#x20;   v                      v

retrieve\_and\_answer   direct\_answer

&#x20;   |                      |

&#x20;   +----------+-----------+

&#x20;              |

&#x20;             END

```



Policy questions retrieve the top three relevant policy chunks from ChromaDB.



General questions use the direct-answer route.



\### Structured Prompt



The policy-answering prompt contains:



\* Role

\* Context

\* Task

\* Format

\* Length

\* Negative constraint

\* Few-shot example



The negative constraint prevents the assistant from inventing policy information outside the retrieved context.



\### Response Schema



API responses follow the Pydantic structure:



```json

{

&#x20; "answer": "string",

&#x20; "sources": \[],

&#x20; "confidence": 0.9

}

```



`confidence` is constrained to the range 0 to 1.



General questions return an empty `sources` list.



\### Mock LLM Mode



The application supports:



```text

MOCK\_LLM=1

```



This mode does not require an external LLM or external network call for answer generation.



On Windows PowerShell:



```powershell

$env:MOCK\_LLM="1"

python .\\support\_assistant\\assistant.py

```



The API runs on:



```text

http://127.0.0.1:7860

```



\### API Endpoint



The application exposes:



```text

POST /ask

```



Example request:



```json

{

&#x20; "query": "What is the delivery fee for orders under INR 500?"

}

```



Example response:



```json

{

&#x20; "answer": "Based on the provided policy context: ...",

&#x20; "sources": \[

&#x20;   "doc\_01\_chunk\_0",

&#x20;   "doc\_03\_chunk\_0",

&#x20;   "doc\_07\_chunk\_0"

&#x20; ],

&#x20; "confidence": 0.9

}

```



Example PowerShell request:



```powershell

Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the delivery fee for orders under INR 500?"}'

```



\## 4. Docker



The support assistant includes a Dockerfile.



Build the image from the repository root:



```powershell

docker build -f .\\support\_assistant\\Dockerfile -t zepto-support-assistant .

```



Run the container:



```powershell

docker run -p 7860:7860 zepto-support-assistant

```



The API is then available at:



```text

http://127.0.0.1:7860

```



\## 5. Generated Outputs



\### Data Pipeline



```text

data\_pipeline/books\_raw.csv

data\_pipeline/books\_clean.csv

data\_pipeline/books.db

```



\### Analytics



```text

analytics/titanic.csv

analytics/titanic\_cleaned.csv

analytics/decision\_tree.png

analytics/regression\_residuals.png

analytics/pipeline.joblib

analytics/charts/

```



\### Support Assistant



```text

support\_assistant/chroma\_db/

```



The ChromaDB directory stores the local vector database used for policy retrieval.



\## 6. End-to-End Execution



Install dependencies:



```powershell

pip install -r requirements.txt

```



Run the data pipeline:



```powershell

python .\\data\_pipeline\\data\_pipeline.py

```



Run exploratory analysis:



```powershell

python .\\analytics\\01\_eda.py

```



Run modeling:



```powershell

python .\\analytics\\02\_modeling.py

```



Run the support assistant:



```powershell

$env:MOCK\_LLM="1"

python .\\support\_assistant\\assistant.py

```



\## 7. Design Summary



The project is divided into three independent modules.



\### Data Pipeline



Responsible for acquiring, cleaning, transforming, validating, and storing book data.



\### Analytics



Responsible for exploratory analysis, visualization, preprocessing, classification, model comparison, hyperparameter tuning, and fare regression.



\### Support Assistant



Responsible for policy-document ingestion, embedding, vector retrieval, intent routing, response validation, and API serving.



This modular design keeps the scraping, analytics, and support-assistant workflows separated while allowing the complete project to be executed from a single repository.



