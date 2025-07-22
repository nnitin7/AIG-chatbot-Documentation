# AIG AI-Powered Chatbot: LLM Fine-Tuning and Scalable Deployment

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Role & Impact
As Machine Learning Engineer at AIG (07/2023 - Present), I engineered a conversational AI chatbot using fine-tuned GPT-3 and BERT with RAG via LangChain, deployed on AWS for 10,000+ monthly queries. Key contributions: NLP for text classification/NER, prompt engineering to overcome latency, and microservices for scalability. Results: 95% F1-score, 27% response time reduction, 12% customer satisfaction increase, and 550+ agent hours saved annually.



## Technical Workflow
1. **Data Preprocessing**: PySpark for scalable cleaning of 1M+ records.
2. **Fine-Tuning**: Hugging Face with k-fold CV on SageMaker.
3. **Deployment**: FastAPI async APIs with CI/CD (Jenkins/Docker).
4. **Optimization**: Model pruning and custom dashboards for monitoring.

## Code Highlights

### Python: BERT Fine-Tuning with Hugging Face
```python
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold

data = pd.read_csv('interactions.csv').dropna(subset=['text'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(data):
    train_enc = tokenizer(data.iloc[train_idx]['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    args = TrainingArguments(output_dir='./results', num_train_epochs=3, learning_rate=2e-5)
    trainer = Trainer(model=model, args=args, train_dataset=train_enc)
    trainer.train()
model.save_pretrained('./fine_tuned')
Technical Workflow
Data Preprocessing: PySpark for scalable cleaning of 1M+ records.
Fine-Tuning: Hugging Face with k-fold CV on SageMaker.
Deployment: FastAPI async APIs with CI/CD (Jenkins/Docker).
Optimization: Model pruning and custom dashboards for monitoring.
Code Highlights
Python: BERT Fine-Tuning with Hugging Face
python

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold

data = pd.read_csv('interactions.csv').dropna(subset=['text'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(data):
    train_enc = tokenizer(data.iloc[train_idx]['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    args = TrainingArguments(output_dir='./results', num_train_epochs=3, learning_rate=2e-5)
    trainer = Trainer(model=model, args=args, train_dataset=train_enc)
    trainer.train()
model.save_pretrained('./fine_tuned')

SQL: Metrics Aggregation for Validation

SELECT user_id, AVG(latency) OVER (PARTITION BY query_type ORDER BY timestamp) AS rolling_avg
FROM queries
WHERE date > '2023-07-01' AND accuracy >= 0.9
GROUP BY user_id
HAVING COUNT(*) > 50;

PySpark: Scalable Embedding Generation


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet('s3://data')
df = df.filter(col('text').isNotNull()).repartition('user_id')
df = df.withColumn('embedding', udf_embed(col('text')))
df.write.parquet('s3://embeddings')
Demo
Run python fine_tune.py for training.
API endpoint: uvicorn app:app --port 8000.
Full code in /src.
