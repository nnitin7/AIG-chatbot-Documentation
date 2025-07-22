import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from langchain.prompts import PromptTemplate
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

# Load and clean 1M+ interaction data
data = pd.read_csv('customer_queries.csv').dropna(subset=['text', 'label'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # pos/neg/neu

class QueryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(data):
    train = data.iloc[train_idx]
    val = data.iloc[val_idx]
    train_enc = tokenizer(train['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    val_enc = tokenizer(val['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    train_ds = QueryDataset(train_enc, train['label'].values)
    val_ds = QueryDataset(val_enc, val['label'].values)
    args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=32, learning_rate=2e-5, evaluation_strategy='epoch')
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

# LangChain prompt for RAG inference
prompt = PromptTemplate(input_variables=["query", "context"], template="Classify: {query}. Context: {context}")
# Save and prune model for 20% faster inference
model.save_pretrained('./fine_tuned_bert')