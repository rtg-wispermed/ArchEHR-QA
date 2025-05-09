from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class RelevanceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class RelevanceClassifier:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.label_map = {"essential": 0, "supplementary": 1, "not-relevant": 2}
        self.id2label = {v: k for k, v in self.label_map.items()}

        # Check if CUDA is available and move model to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

    def prepare_data(self, df):
        # Filter out rows with unknown relevance
        df = df[df['relevance'] != 'unknown'].copy()

        # Create input text by combining question and sentence
        df['input_text'] = df['question'] + " [SEP] " + df['sentence_text']

        # Map labels to integers
        df['label'] = df['relevance'].map(self.label_map)

        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Tokenize
        train_encodings = self.tokenizer(train_df['input_text'].tolist(), truncation=True, padding=True)
        val_encodings = self.tokenizer(val_df['input_text'].tolist(), truncation=True, padding=True)

        # Create datasets
        train_dataset = RelevanceDataset(train_encodings, train_df['label'].tolist())
        val_dataset = RelevanceDataset(val_encodings, val_df['label'].tolist())

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, output_dir="./relevance_model"):
        print(len(train_dataset))
        print(len(val_dataset))
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Get final evaluation metrics
        final_metrics = trainer.evaluate()

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return final_metrics

    def predict(self, questions, sentences):
        inputs = [q + " [SEP] " + s for q, s in zip(questions, sentences)]
        encodings = self.tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

        # Move input tensors to the same device as the model
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1).tolist()

        return [self.id2label[p] for p in predicted_classes], predictions.cpu().tolist()
