import torch
import pandas as pd
import numpy as np



# load data
# data should include text and label(0~5)
path = "<yourpath>"
data_files = {
    'train': path + 'train.csv',
    'valid': path + 'valid.csv',
    'test': path + 'test.csv',
}
labels = ['adhd', 'anxiety', 'bipolar', 'depression', 'non_mh', 'schizo']


from datasets import load_dataset

mental_health = load_dataset("csv", data_files=data_files)



####################
# model_ckpt: name of a model you want to load from huggingface
model_ckpt = 'bert-base-uncased'
####################


# tokenizer
from transformers import AutoTokenizer
# autotokenizer: it can easily switch different models
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)



def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)


mental_encoded = mental_health.map(tokenize, batched=True, batch_size=8)




# extract last hidden state from pre-trained model
from transformers import AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(model_ckpt).to(device)




def extract_hidden_states(batch):
  inputs = {k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

  with torch.no_grad():
    last_hidden_state = model(**inputs).last_hidden_state

  return {'hidden_state': last_hidden_state[:, 0].cpu().numpy()}


mental_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
emotions_hidden = mental_encoded.map(extract_hidden_states, batched=True, batch_size=8)




# for logistic regression
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["valid"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["valid"]["label"])
X_test = np.array(emotions_hidden["test"]["hidden_state"])
y_test = np.array(emotions_hidden["test"]["label"])



"""
# visualization

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

X_scaled = MinMaxScaler().fit_transform(X_train)

mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb['label'] = y_train
df_emb.head()


import matplotlib.pyplot as plt
%matplotlib inline

fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = ['adhd', 'anxiety', 'bipolar', 'depression', 'non_mh', 'schizo']

for i, (label, cmap) in enumerate(zip(labels, cmaps)):
  df_emb_sub = df_emb.query(f"label == {i}")
  axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                 gridsize=20, linewidths=(0,))
  axes[i].set_title(label)
  axes[i].set_xticks([]), axes[i].set_yticks([])

plt.tight_layout()
plt.show()

"""




from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(max_iter=3000)
logistic.fit(X_train, y_train)
logistic.score(X_valid, y_valid)


# for plotting confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Confusion matrix")
    plt.show()


# training set
print(f'acc = {logistic.score(X_train, y_train)*100}%')
plot_confusion_matrix(logistic.predict(X_train), y_train, labels)

# test set
print(f'acc = {logistic.score(X_test, y_test)*100}%')
plot_confusion_matrix(logistic.predict(X_test), y_test, labels)


"""
# make classifier with SVM
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
svc.score(X_valid, y_valid)
"""



# fine-tuning 

from transformers import AutoModelForSequenceClassification

num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average='weighted')
  acc = accuracy_score(labels, preds)
  recall = recall_score(labels, preds, average='weighted')
  precision = precision_score(labels, preds, average='weigted')
  return {"accuracy": acc, "f1": f1, "recall": recall, "precision": precision}


from transformers import Trainer, TrainingArguments

batch_size = 8
logging_steps = len(mental_encoded["train"])
model_name = f"{model_ckpt}-finetuned-mental"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=10,
                                  learning_rate=5e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy='epoch',
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=True,
                                  load_best_model_at_end=True,
                                  log_level='error')

from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=mental_encoded["train"],
                  eval_dataset=mental_encoded["valid"],
                  tokenizer=tokenizer)
trainer.train();


# training set
pred_output = trainer.predict(mental_encoded["train"])
y_pred = np.argmax(pred_output.predictions, axis=1)
acc = pred_output.metrics['test_accuracy']

print(f'acc = {acc}%')
plot_confusion_matrix(y_pred, y_train, labels)


# test set
preds_output = trainer.predict(mental_encoded["test"])
y_preds = np.argmax(preds_output.predictions, axis=1)
acc = preds_output.metrics['test_accuracy']

print(f'acc = {acc}%')
plot_confusion_matrix(y_preds, y_test, labels)

