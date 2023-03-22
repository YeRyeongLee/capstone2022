import logging

from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)



labels = ['adhd', 'anxiety', 'bipolar', 'depression', 'non_mh', 'schizo']

label2id = {labels[i]:i for i in range(6)}
id2label = {i:labels[i] for i in range(6)}



def micro_f1(preds, labels):
    return f1_score(labels, preds, average="micro", labels=list(range(len(label2id)))) * 100.0



def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = micro_f1(preds, labels)
    acc = accuracy_score(labels, preds)

    return {
        'micro f1 score': f1,
        'accuracy': acc
    }