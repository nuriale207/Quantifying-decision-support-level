import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction

def multi_label_metrics(predictions, labels, threshold=0.35):
    print(predictions)
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    print(y_pred[0])
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    metrics=multi_label_metrics_binary(y_true, y_pred)
    print(metrics)
    return metrics

def multi_label_metrics_binary(y_true,y_pred):
    precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision_macro_average = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision_weighted_average = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    precision_samples_average = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
    precision_per_class = precision_score(y_true=y_true, y_pred=y_pred, average=None)

    recall_micro_average = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro_average = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_weighted_average = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_samples_average = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
    recall_per_class = recall_score(y_true=y_true, y_pred=y_pred, average=None)

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_samples_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None)

    # In an article they propose
    optimized_f1 = np.mean(f1_per_class[np.where(f1_per_class > 0)])
    optimized_recall = np.mean(recall_per_class[np.where(recall_per_class > 0)])
    optimized_precision = np.mean(precision_per_class[np.where(precision_per_class > 0)])
    # roc_auc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
    # roc_auc_macro = roc_auc_score(y_true, y_pred, average = 'macro')
    # roc_auc_weighted = roc_auc_score(y_true, y_pred, average = 'weighted')
    # roc_auc_samples = roc_auc_score(y_true, y_pred, average = 'samples')

    accuracy = accuracy_score(y_true, y_pred)

    # return a dictionary with all metrics
    metrics = {'precision_micro': precision_micro_average,
               'precision_macro': precision_macro_average,
               'optimized_macro_precision': optimized_precision,
               'precision_weighted': precision_weighted_average,
               'precision_samples': precision_samples_average,

               'recall_micro': recall_micro_average,
               'recall_macro': recall_macro_average,
               'optimized_macro_recall': optimized_recall,

               'recall_weighted': recall_weighted_average,
               'recall_samples': recall_samples_average,
               'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_weighted': f1_weighted_average,
               'f1_samples': f1_samples_average,
               'optimized_macro_f1': optimized_f1,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
        tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    print(result)
    return result

def get_predictions_ids(predictions, id2label):
    # Convert PyTorch tensor to a NumPy array
    binary_predictions_np = predictions.cpu().numpy()

    # Initialize a list to store the label names for each instance
    predictions_as_labels = []

    # Iterate over each instance's binary predictions
    for instance_predictions in binary_predictions_np:
        # Find the indices where the values are 1
        label_indices = np.where(instance_predictions == 1)[0]
        # Map the label indices to label names using id2label dictionary
        labels = [id2label[index] for index in label_indices]

        # Append the list of label names for this instance to the result list
        predictions_as_labels.append(labels)
    return predictions_as_labels

def get_predictions_by_hierarchy(hierarchy_1_preds, hierarchy_2_preds):
    new_main_preds=[]
    for i in range(len(hierarchy_1_preds)):
        chapter_pred=set(hierarchy_1_preds[i])
        main_pred=set(hierarchy_2_preds[i])
        main_preds_filtered=[]
        for pred in main_pred:
            chapter=pred[:1]
            if chapter in chapter_pred:
                main_preds_filtered.append(pred)
        new_main_preds.append(main_preds_filtered)
    return new_main_preds
def get_binary_predictions(predictions,label2id):
    # Determine the number of labels
    num_labels = len(label2id)

    # Initialize an empty NumPy array with zeros
    binary_predictions_np = np.zeros((len(predictions), num_labels), dtype=np.int)

    # Iterate over each instance's label predictions
    for i, instance_labels in enumerate(predictions):
        for label in instance_labels:
            # Check if the label exists in the label2id dictionary
            if label in label2id:
                # Set the corresponding column to 1 for the label
                label_index = label2id[label]
                binary_predictions_np[i, label_index] = 1

    # Convert the NumPy array to a PyTorch tensor if needed
    binary_predictions = torch.from_numpy(binary_predictions_np)

    return binary_predictions
def get_hierarchical_predictions(pred_full, pred_main, pred_chapter, id2label_full, id2label_main,
                                 id2label_chapter, threshold=0.2):
    sigmoid = torch.nn.Sigmoid()

    Y_pred_full = sigmoid(torch.Tensor(pred_full))
    Y_pred_main = sigmoid(torch.Tensor(pred_main))
    Y_pred_chapter = sigmoid(torch.Tensor(pred_chapter))

    Y_pred_chapter[np.where(Y_pred_chapter >= threshold)] = 1
    Y_pred_chapter[np.where(Y_pred_chapter < threshold)] = 0

    Y_pred_main[np.where(Y_pred_main >= threshold)] = 1
    Y_pred_main[np.where(Y_pred_main < threshold)] = 0

    Y_pred_full[np.where(Y_pred_full >= threshold)] = 1
    Y_pred_full[np.where(Y_pred_full < threshold)] = 0
    preds_ids_full=get_predictions_ids(Y_pred_full, id2label_full)

    preds_ids_chapter=get_predictions_ids(Y_pred_chapter, id2label_chapter)

    preds_ids_main=get_predictions_ids(Y_pred_main, id2label_main)

    new_main_preds = get_predictions_by_hierarchy(preds_ids_chapter, preds_ids_main)
    new_full_preds = get_predictions_by_hierarchy(new_main_preds, preds_ids_full)

    # new_main_preds_bin = get_binary_predictions(new_main_preds, label2id_main)
    # new_full_preds_bin = get_binary_predictions(new_full_preds, label2id_full)

    return preds_ids_chapter,new_main_preds,new_full_preds