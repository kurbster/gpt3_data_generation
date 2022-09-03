from pathlib import Path
from typing import NamedTuple, List

import torch
import pandas as pd

from sklearn.metrics import classification_report

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    preds = torch.from_numpy(preds)
    y_pred_softmax = torch.log_softmax(preds, dim=1)
    _, y_pred = torch.max(y_pred_softmax, dim=1)

    results = classification_report(labels, y_pred, output_dict=True, zero_division=0)
    flattened_results = dict()

    for key, value in results.items():
        # cls report returns a report for every label.
        # Ignore them and only get the overall results
        if isinstance(value, dict):
            for k, v in value.items():
                flattened_results[f"{key}_{k}"] = v
        else:
            flattened_results[key] = value

    return flattened_results
    
def compute_record_metrics(eval_preds):
    preds, labels = eval_preds

    preds = torch.from_numpy(preds)
    y_pred_softmax = torch.log_softmax(preds, dim=1)
    _, y_pred = torch.max(y_pred_softmax, dim=1)

    results = classification_report(labels, y_pred, output_dict=True, zero_division=0)
    flattened_results = dict()

    for key, value in results.items():
        # cls report returns a report for every label.
        # Ignore them an only get the overall results
        if isinstance(value, dict):
            for k, v in value.items():
                flattened_results[f"{key}_{k}"] = v
        else:
            flattened_results[key] = value

    num_correct = (y_pred == 1) & (labels == 1)
    flattened_results['raw_accuracy'] = num_correct.sum().item() / (labels == 1).sum().item()

    return flattened_results

def prediction_metrics(
    predict_results: NamedTuple,
    output_dir: Path,
    input_text: List[str]
):
    preds = torch.from_numpy(predict_results.predictions)
    labels = torch.from_numpy(predict_results.label_ids)

    y_pred_softmax = torch.log_softmax(preds, dim=1)
    _, y_pred = torch.max(y_pred_softmax, dim=1)

    # Output the generated predictions
    predictions = list(map(lambda x: str(x.item()), y_pred))
    with open(output_dir / "generated_predictions.txt", "w") as writer:
        writer.write("\n".join(predictions))

    # Output the labels, predictions, and input text
    df = pd.DataFrame()
    df["labels"] = labels
    df["predictions"] = predictions
    df["text"] = input_text
    df.to_csv(output_dir / "prediction_list.csv")

    results = classification_report(labels, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(results).transpose()
    df.to_csv(output_dir / "performance.csv")