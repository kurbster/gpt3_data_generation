from typing import List
from pathlib import Path

import pandas as pd

from sklearn.metrics import classification_report

def compute_f1_metrics(preds: List[str], labels: List[str]):
    results = classification_report(labels, preds, output_dict=True, zero_division=0)
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

def compute_f1_prediction_metrics(
    preds: List[str],
    labels: List[str],
    output_dir: Path,
    input_text: List[str]
):
    with open(output_dir / "generated_predictions.txt", "w") as writer:
        writer.write("\n".join(preds))

    # Output the labels, predictions, and input text
    df = pd.DataFrame()
    df["labels"] = labels
    df["predictions"] = preds
    df["text"] = input_text
    df.to_csv(output_dir / "prediction_list.csv")

    results = classification_report(labels, preds, output_dict=True, zero_division=0)
    df = pd.DataFrame(results).transpose()
    df.to_csv(output_dir / "performance.csv")
    
def compute_record_f1_metrics(preds: List[str], labels: List[str]):
    flattened_results = compute_f1_metrics(preds, labels)

    # Add extra raw accuracy for ReCORD
    num_correct = (preds == '1') & (labels == '1')
    flattened_results['raw_accuracy'] = num_correct.sum().item() / (labels == '1').sum().item()

    return flattened_results

def compute_record_f1_prediction_metrics(
    preds: List[str],
    labels: List[str],
    output_dir: Path,
    input_text: List[str]
):
    with open(output_dir / "generated_predictions.txt", "w") as writer:
        writer.write("\n".join(preds))

    # Output the labels, predictions, and input text
    df = pd.DataFrame()
    df["labels"] = labels
    df["predictions"] = preds
    df["text"] = input_text
    df.to_csv(output_dir / "prediction_list.csv")

    results = classification_report(labels, preds, output_dict=True, zero_division=0)
    # Add extra raw accuracy for ReCORD
    num_correct = (preds == '1') & (labels == '1')
    results['raw_accuracy'] = num_correct.sum().item() / (labels == '1').sum().item()

    df = pd.DataFrame(results).transpose()
    df.to_csv(output_dir / "performance.csv")