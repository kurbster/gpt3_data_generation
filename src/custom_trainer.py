import json
import logging

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


logger =  logging.getLogger("transformers.trainer")
eval_logger = logging.getLogger("evaluation")

class LogCallBack(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            logger.debug(f'I am the logs: {logs}')
            if "eval_accuracy" in logs:
                eval_output_path = Path(args.output_dir, 'evaluation_results')
                epoch = 1 if state.epoch is None else int(state.epoch)
                fname = eval_output_path / f'eval_results_epoch{epoch}.json'
                with open(fname, 'w') as f:
                    json.dump(logs, f, indent=4)
                eval_logger.info(f'epoch: {epoch} eval acc: {logs["eval_accuracy"]} eval loss: {logs["eval_loss"]}')
                # This was used for the old style of ReCoRD evaluation
                # eval_logger.info(f'epoch: {epoch} eval raw acc: {logs["eval_raw_accuracy"]} eval acc: {logs["eval_accuracy"]} eval loss: {logs["eval_loss"]}')

class LogMetricsTrainer(Trainer):
    def log_metrics(self, split, metrics):
        if not self.is_world_process_zero():
            return

        logger.info(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        flattened_features = [
            [
                {k: v[i] for k, v in feature.items()}
                for i in range(len(feature["input_ids"]))
            ]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = sum(labels, [])
        # batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int8)
        return batch

@dataclass
class OldDataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
