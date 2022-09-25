import sys
import functools

from typing import Type
from pathlib import Path

MAIN_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(MAIN_DIR))

import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)

from src.custom_trainer import (
    LogMetricsTrainer,
    LogMetricsSeq2SeqTrainer
)

encoder_models = ["roberta-base", "bert-base-cased"]

enc_dec_models = [
    "allenai/tk-instruct-3b-pos",
    "allenai/tk-instruct-3b-def",
    "allenai/tk-instruct-3b-def-pos",
    "allenai/tk-instruct-base-pos",
    "allenai/tk-instruct-base-def",
    "allenai/tk-instruct-base-def-pos",
    "facebook/bart-base",
]

class ExperimentType:
    is_generative_model: bool = False

    def __init__(self, model_name):
        if model_name in enc_dec_models:
            self.is_generative_model = True
        elif model_name not in encoder_models:
            raise ValueError(f"The model name '{model_name}' needs to be classified in the util.py file.")
    
    @property
    def model_type(self) -> Type:
        return AutoModelForSeq2SeqLM if self.is_generative_model else AutoModelForSequenceClassification

    @property
    def trainer_type(self) -> Type:
        return LogMetricsSeq2SeqTrainer if self.is_generative_model else LogMetricsTrainer

    def get_data_collator(self, model):
        if self.is_generative_model:
            return functools.partial(DataCollatorForSeq2Seq, model=model)
        return DataCollatorWithPadding

    def generative_model_postprocess(self, preds, labels, tokenizer):
        predictions = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return predictions, labels

    def discriminative_postprocess(self, preds, labels):
        preds = torch.from_numpy(preds)
        y_pred_softmax = torch.log_softmax(preds, dim=1)
        _, y_pred = torch.max(y_pred_softmax, dim=1)
        
        # We need to map the labels and predictions to strings
        labels = list(map(str, labels))
        predictions = list(map(lambda x: str(x.item()), y_pred))
        return predictions, labels

    def postprocess_output(self, preds, labels, tokenizer):
        # If the model is generative then postprocess the
        # predictions and labels with tokenizer decoding
        if self.is_generative_model:
            preds, labels = self.generative_model_postprocess(
                preds, labels, tokenizer
            )
        # If the model is discriminative then only postprocess
        # the predictions from logits to class labels
        else:
            preds, labels = self.discriminative_postprocess(preds, labels)
        return preds, labels

    def eval_postprocess_output(self, func):
        def wrapper(eval_results, tokenizer):
            preds, labels = eval_results
            preds, labels = self.postprocess_output(preds, labels, tokenizer)
            return func(preds, labels)
        return wrapper

    def predict_postprocess_output(self, func):
        def wrapper(predict_results, output_dir, input_text, tokenizer):
            preds, labels = predict_results.predictions, predict_results.label_ids
            preds, labels = self.postprocess_output(preds, labels, tokenizer)
            return func(preds, labels, output_dir, input_text)
        return wrapper