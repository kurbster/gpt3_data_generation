import sys
import logging

from typing import Type
from pathlib import Path

MAIN_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(MAIN_DIR))

import datasets
import transformers

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from src.custom_trainer import (
    LogMetricsTrainer,
    LogMetricsSeq2SeqTrainer
)

logger = logging.getLogger("myLogger")

model_types = {
    "roberta-base": AutoModelForSequenceClassification,
    "bert-base-cased": AutoModelForSequenceClassification,
    "allenai/tk-instruct-3b-pos": AutoModelForSeq2SeqLM,
    "facebook/bart-base": AutoModelForSeq2SeqLM,
}

trainer_types = {
    "roberta-base": LogMetricsTrainer,
    "bert-base-cased": LogMetricsTrainer,
    "allenai/tk-instruct-3b-pos": LogMetricsSeq2SeqTrainer,
    "facebook/bart-base": LogMetricsSeq2SeqTrainer,
}

def setup_logging(training_args):
    # log_level = training_args.get_process_log_level()
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

def get_auto_model_type(model_name: str) -> Type:
    return model_types[model_name]

def get_trainer_type(model_name: str) -> Type:
    return trainer_types[model_name]