import logging

from typing import Type

import datasets
import transformers

from transformers import (
    AutoModelForSequenceClassification,
)

logger = logging.getLogger("myLogger")

model_types = {
    "bert-base-cased": AutoModelForSequenceClassification,
    "roberta-base": AutoModelForSequenceClassification,
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