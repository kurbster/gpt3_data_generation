import sys
import logging
import functools

from typing import Type
from pathlib import Path
from dataclasses import dataclass

MAIN_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(MAIN_DIR))

import datasets
import transformers

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

logger = logging.getLogger("myLogger")

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

# If we start using casual LM models put them here
decoder_models = []

@dataclass
class ExperimentType:
    model_name: str
    is_encoder_model: bool = False
    is_enc_dec_model: bool = False
    is_decoder_model: bool = False

    def __post_init__(self):
        if self.model_name in encoder_models:
            self.is_encoder_model = True
        elif self.model_name in enc_dec_models:
            self.is_enc_dec_model = True
        elif self.model_name in decoder_models:
            self.is_decoder_model = True
        else:
            raise ValueError(f"The model name '{self.model_name}' needs to be classified in the util.py file.")
    
    @property
    def tokenize_labels(self) -> bool:
        return False if self.is_encoder_model else True

    @property
    def model_type(self) -> Type:
        return AutoModelForSequenceClassification if self.is_encoder_model else AutoModelForSeq2SeqLM

    @property
    def trainer_type(self) -> Type:
        return LogMetricsTrainer if self.is_encoder_model else LogMetricsSeq2SeqTrainer

    @property
    def data_collator(self) -> Type:
        return DataCollatorWithPadding if self.is_encoder_model else DataCollatorForSeq2Seq
    
    def get_data_collator(self, model):
        if self.is_encoder_model:
            return DataCollatorWithPadding
        return functools.partial(DataCollatorForSeq2Seq, model=model)
