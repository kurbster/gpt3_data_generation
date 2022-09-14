#!/usr/bin/env python3
import os
import sys
import logging

from pathlib import Path

MAIN_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(MAIN_DIR))

import hydra

from datasets import load_dataset, dataset_dict
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.config import ModelArguments, DataTrainingArguments
from src.run_experiment import run_model

logger = logging.getLogger("myLogger")

def get_original_datasets(data_args: DataTrainingArguments, model_args: ModelArguments) -> dataset_dict:
    datasets = load_dataset(
        'super_glue',
        data_args.dataset_name,
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    # Set the validation as the test set first before modifying it
    datasets['test'] = datasets[data_args.test_set_key]

    # Select from the end of the train set until the max eval samples
    datasets['validation'] = datasets['train'].select(
        range(data_args.max_train_samples, data_args.max_train_samples + data_args.max_eval_samples)
    )

    logger.info(f'Len of Original train after {datasets["train"].num_rows}')
    datasets['train'] = datasets['train'].select(range(data_args.max_train_samples))
    logger.info(f'Len of Original train after {datasets["train"].num_rows}')
    logger.info(f'Len of Validation set: {datasets["validation"].num_rows}')
    logger.info(f'Len of Test set: {datasets["test"].num_rows}')

    return datasets

def get_generated_dataset(train_file: Path, data_args: DataTrainingArguments, model_args: ModelArguments) -> dataset_dict:
    dataset = load_dataset(
        train_file.suffix.strip('.'),
        data_files=str(train_file),
        field="data",
        split="train",
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    # Ensure max value is the length of the dataset
    num_generated_samples = min(data_args.max_train_samples, dataset.num_rows)
    dataset = dataset.select(range(num_generated_samples))
    
    return dataset

@hydra.main(config_path="../conf", config_name="wic", version_base="1.2")
def main(cfg):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(cfg)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    set_seed(training_args.seed)

    main_output_dir = Path(training_args.output_dir)
    logger.info(f"Output directory: {main_output_dir}")
    
    # These are the original train, validation, and test
    datasets = get_original_datasets(data_args, model_args)

    # Get the original and generated preprocessing functions.
    original_preprocessing = hydra.utils.instantiate(
        cfg.preprocessing_config.original,
        # By including these values we override whatever is in the config
        #text_col=data_args.text_column,
        #label_col=data_args.label_column,
    )
    generated_preprocessing = hydra.utils.instantiate(
        cfg.preprocessing_config.generated,
        # By including these values we override whatever is in the config
        #text_col=data_args.text_column,
        #label_col=data_args.label_column,
    )

    # Get the metric function
    metric_function = hydra.utils.instantiate(cfg.metric)
    predict_metric_func = hydra.utils.instantiate(cfg.prediction_metric)

    if data_args.run_original:
        training_args.output_dir = str(main_output_dir / 'original')
        logger.info('STARTING THE MODEL WITH THE ORIGINAL TRAIN DATA')
        logger.info('*'*75)
        run_model(
            model_args,
            data_args,
            training_args,
            datasets,
            metric_func=metric_function,
            predict_metric_func=predict_metric_func,
            train_preprocessing_func=original_preprocessing,
            test_preprocessing_func=original_preprocessing
        )

    if len(data_args.train_files) > 0:
        for train_file in data_args.train_files:
            train_file = MAIN_DIR / train_file
            training_args.output_dir = str(main_output_dir / train_file.stem)

            datasets["train"] = get_generated_dataset(train_file, data_args, model_args)

            run_model(
                model_args,
                data_args,
                training_args,
                datasets,
                metric_func=metric_function,
                predict_metric_func=predict_metric_func,
                train_preprocessing_func=generated_preprocessing,
                test_preprocessing_func=original_preprocessing
            )

    logger.info('STARTING THE MODEL WITHOUT TRAINING')
    logger.info('*'*75)
    training_args.output_dir = str(main_output_dir / 'no_train')
    training_args.do_train = False
    training_args.do_eval = False
    run_model(
        model_args,
        data_args,
        training_args,
        datasets,
        metric_func=metric_function,
        predict_metric_func=predict_metric_func,
        train_preprocessing_func=original_preprocessing,
        test_preprocessing_func=original_preprocessing
    )

if __name__ == '__main__':
    # Change cwd to the main dir so the outputs/ dir
    # Is in the same place every run
    os.chdir(MAIN_DIR)
    main()
