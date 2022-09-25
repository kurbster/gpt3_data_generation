#!/usr/bin/env python3
from cProfile import label
import os
import sys
import logging

from pathlib import Path

MAIN_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(MAIN_DIR))

import hydra

import datasets
import transformers

from datasets import load_dataset, dataset_dict
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.config import ModelArguments, DataTrainingArguments
from src.run_experiment import run_model
from src.util import ExperimentType

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

def get_original_datasets(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    label_col: str
) -> dataset_dict:
    datasets = load_dataset(
        'super_glue',
        data_args.dataset_name,
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    # Set the validation as the test set first before modifying it
    datasets['test'] = datasets[data_args.test_set_key]

    # To maintain the same relative label frequency we need to perform
    # A stratified split to get the reduce train dataset.
    # Then another stratified split to get the final train/val dataset.
    logger.info(f'Len of Original train before {datasets["train"].num_rows}')
    reduced_train_dataset = datasets['train']
    if datasets['train'].num_rows > data_args.max_train_samples:
        reduced_train_dataset = datasets['train'].train_test_split(
            seed=data_args.random_seed,
            stratify_by_column=label_col,
            train_size=data_args.max_train_samples,
        )['train']  # Only grab the train dataset of the split

    # Select from the end of the train set until the max eval samples
    final_train_dataset = reduced_train_dataset.train_test_split(
        seed=data_args.random_seed,
        stratify_by_column=label_col,
        test_size=data_args.eval_split_percent,
    )

    datasets['train'] = final_train_dataset['train']
    datasets['validation'] = final_train_dataset['test']

    logger.info(f'Len of Original train after {datasets["train"].num_rows}')
    logger.info(f'Len of Validation set: {datasets["validation"].num_rows}')
    logger.info(f'Len of Test set: {datasets["test"].num_rows}')

    return datasets

def get_generated_dataset(
    train_file: Path,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    label_col: str
) -> dataset_dict:
    dataset = load_dataset(
        train_file.suffix.strip('.'),
        data_files=str(train_file),
        field="data",
        split="train",
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    # Make the label column a ClassLabel so we can do a stratified split
    dataset = dataset.class_encode_column(label_col)
    dataset = dataset.train_test_split(
        seed=data_args.random_seed,
        stratify_by_column=label_col,
        test_size=data_args.eval_split_percent,
    ) 

    return dataset["train"], dataset["test"]

def resolve_label_column(common_config, specific_config):
    # Check if the common preprocessing config defined the label
    label_col = common_config.get("label_col")
    # If not check the specific config. And if not use default label
    # NOTE: the default value must match the default defined in our_datasets.py
    if label_col is None:
        label_col = specific_config.get("label_col", "label")
    return label_col

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

    experiment = ExperimentType(model_args.model_name)

    common_preprocessing = cfg.preprocessing_config.get("common_config", {})
    # Get the original and generated preprocessing functions.
    original_preprocessing = hydra.utils.instantiate(
        cfg.preprocessing_config.original,
        is_generative_model=experiment.is_generative_model,
        _partial_=True,
        **common_preprocessing
    )
    generated_preprocessing = hydra.utils.instantiate(
        cfg.preprocessing_config.generated,
        is_generative_model=experiment.is_generative_model,
        _partial_=True,
        **common_preprocessing
    )

    # Get the metric function
    metric_func = hydra.utils.instantiate(cfg.metric, _partial_=True)
    predict_metric_func = hydra.utils.instantiate(cfg.prediction_metric, _partial_=True)

    # Add the experiment specific postprocessing as a decorator
    metric_func = experiment.eval_postprocess_output(metric_func)
    predict_metric_func = experiment.predict_postprocess_output(predict_metric_func)
    
    # We need to get the label column from the config so when we split the data
    # We can split the data evenly by labels
    original_label_col = resolve_label_column(
        common_preprocessing,
        cfg.preprocessing_config.original
    )
    generated_label_col = resolve_label_column(
        common_preprocessing,
        cfg.preprocessing_config.generated
    )

    # These are the original train, validation, and test
    datasets = get_original_datasets(data_args, model_args, original_label_col)

    if data_args.run_original:
        training_args.output_dir = str(main_output_dir / 'original')
        logger.info('STARTING THE MODEL WITH THE ORIGINAL TRAIN DATA')
        logger.info('*'*75)
        run_model(
            model_args,
            data_args,
            training_args,
            datasets,
            metric_func=metric_func,
            predict_metric_func=predict_metric_func,
            train_preprocessing_func=original_preprocessing,
            test_preprocessing_func=original_preprocessing,
            experiment=experiment,
        )

    if len(data_args.train_files) > 0:
        for train_file in data_args.train_files:
            train_file = MAIN_DIR / train_file
            training_args.output_dir = str(main_output_dir / train_file.stem)

            # These are the train and validation taken from the generated dataset
            datasets["train"], datasets["validation"] = get_generated_dataset(
                train_file, data_args, model_args, generated_label_col
            )

            run_model(
                model_args,
                data_args,
                training_args,
                datasets,
                metric_func=metric_func,
                predict_metric_func=predict_metric_func,
                train_preprocessing_func=generated_preprocessing,
                test_preprocessing_func=original_preprocessing,
                experiment=experiment,
            )

if __name__ == '__main__':
    # Change cwd to the main dir so the outputs/ dir
    # Is in the same place every run
    os.chdir(MAIN_DIR)
    main()
