#!/usr/bin/env python3
import logging

from pathlib import Path

import hydra

from datasets import load_dataset, dataset_dict, Dataset
from transformers import HfArgumentParser, TrainingArguments

from config import ModelArguments, DataTrainingArguments
from run_experiment import run_model

def get_original_datasets(data_args: DataTrainingArguments, model_args: ModelArguments) -> dataset_dict:
    datasets = load_dataset(
        'super_glue',
        data_args.dataset_name,
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    logger.info(f'Len of Original train before {len(datasets["train"])}')
    datasets['train'] = datasets['train'].select(range(data_args.num_samples))
    logger.info(f'Len of Original train after {len(datasets["train"])}')

    return datasets

def get_generated_dataset(train_file: Path, data_args: DataTrainingArguments, model_args: ModelArguments) -> dataset_dict:
    dataset = load_dataset(
        train_file.suffix,
        data_files=train_file,
        field="data",
        split="train",
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    # Sample the generated dataset accordingly.
    if data_args.num_generated_samples != -1:
        dataset = dataset.select(range(data_args.num_generated_samples))
    
    return dataset

@hydra.main(config_path="../conf", config_name="wic", version_base="1.2")
def main(cfg):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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
        text_col=data_args.text_column,
        label_col=data_args.label_column,
    )
    generated_preprocessing = hydra.utils.instantiate(
        cfg.preprocessing_config.generated,
        text_col=data_args.text_column,
        label_col=data_args.label_column,
    )

    # Get the metric function
    metric_function = hydra.utils.instantiate(cfg.metric)

    training_args.output_dir = main_output_dir / 'original'
    logger.info('STARTING THE MODEL WITH THE ORIGINAL TRAIN DATA')
    logger.info('*'*75)

    run_model(
        model_args,
        data_args,
        training_args,
        datasets,
        metric_func=metric_function,
        train_preprocessing_func=original_preprocessing,
        test_preprocessing_func=original_preprocessing
    )

    if len(data_args.train_files) > 0:
        for train_file in data_args.train_files:
            train_file = Path(train_file)
            training_args.output_dir = main_output_dir / train_file.stem

            datasets["train"] = get_generated_dataset(train_file, data_args, model_args)

            run_model(
                model_args,
                data_args,
                training_args,
                datasets,
                metric_func=metric_function,
                train_preprocessing_func=original_preprocessing,
                test_preprocessing_func=original_preprocessing
            )

if __name__ == '__main__':
    main()