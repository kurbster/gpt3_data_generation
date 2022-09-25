import os
import sys
import logging
import functools

from typing import Callable, Set, Tuple, Union, List
from pathlib import Path

MAIN_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(MAIN_DIR))

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_dataset, dataset_dict

from src.config import ModelArguments, DataTrainingArguments
from src.custom_trainer import LogCallBack
from src.util import ExperimentType

logger = logging.getLogger("myLogger")

def get_checkpoint(training_args: TrainingArguments) -> Union[None, str]:
    # Detecting last checkpoint.
    last_checkpoint = None

    output_dir = training_args.output_dir
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        # If we are training without overwrite. Then we need to make sure there is a checkpoint.
        if (training_args.do_train and not training_args.overwrite_output_dir) and \
           (last_checkpoint is None and len(os.listdir(output_dir))) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.overwrite_output_dir:
        last_checkpoint = None
    return last_checkpoint

def get_model_and_tokenizer(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    checkpoint: str,
    experiment: ExperimentType,
) -> Tuple[AutoModel, AutoTokenizer, AutoConfig]:
    model_name = checkpoint if checkpoint else model_args.model_name
    logger.info('Loading Model Config')
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        num_labels=data_args.num_labels,
    )

    logger.info('Loading Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    logger.info('Loading Model')
    model = experiment.model_type.from_pretrained(model_name, config=config)

    logger.info(f'Len of tokenizer {len(tokenizer)}')
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, config

def get_datasets(data_args: DataTrainingArguments, model_args: ModelArguments) -> dataset_dict:
    extension = data_args.train_file.split(".")[-1]
    datasets = load_dataset('super_glue', data_args.dataset_name).shuffle(seed=data_args.random_seed)

    logger.info(f'Len of train before {len(datasets["train"])}')
    datasets['train'] = datasets['train'].select(range(data_args.max_train_samples))
    logger.info(f'Len of train after {len(datasets["train"])}')

    datasets["generated_train"] = load_dataset(
        extension,
        data_files=data_args.train_file,
        field="data",
        split="train",
        cache_dir=model_args.cache_dir
    ).shuffle(seed=data_args.random_seed)

    def clean_record(examples):
        examples['passage'] = [
            passage.replace('\n@highlight\n', ' ') for passage in examples['passage']
        ]
        return examples

    if data_args.dataset_name == 'record':
        datasets[data_args.test_set_key] = datasets[data_args.test_set_key].map(
            clean_record,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Cleaning record validation dataset"
        )

    return datasets

def get_columns_to_remove(dataset, preprocess_callable) -> Set[str]:
    columns = set(dataset.column_names)
    logger.info(f"columns in the dataset {columns}")
    columns_to_return = set([preprocess_callable.text_col, preprocess_callable.label_col])
    columns.difference_update(columns_to_return)
    return columns
    
def prepare_dataset(
    dataset,
    preprocess_callable,
    data_args: DataTrainingArguments,
    model_config: AutoConfig,
    training_args: TrainingArguments,
):
    columns_to_remove = get_columns_to_remove(dataset, preprocess_callable)
    logger.info(f'I am columns to remove: {columns_to_remove}')

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_callable,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=columns_to_remove,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # TODO: Check if this works with BERT and RoBERTA if we are using them
    dataset = dataset.rename_column(preprocess_callable.label_col, "labels")
    logger.debug(f'Dataset label after {dataset["labels"][0]}')
    logger.debug(f'columns in cleaned dataset {dataset.features}')
    logger.debug(f'number of input_ids {len(dataset["input_ids"])}')
    logger.debug(f'length of first input ids {len(dataset["input_ids"][0])}')

    before_length = dataset.num_rows
    max_length = min(model_config.max_position_embeddings, data_args.max_source_length)
    logger.info(f'Length before filtering by length: {before_length}')
    dataset = dataset.filter(
        lambda example: len(example['input_ids']) < max_length
    )
    after_length = dataset.num_rows
    logger.info(f'Length after filtering by length: {after_length}')
    assert ((before_length - after_length) / before_length) < data_args.max_removed_samples_warning, 'We removed too many samples'

    return dataset

def get_callbacks(
    data_args: DataTrainingArguments
) -> List[TrainerCallback]:
    early_stopping_callback = EarlyStoppingCallback(
        data_args.early_stopping_patience,
        data_args.early_stopping_threshold
    )

    logging_callback = LogCallBack()

    return [early_stopping_callback, logging_callback]

def get_trainer(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    experiment: ExperimentType,
    train_dataset,
    eval_dataset,
    data_collator,
    metric_function: Callable,
) -> Trainer:
    callbacks = get_callbacks(data_args)
    return experiment.trainer_type(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_function,
        callbacks=callbacks
    )

def run_training_loop(
    trainer: Trainer,
    data_args: DataTrainingArguments,
    last_checkpoint: Union[str, None],
    train_dataset
):
    logger.info("***Training ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["num_train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def run_eval_loop(trainer: Trainer, eval_dataset):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("hf_eval", metrics)

def run_test_loop(
    trainer: Trainer,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    predict_metric_func: Callable,
    predict_dataset,
    log_predictions: bool,
):
    logger.info("*** Predict ***")
    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    if log_predictions:
        trainer.log_metrics("predict", metrics)
    trainer.save_metrics("hf_predict", metrics)

    if trainer.is_world_process_zero():
        predict_metric_func(
            predict_results,
            Path(training_args.output_dir),
            predict_dataset[data_args.text_column]
        )

def run_model(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: dataset_dict,
    metric_func: Callable,
    predict_metric_func: Callable,
    train_preprocessing_func: Callable,
    test_preprocessing_func: Callable,
    experiment: ExperimentType,
    log_predictions: bool = True,
):
    """Train the model specified by model_args with options specified
    under training_args with the datasets stored in datasets. The
    train and test preprocessing_func should be a callable object that
    will tokenize the input datasets.

    Args:
        model_args (ModelArguments): The specification of the model to train.
        data_args (DataTrainingArguments): The specification of the datasets to use.
        training_args (TrainingArguments): The specification of the training parameters.
        datasets (dataset_dict): The datasets to use.
        metric_func (Callable): The callable object to calculate metrics.
        predict_metric_func (Callable): The callable object to calculate metrics for prediction.
        train_preprocessing_func (Callable): The preprocessing function for the train set.
        test_preprocessing_func (Callable): The preprocessing function for the test set.
    """
    last_checkpoint = get_checkpoint(training_args)
    logger.info(f'Checkpoint for model: {last_checkpoint}')
    model, tokenizer, model_config = get_model_and_tokenizer(model_args, data_args, last_checkpoint, experiment)

    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    padding = "max_length" if data_args.pad_to_max_length else "longest"
    tokenizer_func = functools.partial(
        tokenizer,
        padding=False, truncation=True
        # padding=padding, truncation=True
    )

    metric_func = functools.partial(metric_func, tokenizer=tokenizer)
    predict_metric_func = functools.partial(predict_metric_func, tokenizer=tokenizer)

    # Instantiate the preprocessing class with the tokenizer
    train_preprocessing = train_preprocessing_func(tokenizer=tokenizer_func)
    test_preprocessing = test_preprocessing_func(tokenizer=tokenizer_func)

    # Init to None incase user only wants to do_predict
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        logger.info("Preparing train dataset")
        train_dataset = prepare_dataset(
            datasets["train"],
            train_preprocessing,
            data_args,
            model_config,
            training_args,
        )

    if training_args.do_eval:
        logger.info("Preparing validation dataset")
        eval_dataset = prepare_dataset(
            datasets["validation"],
            train_preprocessing,
            data_args,
            model_config,
            training_args,
        )

    if training_args.do_predict:
        logger.info("Preparing test dataset")
        predict_dataset = prepare_dataset(
            datasets["test"],
            test_preprocessing,
            data_args,
            model_config,
            training_args,
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"
    data_collator_type = experiment.get_data_collator(model)
    data_collator = data_collator_type(
        tokenizer,
        padding=padding,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = get_trainer(
        training_args=training_args, model=model,
        tokenizer=tokenizer, train_dataset=train_dataset,
        eval_dataset=eval_dataset, data_collator=data_collator,
        metric_function=metric_func, data_args=data_args,
        experiment=experiment
    )

    eval_output_path = Path(training_args.output_dir, 'evaluation_results')
    eval_output_path.mkdir(parents=True, exist_ok=True)

    # Training
    if training_args.do_train:
        run_training_loop(
            trainer=trainer, data_args=data_args,
            last_checkpoint=last_checkpoint,
            train_dataset=train_dataset
        )

    # Evaluation
    if training_args.do_eval:
        run_eval_loop(
            trainer=trainer,
            eval_dataset=eval_dataset
        )

    # Prediction
    if training_args.do_predict:
        run_test_loop(
            trainer=trainer,
            data_args=data_args,
            training_args=training_args,
            predict_dataset=predict_dataset,
            predict_metric_func=predict_metric_func,
            log_predictions=log_predictions,
        )
