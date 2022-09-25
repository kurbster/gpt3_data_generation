# How To Add A Dataset To The Repo
After generating the data with GPT3 you should clean the data and remove any non-sense labels. Also, make sure the data is in the following format.

The only key is `data` which has a list of examples.
```
{
    'data': [
        {
            'passage': ...,
            'question': ...,
            'label': ...
        } ...
    ]
}
```

## Adding A Config File
To add a dataset name `my_dataset` you will need to add two config files `conf/my_dataset.yaml` and `conf/preprocessing_config/my_dataset.yaml`.
It is recommended you just copy an existing file then modify it.
`cp conf/wic.yaml conf/my_dataset.yaml` `cp conf/preprocessing_config/wic.yaml conf/preprocessing_config/my_dataset.yaml`

The first is the main config file that will be read when running the
experiment.

In this file you will need to change the following.
1. Set the `preprocessing_config` under the defaults to `my_dataset`.
2. Set the `dataset_name` under `hydra.env` to `my_dataset`.
3. Set the `num_labels` argument to the number of labels in the dataset.
4. Set the `max_train_samples` to the total number of samples in the generated
dataset. To quickly find out how many there are run `jq '.data | length' datasets/my_dataset/gpt3_generated.json`.
5. *Only if needed* you can add multiple splits for the same dataset under
the `train_files` argument. By adding another value to the list the script
will train and evaluate another model with that training file.

The second file defines how we should preprocess the original and generated
datasets for both generative and discriminative models. ***NOTE:*** every generated
dataset split will use the same config. Currently, there is not support for
different preprocessing across different generated splits.

In this file you will need to change the following.
1. For the original and generated `_target_` arguments you will need to put the
full path to the dataclass that will perform the preprocessing. For the original
datasets most (if not all) will be `src.our_datasets.Dataset`.
    a. The reason a generated dataset would have a different class is to cast
    the labels into an integer.
    b. This can also be done when cleaning the data originally, or it can be done
    in code. See `src/our_datasets.py` for examples.
2. Under the common config you will need to change the `header`, `footer`, and `columns`
arguments to match your dataset.
    a. If the experiment is run using a discriminative model then the prompt is
    contructed by iterating through the keys of the columns dictionary.
    The delimiter token `[SEP]` is also configurable by setting discriminative_delim.
    ```
    dataset[column 1] [SEP] dataset[column 2] ... dataset[column N]
    ```
    b. If the experiment is run using a generative model then the prompt is
    constructed in the following way. The default value for <DELIM> is `\n` and
    is also configurable by setting generative_delim
