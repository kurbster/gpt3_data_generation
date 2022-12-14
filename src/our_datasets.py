#!/usr/bin/env python3
from typing import Callable, Dict, List
from dataclasses import dataclass

import numpy as np

@dataclass
class Dataset:
    header: str
    footer: str
    columns: Dict[str, str]
    tokenizer: Callable
    is_generative_model: bool
    text_col: str = "text"
    label_col: str = "label"
    generative_delim: str = "\n"
    discriminative_delim: str = " [SEP] "
    def _make_generative_model_input(self, example: Dict[str, str]) -> Dict[str, str]:
        # Add the prefix to each column
        for column_name, column_prefix in self.columns.items():
            example[column_name] = column_prefix + example[column_name] 

        # Join the columns together
        text = self.generative_delim.join(
            [example[column_name] for column_name in self.columns]
        )

        # Add the header and footer
        text = self.generative_delim.join([self.header, text, self.footer])
        return text

    def _make_discriminative_model_input(self, example: Dict[str, str]) -> Dict[str, str]:
        return self.discriminative_delim.join(
            [example[column_name] for column_name in self.columns]
        )

    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        if self.is_generative_model:
            text = self._make_generative_model_input(example)
        else:
            text = self._make_discriminative_model_input(example)
        input = self.tokenizer(text)
        input[self.text_col] = text # Save the text for analysis later.
        if self.is_generative_model:
            input[self.label_col] = self.tokenizer(str(example[self.label_col]))['input_ids']
        return input

@dataclass
class WicGeneratedDataset(Dataset):
    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        example[self.label_col] = 1 if example[self.label_col] == 'T' else 0
        input = super().__call__(example)
        return input

@dataclass
class RteGeneratedDataset(Dataset):
    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        example[self.label_col] = 1 if example[self.label_col] == 'entailment' else 0
        input = super().__call__(example)
        return input

@dataclass
class BoolQGeneratedDataset(Dataset):
    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        example[self.label_col] = 1 if example[self.label_col] == 'True' else 0
        input = super().__call__(example)
        return input

@dataclass
class RecordOriginalDataset(Dataset):
    query_col: str = "query"
    entity_col: str = "entities"
    answer_col: str = "answers"
    passage_col: str = "passage"

    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        ent_lengths = [len(ents) for ents in example['entities']]
        passages = [[context] * ent_len for context, ent_len in zip(example['passage'], ent_lengths)]
        query_headers = example['query']
        endings = [
            [f"{query} [SEP] {ent}" for ent in example['entities'][i]]
            for i, query in enumerate(query_headers)
        ]
        passages = sum(passages, [])
        endings = sum(endings, [])
        output = self.tokenizer(passages, endings)
        labels = np.zeros(len(passages), dtype=np.int8)
    
        ent_lengths.insert(0, 0)
        ent_arr = np.array(ent_lengths)
        length_sums = np.cumsum(ent_arr)
    
        for i, ans_list in enumerate(example['answers']):
            for ans in ans_list:
                ans_idx = example['entities'][i].index(ans)
                labels[length_sums[i] + ans_idx] = 1
        output[self.label_col] = labels
        output[self.text_col] = [
            passage + self.delim + end for passage, end in zip(passages, endings)
        ]
        return output

@dataclass
class RecordOriginalDatasetMC(Dataset):
    query_col: str = "query"
    entity_col: str = "entities"
    answer_col: str = "answers"
    passage_col: str = "passage"
    num_choices: int = 3
    def __call__(self, example: Dict[str, List[str]]) -> Dict[str, str]:
        passages = [[passage] * self.num_choices for passage in example[self.passage_col]]
        # TODO: Can this be done in a list comprehension
        # TODO: Check if the label or correct query would always be last then?
        queries = []
        for i, query in enumerate(example[self.query_col]):
            new_query = []
            for j in range(self.num_choices - 1):
                new_query.append(f"{query}{self.delim}{example[self.entity_col][i][j]}")
            new_query.append(f"{query}{self.delim}{example[self.answer_col][i][0]}")

            queries.append(new_query)

        inputs = self.tokenizer(passages, queries)
        inputs[self.label_col] = [self.num_choices - 1] * len(passages)

        return {
            k: [v[i: i+self.num_choices] for i in range(0, len(v), self.num_choices)]
            for k, v in inputs.items()
        }


@dataclass
class RecordOriginalDatasetVariableLength(Dataset):
    query_col: str = "query"
    entity_col: str = "entities"
    passage_col: str = "passage"
    def __call__(self, example: Dict[str, List[str]]) -> Dict[str, str]:
        passages = [[ex] * len(example[self.entity_col][i])
                    for i, ex in enumerate(example[self.passage_col])]
        # Should we form queries this way? Or should we replace the @placeholder?
        queries = [
            [
                f"{query}{self.delim}{entity}"
                for entity in example[self.entity_col][i]
            ]
            for i, query in enumerate(example[self.query_col])
        ]
        
        passages = sum(passages, [])
        queries = sum(queries, [])

        input = self.tokenizer(passages, queries)

        # Output of tokenizer is the flattened input.
        # We need to convert it back into a size of (N, E)
        # Where N is the initial batch size and E is the number of entites
        # For each problem, which changes.
        i = 0
        question_sizes = []
        # Can this be done in a list comprehension?
        for ex in example[self.entity_col]:
            length = len(ex)
            question_sizes.append((i, i + length))
            i += length

        tokenized = {
            k: [v[start: end] for start, end in question_sizes]
            for k, v in input.items()
        }
        tokenized['labels'] = [
            [entity.index(ans) for ans in example['answers'][i]]
            for i, entity in enumerate(example[self.entity_col])
        ]

        return tokenized

@dataclass
class RecordOriginalDatasetNoBatch(Dataset):
    query_col: str = "query"
    entity_col: str = "entities"
    passage_col: str = "passage"
    def __call__(self, example: Dict[str, List[str]]) -> Dict[str, str]:
        texts = [
            f"{example[self.passage_col]}{self.delim}{example[self.query_col]}"
            + f"{self.delim}{entity}" for entity in example[self.entity_col]
        ]
        input = self.tokenizer(texts)

        input['labels'] = [
            example[self.entity_col].index(ans) for ans in example['answers']
        ]

        return input
