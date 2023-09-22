import json
import logging
from pathlib import Path
from typing import Union, List

import numpy as np
from sklearn.utils import shuffle

from constants.tags import TAGS

log = logging.getLogger()


def get_class_ratio(labels):
    # configure class weight by pos/all ratio
    sm = sum(labels)
    pos = sm / len(labels)
    neg = 1 - pos
    # reverse pos/neg to offset weights
    class_ratio = [neg, pos]
    return class_ratio


def verbalize_label(label: Union[List[int], List[List[int]]], granularity: str) -> str:
    if granularity == "seq_coarse":
        return "positive" if label else "negative"
    elif granularity == "seq_multi":
        # label is one-hot encoding
        return ", ".join([TAGS[idx] for idx, binary_val in enumerate(label) if binary_val])
    elif granularity == "seq_multi_per_utterance":
        return "; ".join(", ".join([TAGS[idx] for idx, binary_val in enumerate(utterance_label) if binary_val])
                         for utterance_label in label)
    else:
        raise ValueError(granularity)


def read_split_data(tag: str, dirr: Path, split_i: str, granularity: str = "seq_coarse"):
    """
    Read the data and oversample the minority class - only works if the minority class is at most 1/3 of the dataset.
    """
    label_idx = 3 if granularity == "seq_multi_per_utterance" else 2
    key = "seq_multi" if granularity == "seq_multi_per_utterance" else granularity

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.train')) as json_data:
        root = json.load(json_data)
        train_texts = [x[0] for x in root[key]]
        train_text_pairs = [x[1] for x in root[key]]
        train_labels = [verbalize_label(x[label_idx], granularity) for x in root[key]]
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.dev')) as json_data:
        root = json.load(json_data)
        dev_texts = [x[0] for x in root[key]]
        dev_text_pairs = [x[1] for x in root[key]]
        dev_labels = [verbalize_label(x[label_idx], granularity) for x in root[key]]
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root[key]]
        test_text_pairs = [x[1] for x in root[key]]
        test_labels = [verbalize_label(x[label_idx], granularity) for x in root[key]]
    return (train_texts, train_text_pairs, train_labels,
            dev_texts, dev_text_pairs, dev_labels,
            test_texts, test_text_pairs, test_labels)
