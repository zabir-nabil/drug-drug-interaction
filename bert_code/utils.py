import os
import random
import logging

import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=args.do_lower_case)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    # write labels in csv file
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='micro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, labels=[1, 2, 3, 4], average=average)
    return {
        "acc": acc,
        "f1": f1,
    }
