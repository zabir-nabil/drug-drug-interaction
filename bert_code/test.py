import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer
from data_loader import load_and_cache_examples
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
def main(args):
    tokenizer = load_tokenizer(args)
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=2)

    for d in train_dataloader:
        #print(d[3])
        if torch.tensor(2) in d[3]:
            print('fuck')
        #print(len(d))
        #print(d[0].shape)
        #print(d[1].shape)
        #print(d[2].shape)
        #print(d[3].shape)
        #print(d[4].shape)
        #print(d[5].shape)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="ddi", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data_11", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument("--pretrained_model_name", default="monologg/biobert_v1.1_pubmed", required=False, help="Pretrained model name")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=300, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=400, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=400, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()
    main(args)