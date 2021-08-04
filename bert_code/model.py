import argparse
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(RBERT, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        #self.e1_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        #self.e2_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * (3-2), bert_config.num_labels, args.dropout_rate, use_activation=False)

    # @staticmethod
    # def entity_average(hidden_output, e_mask):
    #     """
    #     Average the entity hidden state vectors (H_i ~ H_j)
    #     :param hidden_output: [batch_size, j-i+1, dim]
    #     :param e_mask: [batch_size, max_seq_len]
    #             e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
    #     :return: [batch_size, dim]
    #     """
    #     e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
    #     length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

    #     sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    #     avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    #     return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        #e1_h = self.entity_average(sequence_output, e1_mask)
        #e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        #e1_h = self.e1_fc_layer(e1_h)
        #e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        #concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        #logits = self.label_classifier(concat_h)
        logits = self.label_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="ddi", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument("--pretrained_model_name", default="monologg/biobert_v1.1_pubmed", required=False, help="Pretrained model name")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization.")
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
    bert_config = BertConfig.from_pretrained('monologg/biobert_v1.1_pubmed', num_labels=2, finetuning_task='ddi')
    model = RBERT(bert_config, args)
    print(model)