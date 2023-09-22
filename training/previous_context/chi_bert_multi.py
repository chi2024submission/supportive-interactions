import json
import os
import sys
from pathlib import Path
import random
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import logging as log
import torch
import transformers
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from sklearn.utils import shuffle
from torch import nn
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback, is_torch_available, is_tf_available, RobertaForSequenceClassification, \
    BertForSequenceClassification, XLMRobertaForSequenceClassification, EvalPrediction


def set_gpus(devices: str):
    """
    @param devices: String of shape: '0,1,...'
    """
    no_gpu = len(devices.split(','))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    print('No. of devices: ' + str(no_gpu) + ' : ' + devices)
    return no_gpu


def set_seed(seed: int):
    """
    Set reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def rminsidedir(directory, prefix: str):
    directory = Path(directory)
    with os.scandir(directory) as d:
        for item in d:
            if item.name.startswith(prefix):
                rmdir(item)


def get_class_ratio(labels):
    # configure class weight by pos/all ratio
    sm = sum(labels)
    pos = sm / len(labels)
    neg = 1 - pos
    # reverse pos/neg to offset weights
    class_ratio = [pos, neg]
    return class_ratio


def read_split_data(tag: str, dirr: Path, split_i: str):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.train')) as json_data:
        root = json.load(json_data)
        train_texts = [x[0] for x in root['multi']]
        train_labels = [x[1] for x in root['multi']]

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.dev')) as json_data:
        root = json.load(json_data)
        dev_texts = [x[0] for x in root['multi']]
        dev_labels = [x[1] for x in root['multi']]

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['multi']]
        test_labels = [x[1] for x in root['multi']]

    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def read_coarse_test_data(tag: str, dirr: Path, split_i: str):
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['coarse']]
        test_labels = [x[1] for x in root['coarse']]
    return test_texts, test_labels


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'f1_weighted': f1_weighted_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def get_probs(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=ARGS['max_length'],
                       return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    return outputs[0].softmax(1)


def load_config(file: Path) -> dict:
    with open(file) as json_data:
        root = json.load(json_data)
    return root


def mk_out_dir(embed_in_dir: str) -> Path:
    if embed_in_dir:
        pth = Path('results', ARGS['run_id'], embed_in_dir)
    else:
        pth = Path('results', str(ARGS['run_id']))
    os.makedirs(pth, exist_ok=True)
    return pth


def plot_loss():
    plt.plot(LOSS)
    plt.plot(F1)
    plt.savefig(Path(ARGS['output_dir'], 'loss.png'))  # out dir must be specified before
    plt.show()  # needs to be after save, bc it deletes the plot


class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)


class LogCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if 'loss' in state.log_history[-2]:
            LOSS.append(state.log_history[-2]['loss'])
        if 'eval_f1' in state.log_history[-1]:
            F1.append(state.log_history[-1]['eval_f1'])
        # if 'eval_cfm' in state.log_history[-1]:
        #     CFM.append(state.log_history[-1]['eval_cfm'])


def predict_one(text: str, tok, mod):
    encoding = tok(text, return_tensors="pt")
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [idx for idx, label in enumerate(predictions) if label == 1.0]
    print(f'{predicted_labels}, probs: {probs}')


def predict_one_ml(text: str, tok, mod, threshold=0.5):
    encoding = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=ARGS['max_length'])
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    return predictions


def predict_one_m2b(text: str, tok, mod):
    encoding = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=ARGS['max_length'])
    encoding = {k: v.to(mod.device) for k, v in encoding.items()}
    outputs = mod(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [idx for idx, label in enumerate(predictions) if label == 1.0]
    return 1 if sum(predicted_labels) > 0 else 0


def bin_f1(true_labels, pred_labels):
    return round(f1_score(y_true=true_labels, y_pred=pred_labels), 4)


def multi_f1(true_labels, pred_labels, method):
    return round(f1_score(y_true=true_labels, y_pred=pred_labels, average=method), 4)


def bootstrap_michal(true_labels, pred_labels, threshold=0.5):
    bootstrap_sample = int(len(true_labels)/10)
    bootstrap_repeats = 50
    evaluations = []
    ret = {}
    for fscore_method in ["weighted", "micro", "macro"]:
        for repeat in range(bootstrap_repeats):
            sample_idx = random.sample(list(range(len(pred_labels))), k=bootstrap_sample)
            pred_sample = [pred_labels[idx] for idx in sample_idx]
            true_sample = [true_labels[idx] for idx in sample_idx]
            evaluation = f1_score(true_sample, pred_sample, average=fscore_method)
            evaluations.append(evaluation)

        lower = np.quantile(evaluations, q=0.025)
        upper = np.quantile(evaluations, q=0.975)
        mean = np.mean(evaluations)
        log.info("Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean))
        ret[fscore_method] = "Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean)
    return ret


def run():
    log.info('Loading datasets...')
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = read_split_data(ARGS['tag'],
                                                                                                ARGS['dir'],
                                                                                                ARGS['split'])
    _, coarse_labels = read_coarse_test_data(ARGS['tag'], ARGS['dir'], ARGS['split'])

    log.info('Loading tokenizer and tokenizing...')
    tokenizer = AutoTokenizer.from_pretrained(ARGS['model_name'], use_fast=False, truncation_side='left')
    assert tokenizer.truncation_side == 'left'
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=ARGS['max_length'])
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=ARGS['max_length'])
    train_dataset = EncodingDataset(train_encodings, train_labels)
    dev_dataset = EncodingDataset(dev_encodings, dev_labels)

    log.info('Load model and move to GPU...')
    if ARGS['model_name'] == 'Seznam/small-e-czech':
        model = ElectraForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=6, problem_type="multi_label_classification").to("cuda")
    elif ARGS['model_name'] == 'ufal/robeczech-base':
        model = RobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=6, problem_type="multi_label_classification").to("cuda")
    elif ARGS['model_name'] == 'xlm-roberta-base' or ARGS['model_name'] == 'xlm-roberta-large':
        model = XLMRobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=6, problem_type="multi_label_classification").to("cuda")
    else:
        raise Exception('Unknown model name.')

    log.info('Set training ARGS...')
    w_steps = int((ARGS['epochs'] * len(train_texts)) / (10 * ARGS['per_device_batch'] * ARGS['visible_devices']))
    training_args = TrainingArguments(
        output_dir=ARGS['output_dir'],  # output directory
        num_train_epochs=ARGS['epochs'],  # total number of training epochs
        per_device_train_batch_size=ARGS['per_device_batch'],  # batch size per device during training
        per_device_eval_batch_size=ARGS['per_device_batch'],  # batch size for evaluation
        gradient_accumulation_steps=ARGS['gradient_acc'],
        warmup_steps=w_steps,  # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        # logging_steps='epoch',#ARGS['logging_steps'],  # log & save weights each logging_steps
        logging_strategy=ARGS['logging_strategy'],
        evaluation_strategy=ARGS['evaluation_strategy'],  # 'steps',  # evaluate each `logging_steps`
        save_strategy=ARGS['save_strategy'],
        # logging_steps=ARGS['logging_steps'],  # log & save weights each logging_steps
        # evaluation_strategy='steps',  # evaluate each `logging_steps`
        learning_rate=ARGS['learning_rate'],
        save_total_limit=ARGS['save_total_limit'],
        disable_tqdm=True,
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[LogCallback, EarlyStoppingCallback(early_stopping_patience=ARGS['early_stopping_patience'])]
    )

    log.info('Train...')
    trainer.train()
    # log.info(trainer.evaluate())
    model_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    trainer.save_model(model_pth)

    tok_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}-tokenizer'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    tokenizer.save_pretrained(tok_pth)

    log.info('Evaluation multi-multi...')
    preds = [predict_one_ml(tt, tokenizer, model) for tt in test_texts]
    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}_m2m.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    roc_auc = roc_auc_score(test_labels, preds, average='weighted')
    ci: Dict = bootstrap_michal(test_labels, preds)
    cfm = list(multilabel_confusion_matrix(test_labels, preds))
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
        result = {
            'multi_f1_weighted': multi_f1(test_labels, preds, method="weighted"),
            'multi_f1_micro': multi_f1(test_labels, preds, method="micro"),
            'multi_f1_macro': multi_f1(test_labels, preds, method="macro"),
            'auc': f'{roc_auc:.4f}',
            'cfm': str(cfm)
        }
        result.update(ci)
        json.dump(result, outfile, ensure_ascii=False)

    log.info('Evaluation multi-coarse...')
    preds = [predict_one_m2b(tt, tokenizer, model) for tt in test_texts]
    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}_m2c.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    bf1 = bin_f1(coarse_labels, preds)
    roc_auc = roc_auc_score(coarse_labels, preds, average='weighted')
    ci: Dict = bootstrap_michal(coarse_labels, preds)
    cfm = list(confusion_matrix(coarse_labels, preds).tolist())
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
        result = {
            'test_f1': bf1,
            'auc': f'{roc_auc:.4f}',
            'cfm': str(cfm)
        }
        result.update(ci)
        json.dump(result, outfile, ensure_ascii=False)

    rminsidedir(ARGS['output_dir'], 'checkpoint')
    log.info('Finished.')
    return model, tokenizer


def main():
    # set seed and GPUs
    set_seed(1)
    no_gpus = set_gpus(ARGS['gpu'])
    ARGS['visible_devices'] = no_gpus

    # run
    model, tokenizer = run()  # results are saved in split_# dir
    plot_loss()  # only plots for this split
    return model, tokenizer


if __name__ == '__main__':
    # load args
    ARGS = load_config(Path(sys.argv[1]))
    ARGS['split'] = str(sys.argv[2])  # split_#
    ARGS['gpu'] = str(sys.argv[3])  # single string number
    ARGS['output_dir'] = mk_out_dir(ARGS['split'])  # results/<run_id>/<split_#>

    # create global loss logger
    LOSS = []
    F1 = []
    CFM = []

    # file logging
    log_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}.log'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    file_handler = log.FileHandler(filename=log_pth)
    std_handler = log.StreamHandler(sys.stdout)
    err_handler = log.StreamHandler(sys.stderr)
    handlers = [file_handler, std_handler, err_handler]
    log.basicConfig(
        level=log.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    # run
    try:
        MODEL, TOKENIZER = main()
    except Exception as e:
        log.exception(e)
        raise
