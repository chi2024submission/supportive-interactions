import json
import os
import sys
from pathlib import Path
import random
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import logging as log

import scipy
import torch
import transformers
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle
from torch import nn
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback, is_torch_available, is_tf_available, RobertaForSequenceClassification, \
    BertForSequenceClassification, XLMRobertaForSequenceClassification, XLMRobertaXLForSequenceClassification


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
    class_ratio = [neg, pos]
    return class_ratio


def read_split_data(tag: str, dirr: Path, split_i: str):
    """
    Read the data and oversample the minority class - only works if the minority class is at most 1/3 of the dataset.
    """
    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.train')) as json_data:
        root = json.load(json_data)
        train_texts = [x[0] for x in root['coarse']]
        train_labels = [x[1] for x in root['coarse']]

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.dev')) as json_data:
        root = json.load(json_data)
        dev_texts = [x[0] for x in root['coarse']]
        dev_labels = [x[1] for x in root['coarse']]

    with open(Path(dirr, 'split-' + split_i + '_' + tag + '.test')) as json_data:
        root = json.load(json_data)
        test_texts = [x[0] for x in root['coarse']]
        test_labels = [x[1] for x in root['coarse']]
    neg_ratio, pos_ratio = get_class_ratio(train_labels)
    log.info('Class weights before ovesample: {:.2f}:{:.2f}'.format(neg_ratio, pos_ratio))

    # oversample the minority class - only works if the minority class is at most 1/3 of the dataset
    os_texts = []
    os_labels = []
    minority_ratio = min(pos_ratio, neg_ratio)
    minority_label = int(np.argmin([neg_ratio, pos_ratio]))
    if minority_ratio * 2 <= max(pos_ratio, neg_ratio):
        for i, v in enumerate(train_labels):
            if v != minority_label:
                continue
            add_n = int((1 / minority_ratio) - 1)
            for j in range(add_n):
                os_labels.append(v)
                os_texts.append(train_texts[i])
        train_texts.extend(os_texts)
        train_labels.extend(os_labels)
        train_texts, train_labels = shuffle(train_texts, train_labels)
        log.info(f'Oversampling class: {minority_label}')

    ARGS['class_ratio'] = get_class_ratio(train_labels)
    log.info('Class weights after ovesample: {:.2f}:{:.2f}'.format(ARGS['class_ratio'][0], ARGS['class_ratio'][1]))
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def compute_metrics(pred):
    labels = pred.label_ids
    pred = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    cfm = confusion_matrix(labels, pred).tolist()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "cfm": cfm}


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
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(ARGS['class_ratio']).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class LogCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if 'loss' in state.log_history[-2]:
            LOSS.append(state.log_history[-2]['loss'])
        if 'eval_f1' in state.log_history[-1]:
            F1.append(state.log_history[-1]['eval_f1'])
        if 'eval_cfm' in state.log_history[-1]:
            CFM.append(state.log_history[-1]['eval_cfm'])


def bin_f1(true_labels, probs, threshold=0.5):
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    return round(f1_score(y_true=true_labels, y_pred=predictions.argmax(-1)), 4)


def bootstrap_michal(true_labels, probs, threshold=0.5):
    bootstrap_sample = int(len(true_labels)/10)
    bootstrap_repeats = 200
    evaluations = []
    pred_int = preds2class(probs, threshold)
    ret = {}
    for fscore_method in ["weighted", "micro", "macro"]:
        for repeat in range(bootstrap_repeats):
            sample_idx = random.sample(list(range(len(pred_int))), k=bootstrap_sample)
            pred_sample = [pred_int[idx] for idx in sample_idx]
            true_sample = [true_labels[idx] for idx in sample_idx]
            evaluation = f1_score(true_sample, pred_sample, average=fscore_method)
            evaluations.append(evaluation)

        lower = np.quantile(evaluations, q=0.025)
        upper = np.quantile(evaluations, q=0.975)
        mean = np.mean(evaluations)
        log.info("Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean))
        ret[fscore_method] = "Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean)
    return ret


def bootstrap_f1(*data):
    true_labels = []
    preds = []
    for x in data[0]:
        t = 0
        p = 0
        if x == 1:
            t = 1
        elif x >= 10:
            p = 1
            t = x % 10
        true_labels.append(t)
        preds.append(p)
    preds = np.array(preds)
    return f1_score(y_true=true_labels, y_pred=preds)


def preds2class(probs, threshold=0.5):
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    return predictions.argmax(-1)


def run():
    log.info('Loading datasets...')
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = read_split_data(ARGS['tag'], ARGS['dir'], ARGS['split'])

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
            ARGS['model_name'], num_labels=len(ARGS['target_names'])).to("cuda")
    elif ARGS['model_name'] == 'ufal/robeczech-base':
        model = RobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=len(ARGS['target_names'])).to("cuda")
    elif ARGS['model_name'] == 'xlm-roberta-base' or ARGS['model_name'] == 'xlm-roberta-large':
        model = XLMRobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=len(ARGS['target_names'])).to("cuda")
    elif ARGS['model_name'] == 'facebook/xlm-roberta-xl':
        model = XLMRobertaXLForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=len(ARGS['target_names'])).to("cuda")
    elif ARGS['model_name'] == 'roberta-base':
        model = RobertaForSequenceClassification.from_pretrained(
            ARGS['model_name'], num_labels=len(ARGS['target_names'])).to("cuda")
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
    trainer = CustomTrainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        callbacks=[LogCallback, EarlyStoppingCallback(early_stopping_patience=ARGS['early_stopping_patience'])]
        # callbacks=[LogCallback]
    )

    log.info('Train...')
    trainer.train()
    # log.info(trainer.evaluate())
    model_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    trainer.save_model(model_pth)

    tok_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}-tokenizer'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    tokenizer.save_pretrained(tok_pth)

    log.info('Evaluation...')
    # TODO: resolve text pair
    predictions = np.array(
        [get_probs(test_texts[i], tokenizer, model).cpu().detach().numpy()[0] for i in range(len(test_texts))])
    bf1 = bin_f1(test_labels, predictions)
    cfm = confusion_matrix(test_labels, np.argmax(predictions, -1)).tolist()
    log.info(cfm)


    ci: Dict = bootstrap_michal(test_labels, predictions)

    probs1 = [x[1] for x in predictions]
    ras = roc_auc_score(test_labels, probs1)
    log.info(f"ROC AUC: {ras:.4f}")


    results_pth = Path(ARGS['output_dir'], 'split-{}_{}_id-{}.json'.format(ARGS['split'], ARGS['tag'], ARGS['run_id']))
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        ARGS['output_dir'] = str(ARGS['output_dir'])  # need to convert Path to str before dump
        result = {
            'test_f1': bf1,
            'test_cfm': cfm,
            'train_loss': LOSS,
            'auc': f'{ras:.4f}',
            'dev_f1': F1,
            'dev_cfms': CFM,
            'args': ARGS
        }
        result.update(ci)
        json.dump(result, outfile, ensure_ascii=False)
    rminsidedir(ARGS['output_dir'], 'checkpoint')
    log.info('Finished.')


def main():
    # set seed and GPUs
    set_seed(1)
    no_gpus = set_gpus(ARGS['gpu'])
    ARGS['visible_devices'] = no_gpus

    # run
    run()  # results are saved in split_# dir
    plot_loss()  # only plots for this split


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
        main()
    except Exception as e:
        log.exception(e)
        raise
