# evaluation
import random

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path
import torch

from seq2seq.utils import read_split_data

(train_texts, train_text_pairs, train_labels,
 dev_texts, dev_text_pairs, dev_labels,
 test_texts, test_text_pairs, test_labels) = read_split_data('chi2_256', Path('data/v2'), "1",
                                                                   granularity="seq_coarse")

def verbalize_input(text: str, text_pair: str) -> str:
    return "Utterance: %s\nContext: %s" % (text, text_pair)



checkpoint_path = "trained_models/proud-butterfly-15-seq_coarse"
# checkpoint_path = "train_dir_multi/checkpoint-5500/Sequence2Sequence"
test_batch_size = 32

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

test_predicted = []

# for text, pair, label in tqdm(zip(test_texts, test_text_pairs, test_labels), total=len(test_labels)):
for batch_offset in tqdm(range(0, len(test_labels), test_batch_size)):
    input_text_batch = [verbalize_input(text, pair)
                        for text, pair in zip(test_texts[batch_offset: batch_offset+test_batch_size],
                                             test_text_pairs[batch_offset: batch_offset+test_batch_size])]
    inputs = tokenizer(input_text_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(**inputs)
    decoded = [text.strip() for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    test_predicted.extend(decoded)

pred_int = [1 if l == "positive" else 0 for l in test_predicted]
true_int = [1 if l == "positive" else 0 for l in test_labels]
assert len(pred_int) == len(true_int)

bootstrap_sample = int(len(test_texts)/10)
bootstrap_repeats = 200

evaluations = []

for fscore_method in ["weighted", "micro", "macro"]:
    for repeat in range(bootstrap_repeats):
        sample_idx = random.sample(list(range(len(pred_int))), k=bootstrap_sample)
        pred_sample = [pred_int[idx] for idx in sample_idx]
        true_sample = [true_int[idx] for idx in sample_idx]
        evaluation = f1_score(true_sample, pred_sample, average=fscore_method)
        evaluations.append(evaluation)

    lower = np.quantile(evaluations, q=0.025)
    upper = np.quantile(evaluations, q=0.975)
    mean = np.mean(evaluations)

    print("Coarse-seq F-scores (%s) interval: (%s; %s), mean: %s" % (fscore_method, lower, upper, mean))
