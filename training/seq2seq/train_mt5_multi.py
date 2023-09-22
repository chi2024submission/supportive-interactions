# TODO: run this script from project root, added to python path:
# cd irtisdb
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# CUDA_VISIBLE_DEVICES=xx python seq2seq/train_mt5_coarse.py

from pathlib import Path

import wandb
from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import SequentialSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy

from seq2seq.utils import read_split_data

wandb.init(project="irtisdb")

def verbalize_input(text: str, text_pair: str) -> str:
    return "Utterance: %s\nContext: %s" % (text, text_pair)


training_arguments = AdaptationArguments(output_dir="train_dir_multi",
                                         learning_rate=2e-5,  # we set LR=2e-4 for pre-training experiments
                                         # stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=6,  # TODO: set
                                         eval_steps=500,  # TODO: set
                                         logging_steps=10,
                                         save_steps=500,
                                         num_train_epochs=50,
                                         evaluation_strategy="steps",
                                         save_total_limit=6,
                                         stopping_patience=5)

# lang_module = LangModule("stas/mt5-tiny-random")  # TODO adjust
lang_module = LangModule("google/mt5-base")

(train_texts, train_text_pairs, train_labels,
 dev_texts, dev_text_pairs, dev_labels, _, _, _) = read_split_data('chi2_256', Path('data/v2'), "1",
                                                                   granularity="seq_multi")

seq_obj = Sequence2Sequence(lang_module,
                             texts_or_path=[verbalize_input(t, pair) for t, pair in zip(train_texts, train_text_pairs)],
                             labels_or_path=train_labels,
                             val_texts_or_path=[verbalize_input(t, pair) for t, pair in zip(dev_texts, dev_text_pairs)],
                             val_labels_or_path=dev_labels,
                             val_evaluators=[BLEU()],
                             batch_size=6,  # TODO: set
                             )

schedule = SequentialSchedule(objectives=[seq_obj], args=training_arguments)

# TODO: evaluate multiclass seq2seq
adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
