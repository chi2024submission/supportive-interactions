from pathlib import Path
from typing import List, Tuple, Set, Dict, Union, Any

import emoji
import numpy as np

import queries
from algorithms.label_concat_strategies import multi_any_ts0, is_window_certain
from connector import DB

def save_to_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def is_crap(text, crap_ratio):
    """Very slow"""
    return len(text) > 30 and len(emoji.emoji_list(text)) > len(text)*crap_ratio


def compose_training_ex_nhot(texts: List[str], labels: List[List[int]], sep: str) \
        -> Tuple[str, int, List[int], List[int], List[List[int]]]:
    """
        Construct training examples with 4 variants of labels:
            coarse, multi-label, coarse-seq, multi-label-seq
        Should be done in numpy but I'm lazy.
        @param texts: context window of utterance texts
        @param labels: nhot vector of labels (sparse - e.g. [0,1,0,0,0,1]) for each utterance
        @param sep: concat texts separator
    """
    ret_text = sep.join(texts)
    is0 = sum([l[0] for l in labels]) == len(labels)

    # coarse
    ret_coarse: int = 0 if is0 else 1

    # multi-label
    ret_multi: List[int] = [0 for _ in range(len(labels[0]))]
    if is0:  # all utterances are 0 label => example is 0 label
        ret_multi[0] = 1
    else:
        for l in labels:
            for idx, v in enumerate(l):
                if idx > 0 and v > 0: # example is gets the idx label if any utterance has it
                    ret_multi[idx] = 1

    # coarse seq
    ret_coarse_seq: List[int] = [0 if l[0] == 1 else 1 for l in labels]
    # multi-label seq
    ret_multi_seq: List[List[int]] = labels

    return ret_text, ret_coarse, ret_multi, ret_coarse_seq, ret_multi_seq


def to_examples_anylabel_multi(
        rows: List[Tuple[List[str], List[List[int]], List[str], int, int, int]],
        ctx: int,
        sep: str,
        no_small_pcs: bool = False,
        crap_ratio=1.0) \
        -> Tuple[Dict, Dict[str, int]]:
    """
    Transforms a list of utterance-label tuples to training examples.

    @param rows: list of tuples(ordered list of utterances, ordered list of labels, authors, user_id, thread_id, conv_id)
    @param ctx: context length (in character count)
    @param sep: separator between utterances within context
    @param no_small_pcs: stop if window reaches start & contains all 0s
    @returns: set of examples as tuples (text, label, metadata)
    """
    used_windows: Set[str] = set()
    out_coarse: List[Tuple[str, int, int, int]] = []
    out_multi: List[Tuple[str, List[int], int, int]] = []
    out_seq_coarse: List[Tuple[str, List[int], int, int]] = []
    out_seq_multi: List[Tuple[str, List[int], int, int]] = []
    duplicates: Dict[str, int] = {}

    # adds example to output set (if it isn't an exact match)
    def add_example(text, l_coarse, l_multi, l_coarse_seq, l_multi_seq, fauid, tid) -> None:
        if text not in used_windows and not is_crap(text, crap_ratio):
            out_coarse.append((text, l_coarse, fauid, tid))
            out_multi.append((text, l_multi, fauid, tid))
            out_seq_coarse.append((text, l_coarse_seq, fauid, tid))
            out_seq_multi.append((text, l_multi_seq, fauid, tid))
            used_windows.add(text)
        else:
            if text in duplicates:
                duplicates[text] = duplicates[text] + 1
            else:
                duplicates[text] = 1

    for t, l, authors, ui, fi, ti in rows:
        # ignore if only one utterance in dialogue
        if len(t) < 2:
            continue

        # 2 and more utterances
        target_i = len(t) - 1
        while target_i > 0:
            last_i = target_i
            char_cnt = 0
            while last_i > 0 and char_cnt < ctx:
                last_i -= 1
                char_cnt += len(t[last_i])

            add_example(*compose_training_ex_nhot(t[last_i:target_i+1], l[last_i:target_i+1], sep), fi, ti)

            # break if the window gets to the dialog's beginning and labels are all 0s
            if no_small_pcs \
                    and last_i == 0 \
                    and sum([0 if nhot[0] == 1 else 1 for nhot in l[0:target_i]]) == 0: # 1 in the 0th index in nhot vector should be exclusive with other labels
                break
            target_i -= 1
    ret = {
        'coarse': out_coarse,
        'multi': out_multi,
        'seq_coarse': out_seq_coarse,
        'seq_multi': out_seq_multi
    }
    return ret, duplicates


def generate_chi_datasets_ts0(db: DB, ctx_len: int, sep: str, nsp: bool, save_name: str, is_en=False, crap_ratio=1.0) -> None:
    '''
    Generate and save datasets with previous context for CHI 2024 for supportive interactions.
    @param db: DB connector
    @param ctx_len: soft length of the context in chars
    @param sep: separator for individual utterances
    @param nsp: no small pieces - join the last context window into one if there are no more SI instances
    @param save_name: name for the dataset files
    @param is_en: pull the data from the table with english data
    @param crap_ratio: filters examples by the ratio of emojis and other characters
    '''
    # Computed split on donor users
    idx = [(2829, 2941, 3025, 3033, 3034), (2024, 2104, 2214, 2223, 2423, 2425, 2482, 2605, 2666, 2728, 2729, 2731, 2779, 2932, 2990, 3018, 3030)]

    print('Pull the data for splits...')
    cur = db.conn.cursor()
    query = queries.GET_EN_CONVERSATIONS_TAGSET_0_IN_CHI_NHOT if is_en else queries.GET_CONVERSATIONS_TAGSET_0_IN_CHI_NHOT
    cur.execute(query, (idx[1],))
    train: List[Tuple[List[str], List[List[int]], List[str], int, int, int]] = cur.fetchall()
    
    cur.execute(query, (idx[0],))
    test: List[Tuple[List[str], List[List[int]], List[str], int, int, int]] = cur.fetchall()

    print('Saving data...')
    train_examples, _ = to_examples_anylabel_multi(train, ctx_len, sep, nsp, crap_ratio)
    test_examples, _ = to_examples_anylabel_multi(test, ctx_len, sep, nsp, crap_ratio)


    len_coarse = len(test_examples['coarse'])
    assert len_coarse == len(test_examples['multi'])
    assert len_coarse == len(test_examples['seq_coarse'])
    assert len_coarse == len(test_examples['seq_multi'])
    thr = int(len_coarse / 4)
    dev_examples = {
        'coarse': test_examples['coarse'][:thr],
        'multi': test_examples['multi'][:thr],
        'seq_coarse': test_examples['seq_coarse'][:thr],
        'seq_multi': test_examples['seq_multi'][:thr],
    }
    test_examples['coarse'] = test_examples['coarse'][thr:]
    test_examples['multi'] = test_examples['multi'][thr:]
    test_examples['seq_coarse'] = test_examples['seq_coarse'][thr:]
    test_examples['seq_multi'] = test_examples['seq_multi'][thr:]

    save_to_json(Path('outputs', f'split-1_{save_name}.train'), train_examples)
    save_to_json(Path('outputs', f'split-1_{save_name}.dev'), dev_examples)
    save_to_json(Path('outputs', f'split-1_{save_name}.test'), test_examples)

    cur.close()


def compose_training_ex_nhot_twosided(target_text: str, target_label: List[int], texts: List[str], labels: List[List[int]], sep: str) \
        -> Tuple[str, str, int, List[int], List[int], List[List[int]]]:
    """
        Construct training examples with 4 variants of labels:
            coarse, multi-label, coarse-seq, multi-label-seq
        TODO: Should be done in numpy.
        @param texts: context window of utterance texts
        @param labels: nhot vector of labels (sparse - e.g. [0,1,0,0,0,1]) for each utterance
        @param sep: concat texts separator
    """
    context = sep.join(texts)

    # coarse
    ret_coarse: int = 0 if target_label[0] == 1 else 1
    # multi-label
    ret_multi = target_label

    # coarse seq
    ret_coarse_seq: List[int] = [0 if l[0] == 1 else 1 for l in labels]
    # multi-label seq
    ret_multi_seq: List[List[int]] = labels

    return target_text, context, ret_coarse, ret_multi, ret_coarse_seq, ret_multi_seq


def to_examples_nhot_multi_twosided(
        rows: List[Tuple[List[str], List[List[int]], List[str], int, int, int]],
        ctx: int,
        sep: str,
        no_small_pcs: bool = False, crap_ratio=1.0) \
        -> Dict:
    """
    Transforms a list of utterance-label tuples to training examples.

    @param rows: list of tuples(ordered list of utterances, ordered list of labels, authors, user_id, thread_id, conv_id)
    @param ctx: context length (in character count)
    @param sep: separator between utterances within context
    @param no_small_pcs: stop if window reaches start & contains all 0s
    @returns: set of examples as tuples (text, label, metadata)
    """
    used_windows: Dict[str, str] = {}
    out_coarse: List[Tuple[str, str, int, int, int]] = []
    out_multi: List[Tuple[str, str, List[int], int, int]] = []
    out_seq_coarse: List[Tuple[str, str, int, List[int], int, int]] = []
    out_seq_multi: List[Tuple[str, str, List[int], List[List[int]], int, int]] = []

    # adds example to output set (if it isn't an exact match)
    def add_example(text, context, l_coarse, l_multi, l_coarse_seq, l_multi_seq, fauid, tid) -> None:
        if (text in used_windows and used_windows[text] == context) or is_crap(text + context, crap_ratio):
            pass
        else:
            out_coarse.append((text, context, l_coarse, fauid, tid))
            out_multi.append((text, context, l_multi, fauid, tid))
            out_seq_coarse.append((text, context, l_coarse, l_coarse_seq, fauid, tid))
            out_seq_multi.append((text, context, l_multi, l_multi_seq, fauid, tid))
            used_windows[text] = context

    for t, l, authors, ui, fi, ti in rows:
        # only one utterance in dialogue - is not an interaction
        if len(t) < 2:
            continue

        # 2 and more utterances
        target_i = len(t) - 1
        while target_i >= 0:
            low_i = target_i
            hi_i = target_i
            prev_chars = 0
            next_chars = 0
            while low_i > 0 and prev_chars < ctx:
                low_i -= 1
                prev_chars += len(t[low_i])
            while hi_i < len(t) and next_chars < ctx:
                if hi_i != target_i:
                    next_chars += len(t[hi_i])
                hi_i += 1

            add_example(*compose_training_ex_nhot_twosided(t[target_i], l[target_i], t[low_i:hi_i], l[low_i:hi_i], sep), fi, ti)
            target_i -= 1
    ret = {
        'coarse': out_coarse,
        'multi': out_multi,
        'seq_coarse': out_seq_coarse,
        'seq_multi': out_seq_multi
    }
    return ret


def generate_chi2_datasets_ts0(db: DB, ctx_len: int, sep: str, nsp: bool, save_name: str, is_en: bool=False, crap_ratio=1.0) -> None:
    '''
    Generate and save datasets with bi-directional context for CHI 2024 for supportive interactions.
    @param db: DB connector
    @param ctx_len: soft length of the context in chars: in this case the resulting length can be 2x this
    @param sep: separator for individual utterances
    @param nsp: no small pieces - join the last context window into one if there are no more SI instances
    @param save_name: name for the dataset files
    @param is_en: pull the data from the table with english data
    @param crap_ratio: filters examples by the ratio of emojis and other characters
    '''
    # Computed split on donor users
    idx = [(2829, 2941, 3025, 3033, 3034), (2024, 2104, 2214, 2223, 2423, 2425, 2482, 2605, 2666, 2728, 2729, 2731, 2779, 2932, 2990, 3018, 3030)]

    print('Pull the data for splits...')
    cur = db.conn.cursor()
    query = queries.GET_EN_CONVERSATIONS_TAGSET_0_IN_CHI_NHOT if is_en else queries.GET_CONVERSATIONS_TAGSET_0_IN_CHI_NHOT
    cur.execute(query, (idx[1],))
    train: List[Tuple[List[str], List[List[int]], List[str], int, int, int]] = cur.fetchall()
    train_examples = to_examples_nhot_multi_twosided(train, ctx_len, sep, nsp, crap_ratio=crap_ratio)

    cur.execute(query, (idx[0],))
    test: List[Tuple[List[str], List[List[int]], List[str], int, int, int]] = cur.fetchall()
    test_examples = to_examples_nhot_multi_twosided(test, ctx_len, sep, nsp, crap_ratio=crap_ratio)

    len_coarse = len(test_examples['coarse'])
    assert len_coarse == len(test_examples['multi'])
    assert len_coarse == len(test_examples['seq_coarse'])
    assert len_coarse == len(test_examples['seq_multi'])
    thr = int(len_coarse / 4)
    dev_examples = {
        'coarse': test_examples['coarse'][:thr],
        'multi': test_examples['multi'][:thr],
        'seq_coarse': test_examples['seq_coarse'][:thr],
        'seq_multi': test_examples['seq_multi'][:thr],
    }
    test_examples['coarse'] = test_examples['coarse'][thr:]
    test_examples['multi'] = test_examples['multi'][thr:]
    test_examples['seq_coarse'] = test_examples['seq_coarse'][thr:]
    test_examples['seq_multi'] = test_examples['seq_multi'][thr:]

    save_to_json(Path('outputs', f'split-1_{save_name}.train'), train_examples)
    save_to_json(Path('outputs', f'split-1_{save_name}.dev'), dev_examples)
    save_to_json(Path('outputs', f'split-1_{save_name}.test'), test_examples)

    cur.close()


def main():
    conn = DB()

    # generate dataset with previous context
    generate_chi_datasets_ts0(conn, 256, sep=';', nsp=True, save_name='chi_256_02c', is_en=False, crap_ratio=0.2)

    # generate dataset with bi-directional context
    generate_chi2_datasets_ts0(conn, 256, sep=';', nsp=True, save_name='chi2_0_02c', is_en=False, crap_ratio=0.2)
    
    conn.close_connection()


if __name__ == '__main__':
    main()
