from typing import List


def multi_any_ts0(x: int, y: int, min_incl: int, max_excl: int) -> List[int]:
    ret = []
    if x >= max_excl:
        x = 0
    if y >= max_excl:
        y = 0
    if x == min_incl and y == min_incl:
        ret.append(min_incl)
    elif x == y:
        ret.append(x)
    else:
        ret.append(x)
        ret.append(y)
    return ret


def _0_or_lower(x: int, max_exclusive: int) -> int:
    if x >= max_exclusive:
        x = 0
    return x


def chi_gs_any_multi(l1: int, l2: int, gs: List[int], is_checked_sup: bool, min_incl: int, max_excl: int) -> List[int]:
    """
    @param l1: label 1
    @param l2: label 2
    @param gs: gold standard labels. {} means two things: no gs, gs with 0 label
    @param is_checked_sup: DB field is_decided_by_supervisor_tagset_0
    @param min_incl: should be 0 label which is special (cannot be multilabel)
    @param max_excl: upper limit for the label idx, i.e. TS0 is 1-5, so this would be 6
    """
    ret = []
    l1 = _0_or_lower(l1, max_excl)
    l2 = _0_or_lower(l2, max_excl)
    gs = [g for g in gs if g < max_excl]

    # if GS exists, respect it
    if len(gs) > 0:
        return gs
    if is_checked_sup: # ...and gs={}
        return [min_incl]

    # there is no GS
    if l1 == min_incl and l2 == min_incl:
        ret.append(min_incl)
    elif l1 == l2:
        ret.append(l1)
    else:
        if l1 > min_incl:
            ret.append(l1)
        if l2 > min_incl:
            ret.append(l2)
    return ret


def nhot_to_coarse(label: List[int]):
    if label[0] == 1:
        l = 0
    elif sum(label[1:]) > 0:
        l = 1
    else:
        raise ValueError(label)
    return l


def is_window_certain(a1: List[int], a2: List[int], is_dec: List[bool]) -> bool:
    for a,b,d in zip(a1, a2, is_dec):
        if a != b and d is False:
            return False
    return True
