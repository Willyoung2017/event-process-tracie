import json
import pickle
import sys
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from copy import deepcopy
import torch
from torch import nn
import heapq

import argparse

import allennlp
from allennlp.common.checks import check_for_gpu
if allennlp.__version__ == '0.8.5':
    from allennlp.common.util import import_submodules as import_module_and_submodules
elif allennlp.__version__ >= '1.1.0':
    from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive


TEMPORAL_REL_MAP = {
    "starts after": {"positive": "after", "negative": "before"},
    "starts before": {"positive": "before", "negative": "after"},
    "ends after": {"positive": "after", "negative": "before"},
    "ends before": {"positive": "before", "negative": "after"}
}


def normalize_arg_type(arg_type):
    if arg_type[0] in ['R', 'C']:
        return arg_type[2:]
    else:
        return arg_type


def get_flatten_varg_toks(varg):
    varg_toks = [varg['V_toks']] + varg['ARGS_toks']
    varg_span = [varg['V_span']] + varg['ARGS_span']
    varg_type = ['V'] + [normalize_arg_type(arg_type) for arg_type in varg['ARGS_type']]
    assert len(varg_toks) == len(varg_span) and len(varg_toks) == len(varg_type)
    indices = list(range(len(varg_toks)))
    # sort pred/args by their textual order
    indices = sorted(indices, key=lambda x: varg_span[x])
    varg_toks = [varg_toks[i] for i in indices]
    varg_type = [varg_type[i] for i in indices]
    flatten_toks = []
    for i, toks in enumerate(varg_toks):
        flatten_toks.extend(toks)
    return flatten_toks


def chain_str(chain):
    texts = []
    for varg in chain:
        if (not 'Description' in varg) and (not "event" in varg):
            varg['Description'] = " ".join(get_flatten_varg_toks(varg))
        if "event" in varg:
            varg["Description"] = varg["event"]
        texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
    return texts


def check_chain_fulfill_constraints(events, constraints):
    def fulfill_constraint(e1, e2):
        for e in events:
            if e == e1:
                return True
            elif e == e2:
                return False
    return all(fulfill_constraint(e1, e2) for e1, e2 in constraints)


def predict_on_unseen_events(data, predictor, args, file="output.txt"):
    file = open(file, "w")
    question_event_in_context_idx = data['qe2_idx']
    question_event_in_context = data['chain'][question_event_in_context_idx]
    label = data["raw"].split("answer: ")[-1]
    for rel in ["starts after", "starts before", "ends after", "ends before"]:
        if rel in data["raw"]:
            temp_rel = TEMPORAL_REL_MAP[rel][label]
            temp_compartor = rel.split()[0]
            break

    assert temp_rel in {'before', 'after'}
    implicit_event = data["qe1"]
    if temp_rel == 'before':
        constraints = [(implicit_event, question_event_in_context)]
    elif temp_rel == 'after':
        constraints = [(question_event_in_context, implicit_event)]

    test_json = {
        'events': data['chain'],
        'cand_event': implicit_event,
        'beams': args.beams,
        'feed_unseen': args.feed_unseen
    }
    context = data["raw"].split("\t")[0]
    story = context.split(" story: ")[-1]
    query = context.split(" story: ")[0]
    output = predictor.predict_json(test_json)
    print('---'*3, file=file)
    print('##Context##', file=file)
    print(story, file=file)
    print(file=file)
    print('##Question##', file=file)
    print(query, file=file)
    print(file=file)
    print('##Implicit Event##', file=file)
    print(implicit_event, file=file)
    print(file=file)
    print("##Relation##", file=file)
    print("[Implicit]", temp_compartor+" "+temp_rel, "[Explicit]", file=file)
    print(file=file)
    print('---'*3, file=file)
    print("input_repr:", file=file)
    for r in chain_str(output['input_vargs']):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("question_repr:", file=file)
    for r in chain_str([question_event_in_context]):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("cand_repr:", file=file)
    for r in chain_str(output['unseen_vargs']):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("Max: {:.4f} - Min: {:.4f} - Mean: {:.4f} - Std: {:.4f} - Best POS: {:.4f}".format(np.max(output['all_beam_scores']), np.min(output['all_beam_scores']), np.mean(output['all_beam_scores']), np.std(output['all_beam_scores']), output["best_pos_score"]), file=file)
    beam_matches = []
    for b_idx, pred in enumerate(output['beam_pred']):
        if "EVENT_SEP" in pred['pred_vargs'][0]:
            for v in pred['pred_vargs']:
                v.pop("EVENT_SEP")
        assert question_event_in_context in pred['pred_vargs']
        assert implicit_event in pred['pred_vargs']
        match = check_chain_fulfill_constraints(pred['pred_vargs'], constraints)
        beam_matches.append(match)
        print("Beam {:d} (gold: {} - score: {:.4f})".format(b_idx, match, pred['score']), file=file)
        for r in chain_str(pred['pred_vargs']):
            print(r, file=file)
        print(file=file)
    print("\n\n", file=file)
    return beam_matches, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test the predictor above')

    parser.add_argument('--archive-path', type=str, required=True, help='path to trained archive file')
    parser.add_argument('--predictor', type=str, required=True, help='name of predictor')
    parser.add_argument('--weights-file', type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')
    parser.add_argument('--input-path', type=str, nargs='+', help='input data')
    parser.add_argument('--beams', type=int, help='beam size', default=1)
    parser.add_argument('--num_instances', type=int, default=-1,
                        help='number of instances to process')
    parser.add_argument('--feed-unseen', action='store_true', help='whether to feed unseen events as inputs', default=False)

    args = parser.parse_args()

    # Load modules
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_path,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    predictor = Predictor.from_archive(archive, args.predictor)

    data = []
    for path_regex in args.input_path:
        for path in sorted(glob.glob(path_regex)):
            with open(path, 'r') as f:
                for line in f:

                    data.append(json.loads(line))
    if args.num_instances > 0:
        data = data[:args.num_instances]
    print("Num Instances:", len(data))

    total_confusion = {
        "gold positive": {
            "pred positive": 0.,
            "pred negative": 0.
        },
        "gold negative": {
            "pred positive": 0.,
            "pred negative": 0.
        }
    }
    total_correct = 0.
    total_examples = 0
    with open("predictions.txt", "w") as f:
        for d_idx, d in enumerate(tqdm(data)):
            beam_matches, label = predict_on_unseen_events(d, predictor, args)
            if beam_matches[0]:
                pred_temp_rel = label
            else:
                if label == "positive":
                    pred_temp_rel = "negative"
                else:
                    pred_temp_rel = "positive"
            total_confusion['gold '+label]['pred '+pred_temp_rel] += 1
            total_correct += int(beam_matches[0])
            total_examples += 1
            f.write("answer: "+pred_temp_rel+"\n")
    assert sum(pv for gk, gv in total_confusion.items() for pk, pv in gv.items()) == total_examples
    assert sum(pv for gk, gv in total_confusion.items() for pk, pv in gv.items() if gk[5:] == pk[5:]) == total_correct
    print("Acc: {:.4f} ({:.4f} / {:d})".format(total_correct / total_examples, total_correct, total_examples))
    # BEFORE f1
    if sum(pv for pk, pv in total_confusion['gold positive'].items()) > 0:
        recl = total_confusion['gold positive']['pred positive'] / sum(pv for pk, pv in total_confusion['gold positive'].items())
    else:
        recl = 0.
    if sum(gv['pred positive'] for gk, gv in total_confusion.items()) > 0:
        prec = total_confusion['gold positive']['pred positive'] / sum(gv['pred positive'] for gk, gv in total_confusion.items())
    else:
        prec = 0.
    if prec + recl > 0:
        before_f1 = (2 * prec * recl) / (prec + recl)
    else:
        before_f1 = 0.
    print("positive P: {:.4f} - R: {:.4f} - F1: {:.4f}".format(prec, recl, before_f1))
    # AFTER f1
    if sum(pv for pk, pv in total_confusion['gold negative'].items()) > 0:
        recl = total_confusion['gold negative']['pred negative'] / sum(pv for pk, pv in total_confusion['gold negative'].items())
    else:
        recl = 0.
    if sum(gv['pred negative'] for gk, gv in total_confusion.items()) > 0:
        prec = total_confusion['gold negative']['pred negative'] / sum(gv['pred negative'] for gk, gv in total_confusion.items())
    else:
        prec = 0.
    if prec + recl > 0:
        after_f1 = (2 * prec * recl) / (prec + recl)
    else:
        after_f1 = 0.
    print("negative  P: {:.4f} - R: {:.4f} - F1: {:.4f}".format(prec, recl, after_f1))
    macro_f1 = (before_f1 + after_f1) / 2.
    print("Macro F1: {:.4f})".format(macro_f1))
