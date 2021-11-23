from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk
import json
import os
import argparse
from tqdm import tqdm

TMP_PATH = 'data/chains/tmp.txt'


def parse_story_events(string, predictor):
    string = string.strip()
    string = string.strip('.')
    n_temp = 0
    sentences = nltk.tokenize.sent_tokenize(string)
    events = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent == '':
            continue
        if 'before' in sent or 'after' in sent:  # TODO
            with open(TMP_PATH, 'a') as fout:
                fout.write(sent + '\n')
            n_temp += 1
        events.append(sent)
    srl_results = predictor.predict_batch_json([{'sentence': e} for e in events])
    res = []
    assert len(srl_results) == len(events)
    for srl, e in zip(srl_results, events):
        if len(srl['verbs']) == 0:
            print(f'Event [{e}] SRL failed')
            continue
        res.append({'verb': srl['verbs'][0]['verb'], 'event': e})  # TODO
    return res


def parse_query_events(string, predictor):
    string = string.strip()
    for temp_rel in ['starts before', 'starts after',
                     'ends before', 'ends after']:
        if temp_rel in string:
            e1, e2 = string.split(temp_rel)
            e1 = e1.replace('.', '').strip()
            e2 = e2.replace('.', '').strip()
            return e1, e2, temp_rel
    raise ValueError('Invalid input')


def extract_temporal_chains(input_path, output_path, predictor):
    with open(input_path, 'r') as fin:
        lines = [line.rstrip() for line in fin]

    n_events = 0
    with open(output_path, 'w') as fout:
        for line in tqdm(lines, total=len(lines)):
            str_event, line = line.split('story:')
            str_story, str_answer = line.split('answer:')
            qe1, qe2, temp_rel = parse_query_events(str_event, predictor)  # TODO: add query events
            events = parse_story_events(str_story, predictor)
            n_events += len(events)
            fout.write(json.dumps(events) + '\n')

    print()
    print(f'avg_event_per_instance: {n_events / len(lines)}')
    print()
    print(f'Event chains saved to {output_path}')


def make_dir(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='data/iid/')
    parser.add_argument('-o', '--output_dir', default='data/chains/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    url = "https://storage.googleapis.com/allennlp-public-models/" \
          "structured-prediction-srl-bert.2020.12.15.tar.gz"
    predictor = Predictor.from_path(url, cuda_device=0)

    make_dir(TMP_PATH)
    with open(TMP_PATH, 'w') as fout:
        pass

    for f in os.listdir(args.input_dir):
        if f.endswith('.txt'):
            input_path = os.path.join(args.input_dir, f)
            output_path = os.path.join(args.output_dir, f.replace('.txt', '.jsonl'))
            make_dir(output_path)
            extract_temporal_chains(input_path, output_path, predictor)


if __name__ == '__main__':
    main()
