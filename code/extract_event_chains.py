from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk
from nltk.stem.snowball import SnowballStemmer
import json
import os
import argparse
from tqdm import tqdm

TMP_PATH = 'data/chains/tmp.txt'


class SRLExtractor:
    srl_model_url = "https://storage.googleapis.com/allennlp-public-models/" \
                    "structured-prediction-srl-bert.2020.12.15.tar.gz"
    stemmer_class = nltk.stem.snowball.SnowballStemmer

    def __init__(self, use_cuda=True):
        self.predictor = Predictor.from_path(self.srl_model_url,
                                             cuda_device=0 if use_cuda else -1)
        self.stemmer = self.stemmer_class('english')

    def select_verb(self, srl_dic):
        """select the most appropriate frame from the predicted frames"""
        words = srl_dic['words']
        stem = ' '.join([self.stemmer.stem(w) for w in words])
        frames = srl_dic['verbs']
        verbs = [d['verb'] for d in frames]

        if any(f'{x} going to' in stem for x in ['was', 'were', 'is', 'are']):
            return frames[0]['verb']
        else:
            frames.sort(key=lambda d: sum(1 for tag in d['tags'] if tag != 'O'),
                        reverse=True)
            return frames[0]['verb']

    def parse_story_events(self, string):
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
        srl_results = self.predictor.predict_batch_json([{'sentence': e} for e in events])
        res = []
        assert len(srl_results) == len(events)
        for srl, e in zip(srl_results, events):
            if len(srl['verbs']) == 0:
                print(f'Event [{e}] SRL failed')
                continue
            verb = self.select_verb(srl)
            res.append({'verb': verb, 'event': e})  # TODO
        return res

    def parse_query_events(self, string):
        string = string.strip()
        for temp_rel in ['starts before', 'starts after',
                         'ends before', 'ends after']:
            if temp_rel in string:
                e1, e2 = string.split(temp_rel)
                e1 = e1.replace('.', '').strip()
                e2 = e2.replace('.', '').strip()
                return e1, e2, temp_rel
        raise ValueError('Invalid input')

    def process_file(self, input_path, output_path):
        with open(input_path, 'r') as fin:
            lines = [line.rstrip() for line in fin]

        n_events = 0
        with open(output_path, 'w') as fout:
            for line in tqdm(lines, total=len(lines)):
                str_event, line = line.split('story:')
                str_story, str_answer = line.split('answer:')
                qe1, qe2, temp_rel = self.parse_query_events(str_event)  # TODO: add query events
                events = self.parse_story_events(str_story)
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


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='data/iid/')
    parser.add_argument('-o', '--output_dir', default='data/chains/')
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    make_dir(TMP_PATH)
    with open(TMP_PATH, 'w') as fout:
        pass

    extractor = SRLExtractor(use_cuda=args.cuda)

    for f in os.listdir(args.input_dir):
        if f.endswith('.txt'):
            input_path = os.path.join(args.input_dir, f)
            output_path = os.path.join(args.output_dir, f.replace('.txt', '.jsonl'))
            make_dir(output_path)
            extractor.process_file(input_path, output_path)


if __name__ == '__main__':
    main()
