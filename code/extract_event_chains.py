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
    sp_path = 'data/chains/special_cases.txt'

    def __init__(self, use_cuda=True):
        self.srl = Predictor.from_path(self.srl_model_url,
                                       cuda_device=0 if use_cuda else -1)
        self.stemmer = self.stemmer_class('english')

        make_dir(self.sp_path)
        self.sp_fout = open(self.sp_path, 'w')

    def free(self):
        self.sp_fout.close()

    def select_verb(self, srl_dic):
        """select the most appropriate frame from the predicted frames"""
        words = srl_dic['words']
        stem = ' '.join([self.stemmer.stem(w.lower()) for w in words])
        frames = srl_dic['verbs']
        verbs = [d['verb'] for d in frames]

        if any(f'{x} going to' in stem for x in ['was', 'were', 'is', 'are']):
            return frames[0]['verb']
        else:
            frames.sort(key=lambda d: sum(1 for tag in d['tags'] if tag != 'O'),
                        reverse=True)
            return frames[0]['verb']

    def get_tmp_arg(self, tags):
        res = []
        for i, tag in enumerate(tags):
            if tag == "B-ARGM-TMP":
                cur_group = [i, -1]
                for j in range(i + 1, len(tags) + 1):
                    if j == len(tags) or tags[j] != "I-ARGM-TMP":
                        cur_group[1] = j
                        break
                res.append(cur_group)
        return res

    def split_event(self, sentence):
        sentence = sentence.strip()
        sentence = sentence.strip('.\"\'')
        srl_dic = self.srl.predict(sentence)
        words = srl_dic['words']
        frames = srl_dic['verbs']
        tmp_arg_groups = [self.get_tmp_arg(d['tags']) for d in frames]
        if all(len(groups) == 0 for groups in tmp_arg_groups):
            res = [sentence]
        else:
            tmp_arg_lens = [max([0] + [j - i for i, j in tmp_arg]) for tmp_arg in tmp_arg_groups]
            max_len = max(tmp_arg_lens)
            frame_id = tmp_arg_lens.index(max_len)  # find the frame with the longest temporal argument
            frame = frames[frame_id]
            groups = tmp_arg_groups[frame_id]
            i, j = groups[[j - i for i, j in groups].index(max_len)]
            if i != 0 and j != len(words):
                self.sp_fout.write('*** Warning ***\n')
            tags = frame['tags']
            assert len(tags) == len(words)
            if i > len(words) - j:
                event1 = ' '.join(words[:i])
            else:
                event1 = ' '.join(words[j:])
            event2 = ' '.join(words[i + 1:j])
            if words[i].lower() == 'after':
                res = [event2, event1]
            elif words[i].lower() == 'before':
                res = [event1, event2]
            else:
                res = [sentence]
        self.sp_fout.write(f'Input: {sentence}\n')
        self.sp_fout.write(f'Output: {res}\n')
        self.sp_fout.write(f'\n')
        return res

    def parse_story_events(self, string):
        string = string.strip()
        n_temp = 0
        sentences = nltk.tokenize.sent_tokenize(string)
        events = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if sent == '':
                continue
            if 'before' in sent or 'after' in sent:
                events += self.split_event(sent)
                n_temp += 1
            else:
                events.append(sent)
        srl_results = self.srl.predict_batch_json([{'sentence': e} for e in events])
        res = []
        assert len(srl_results) == len(events)
        for srl, e in zip(srl_results, events):
            if len(srl['verbs']) == 0:
                print(f'Event [{e}] SRL failed')
                continue
            verb = self.select_verb(srl)
            res.append({'verb': verb, 'event': e})
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

    try:
        for f in os.listdir(args.input_dir):
            if f.endswith('.txt'):
                input_path = os.path.join(args.input_dir, f)
                output_path = os.path.join(args.output_dir, f.replace('.txt', '.jsonl'))
                make_dir(output_path)
                extractor.process_file(input_path, output_path)
    finally:
        extractor.free()


if __name__ == '__main__':
    main()
