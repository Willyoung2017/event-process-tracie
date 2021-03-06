from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.stem.snowball import SnowballStemmer
import json
import os
import argparse
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
from tqdm import tqdm
from typing import Dict, List


class SRLExtractor:
    srl_model_url = "https://storage.googleapis.com/allennlp-public-models/" \
                    "structured-prediction-srl-bert.2020.12.15.tar.gz"
    stemmer_class = nltk.stem.snowball.SnowballStemmer

    def __init__(self, use_cuda=True, log_path='data/chains/extractor.log'):
        self.srl = Predictor.from_path(self.srl_model_url,
                                       cuda_device=0 if use_cuda else -1)
        self.stemmer = self.stemmer_class('english')
        self.lemmatizer = WordNetLemmatizer()
        self.bert = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0' if use_cuda else 'cpu')
        self.stopwords = nltk.corpus.stopwords.words('english')

        make_dir(log_path)
        self.log_fout = open(log_path, 'w')

    def free(self):
        self.log_fout.close()

    def select_verb(self, srl_dic, sentence):
        """select the most appropriate frame from the predicted frames"""
        words = srl_dic['words']
        stem = ' '.join([self.stemmer.stem(w.lower()) for w in words])
        frames = srl_dic['verbs']
        verbs = [d['verb'] for d in frames]
        if len(frames) == 0:
            self.log_fout.write('[SRL]\n')
            self.log_fout.write(f'sentence: {sentence}\n')
            self.log_fout.write(f'SRL: {srl_dic}\n')
            self.log_fout.write(f'\n')
            return None
        elif any(f'{x} going to' in stem for x in ['was', 'were', 'is', 'are']):
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
        is_tmp = [False]
        if all(len(groups) == 0 for groups in tmp_arg_groups):
            res = [sentence]
            temp_rel = 'NULL'
        else:
            tmp_arg_lens = [max([0] + [j - i for i, j in tmp_arg]) for tmp_arg in tmp_arg_groups]
            max_len = max(tmp_arg_lens)
            frame_id = tmp_arg_lens.index(max_len)  # find the frame with the longest temporal argument
            frame = frames[frame_id]
            groups = tmp_arg_groups[frame_id]
            i, j = groups[[j - i for i, j in groups].index(max_len)]
            tags = frame['tags']
            assert len(tags) == len(words)
            if i > len(words) - j:
                event1 = ' '.join(words[:i]).strip(' \t\n.,"\'')
            else:
                event1 = ' '.join(words[j:]).strip(' \t\n.,"\'')
            event2 = ' '.join(words[i + 1:j]).strip(' \t\n.,"\'')
            srl = self.srl.predict_batch_json([{'sentence': event1}, {'sentence': event2}])
            if len(srl[0]['verbs']) == 0 or len(srl[1]['verbs']) == 0:
                res = [sentence]
                temp_rel = 'NULL'
            elif j - i > 1 and words[i].lower() == 'after':
                res = [event2, event1]
                temp_rel = 'AFTER'
                is_tmp = [True, False]
            elif j - i > 1 and words[i].lower() == 'before':
                res = [event1, event2]
                temp_rel = 'BEFORE'
                is_tmp = [False, True]
            # elif j - i > 1 and words[i].lower() == 'when':
            #     res = [event2, event1]
            #     temp_rel = 'WHEN'
            else:
                res = [sentence]
                temp_rel = 'NULL'

            if temp_rel != 'NULL':
                if i != 0 and j != len(words):
                    self.log_fout.write('*** Warning ***\n')
                self.log_fout.write(f'[SPLIT_TMP_{temp_rel}]\n')
                self.log_fout.write(f'Input: {sentence}\n')
                self.log_fout.write(f'Output: {res}\n')
                self.log_fout.write(f'\n')
        assert len(res) == len(is_tmp)
        return res, is_tmp

    def parse_story_events(self, string):
        string = string.strip()
        n_temp = 0
        sentences = nltk.tokenize.sent_tokenize(string)
        events = []
        is_tmp = []
        for i, sent in enumerate(sentences):
            sent = sent.strip(' \t\n.,"\'')
            if sent == '':
                continue
            if any(x in sent for x in ['before', 'after']):
                split, split_is_tmp = self.split_event(sent)
                events += split
                is_tmp += split_is_tmp
                n_temp += 1
            else:
                events.append(sent)
                is_tmp.append(False)

        srl_results = self.srl.predict_batch_json([{'sentence': e} for e in events])
        res = []
        tmp_indexes = []
        assert len(srl_results) == len(events) == len(is_tmp)
        for srl, e, tmp in zip(srl_results, events, is_tmp):
            verb = self.select_verb(srl, e)
            if verb is None:
                continue
            if tmp:
                tmp_indexes.append(len(res))
            res.append({'V_toks': [verb], 'event': e})
        assert len(res) >= 1
        return res, tmp_indexes

    def get_any_verb(self, tokens):
        assert len(tokens) >= 1
        tagged = nltk.pos_tag(tokens)
        verb = tokens[0]
        for w, tag in tagged:
            if tag.startswith('VB'):
                verb = w
                break

        self.log_fout.write('[SRL-HARD]\n')
        self.log_fout.write(f'tokens: {tokens}\n')
        self.log_fout.write(f'verb: {verb}\n')
        self.log_fout.write(f'\n')
        return verb

    def parse_query_events(self, string):
        string = string.strip()
        for comparator in ['starts', 'ends']:
            for temp in ['before', 'after']:
                temp_rel = f'{comparator} {temp}'
                if temp_rel in string:
                    e1, e2 = string.split(temp_rel)
                    e1 = e1.strip(' \t\n.,"\'')
                    e2 = e2.strip(' \t\n.,"\'')
                    srl1, srl2 = self.srl.predict_batch_json([{'sentence': e1}, {'sentence': e2}])
                    v1 = self.select_verb(srl1, e1)  # v1 == None if SRL failed
                    v2 = self.select_verb(srl2, e2)  # v2 == None if SRL failed
                    e1 = e1 + ' ' + comparator  # event1 -> event1 starts/ends
                    e1 = {'V_toks': [v1], 'event': e1}
                    e2 = {'V_toks': [v2], 'event': e2}
                    return e1, e2, temp_rel
        raise ValueError('Invalid input')

    def locate_event_bert(self, event: dict, event_chain: List[dict]):
        """Event coreference using SentenceBERT"""
        sentences = [event['event']] + [d['event'] for d in event_chain]
        embeddings = self.bert.encode(sentences)
        cos_sim = util.cos_sim(embeddings[:1], embeddings[1:])
        pos = int(cos_sim[0].cpu().numpy().argmax())
        return pos

    def locate_event_verb(self, event: dict, event_chain: List[dict]):
        """Event coreference by comparing verbs"""
        srl = self.srl.predict_batch_json([{'sentence': event['event']}]
                                          + [{'sentence': d['event']} for d in event_chain])
        v_event = {self.lemmatizer.lemmatize(d['verb'].lower(), 'v') for d in srl[0]['verbs'] if
                   d['verb'] not in self.stopwords}
        v_chain = [{self.lemmatizer.lemmatize(d['verb'].lower(), 'v') for d in dd['verbs'] if
                    d['verb'] not in self.stopwords} for dd in srl[1:]]
        assert len(v_chain) == len(event_chain)
        scores = [len(v_event & vs) for vs in v_chain]
        max_score = max(scores)
        pos = scores.index(max(scores))
        return pos, max_score

    def locate_event(self, event: dict, event_chain: List[dict]):
        """A simple event coreference system based on stemming and longest-common-subsequence matching"""
        assert len(event_chain) >= 1
        event_tok = nltk.word_tokenize(event['event'].lower())
        chain_tok = [nltk.word_tokenize(d['event'].lower()) for d in event_chain]
        event_stem = [self.stemmer.stem(self.lemmatizer.lemmatize(w, 'v')) for w in event_tok]
        chain_stem = [[self.stemmer.stem(self.lemmatizer.lemmatize(w, 'v')) for w in e] for e in chain_tok]
        scores = []
        for e_chain in chain_stem:
            matcher = SequenceMatcher(a=event_stem, b=e_chain)
            scores.append(sum(block.size for block in matcher.get_matching_blocks()))
        max_score = max(scores)
        pos = scores.index(max_score)
        if (max_score / len(event_stem)) <= 0.5:  # hard cases, resolve coreferene SentenceBERT
            pos2, score2 = self.locate_event_verb(event, event_chain)
            if score2 >= 1 and event['V_toks'][0] not in self.stopwords:
                pos = pos2
            else:
                pos = self.locate_event_bert(event, event_chain)
            self.log_fout.write('[COREF]\n')
            self.log_fout.write(f'event: {event["event"]}\n')
            self.log_fout.write(
                f'chain: {[("***" * int(i == pos)) + e["event"] for i, e in enumerate(event_chain)]}\n')
            self.log_fout.write(f'\n')
        return pos

    def process_file(self, input_path, output_path, is_test):
        with open(input_path, 'r') as fin:
            lines = [line.rstrip() for line in fin]

        n_events = 0
        with open(output_path, 'w') as fout:
            for line in tqdm(lines, total=len(lines)):
                str_event, remains = line.replace('event:', '').split('story:')
                str_story, str_answer = remains.split('answer:')
                qe1, qe2, temp_rel = self.parse_query_events(str_event)
                events, tmp_indexes = self.parse_story_events(str_story)

                # find the location of qe2 (explicit event)
                pos = self.locate_event(qe2, events)

                # delete temporal arguments that are not query events
                tmp_indexes = [i for i in tmp_indexes if i != pos]
                events = [e for i, e in enumerate(events) if i not in tmp_indexes]
                pos -= sum(1 for i in tmp_indexes if i < pos)
                assert len(events) >= 1

                # insert qe1 (implicit event)
                if not is_test:
                    assert ('positive' in str_answer) ^ ('negative' in str_answer)
                    qe1_before_qe2 = ('before' in temp_rel and 'positive' in str_answer) \
                                     or ('after' in temp_rel and 'negative' in str_answer)
                    if qe1_before_qe2:
                        idx1, idx2 = pos, pos + 1
                        events.insert(pos, qe1)
                    else:
                        idx1, idx2 = pos + 1, pos
                        events.insert(pos + 1, qe1)
                else:  # for test instances, do not insert the implicit event
                    idx1, idx2 = -1, pos
                instance = {
                    'raw': line,
                    'qe1': qe1,
                    'qe2': qe2,
                    'qe1_idx': idx1,
                    'qe2_idx': idx2,
                    'chain': events,
                }
                n_events += len(events)
                fout.write(json.dumps(instance) + '\n')

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
    parser.add_argument('-v', '--version', default='v2')
    parser.add_argument('--log_path', default='data/chains/extractor_{version}.log')
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    extractor = SRLExtractor(use_cuda=args.cuda,
                             log_path=args.log_path.format(version=args.version))

    try:
        for f in os.listdir(args.input_dir):
            if f.endswith('.txt'):
                assert ('train' in f) ^ ('test' in f)
                is_test = 'test' in f
                input_path = os.path.join(args.input_dir, f)
                output_path = os.path.join(args.output_dir, f.replace('.txt', f'_{args.version}.jsonl'))
                make_dir(output_path)
                extractor.process_file(input_path, output_path, is_test=is_test)
    finally:
        extractor.free()


if __name__ == '__main__':
    main()
