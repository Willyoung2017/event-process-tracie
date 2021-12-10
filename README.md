# Introduction
This is our reproduction code repository for the PTNTIME mdoel on the paper "Temporal Reasoning on Implicit Events from Distant Supervision". The code is the same as their official repository on [GitHub](https://github.com/allenai/tracie).

# Dependencies
- python == 3.8
- allennlp==2.1.0
- allennlp-models==2.1.0
- sentence-transformers == 2.1.0
- nltk


# Usage
To extract event chains:
```bash
# w/ GPU:
python code/extract_event_chains.py -o data/chains/

# w/ CPU:
python code/extract_event_chains.py -o data/chains/ --cuda 0
```
Output format:
```json
{
  "raw": "event: Chad looked for his baseball cap starts after he got off the ride story: Chad had gone to an amusement park. He was riding on the roller coaster. Chad was wearing a baseball cap. The baseball cap fell off of Chad's head. Chad found the cap after he got off of the ride.\tanswer: positive",
  "qe1": {
    "verb": "looked",
    "event": "Chad looked for his baseball cap starts"
  },
  "qe2": {
    "verb": "got",
    "event": "he got off the ride"
  },
  "qe1_idx": 5,  // for test intances, qe1_idx is always -1
  "qe2_idx": 4,
  "chain": [
    {
      "verb": "gone",
      "event": "Chad had gone to an amusement park."
    },
    {
      "verb": "riding",
      "event": "He was riding on the roller coaster."
    },
    ...
  ]
}
```

# TRACIE Dataset
We include the TRACIE dataset under `data/`. There are two splits: IID (`data/iid/`) and Uniform-Prior (`data/uniform-prior`).
The data is in a NLI format with each line being `event: [query] story: [context] \t answer: [label] \n`.

# Models
## PtnTime
PtnTime is simply the T5 sequence-to-sequence implementation from Huggingface's transformer (v2.11.0).
Instead of Google's pre-trained weights, PtnTime uses different model weights for T5, which is the only difference comparing to the T5-large baseline.

### Pre-trained Model
Download the entire directory `ptntime-pretrained-model` from [Google Drive](https://drive.google.com/drive/folders/1GirBYMWHJ13zqKl5qPcTjJQNJVtCfVaP?usp=sharing)
and put it under `code/models/ptntime/` 

### Run experiments
We provide the source code as both a Dockerfile and shell scripts. We introduce how to run the shell scripts here.

- Work under the directory `code/models/ptntime` (This is very important as we refer all paths relative to this working directory below.)
- Install requirements by `pip install -r requirements.txt`

To run the T5-large baseline, use `sh run_t5_large_baseline_on_uniform_prior.sh`

To run the PtnTime pre-trained model, use `sh run_ptntime_on_uniform_prior.sh`

## Our Method
With the task reformulation and the extracted temporal chains, our method directly uses [TemporlBART](https://github.com/jjasonn0717/TemporalBART) to run. There are two settings: zero-shot and fine-tuning

### Pre-trained Model
Download the pretrained TemporalBART model from [Google Drive](https://drive.google.com/file/d/1SdSrGhB4KMWIMzbD42GobKQmKPOIuRKL/view?usp=sharing)
and put it under `ckpts/temporal-bart/` 

### Run experiments
To run the zero-shot version, use `sh run.sh` under `code/`

To run the fine-tuning version, use `sh run_finetune.sh` under `code/`

# Output & Evaluations
The output file is in `\output`, the content of each file is like:
```
eval_results_lm.txt                  baseline(PTNTIME) prediction
output-predictions-finetune.txt      fine-tune model output
output-predictions-zero-shot.txt     zero-shot model output
predictions-finetune.txt             fine-tune model predictions
predictions-zero-shot.txt            zero-shot model predictions
```
To evaluate, run `python evaluator.py`.
We provide the predictions from our experiments with under `experiment_result`. We also provide the results of our error analysis under  `error.txt`.

# Citation
See the following paper: 
```
@inproceedings{ZRNKSR21,
    author = {Ben Zhou and Kyle Richardson and Qiang Ning and Tushar Khot and Ashish Sabharwal and Dan Roth},
    title = {Temporal Reasoning on Implicit Events from Distant Supervision},
    booktitle = {NAACL},
    year = {2021},
}
```
