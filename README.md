# Introduction
This is our reproduction code repository for the PTNTIME mdoel on the paper "Temporal Reasoning on Implicit Events from Distant Supervision". The code is the same as their official repository on [GitHub](https://github.com/allenai/tracie).

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

Both scripts will create a result file `experiment_result/eval_results_lm.txt`. To evaluate, run `python evaluator.py`.
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
