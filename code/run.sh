python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_tracie_before_after.py \
--archive-path ../ckpts/temporal-bart/temporal-bart-100k.tar.gz \
--predictor demo_denoising_event_lm_mctaco_before_after_predictor \
--include-package denoising_event_lm \
--cuda-device 0 \
--overrides '{}' \
--input-path ../data/chains/tracie_test.jsonl \
--beams 1 \
--feed-unseen