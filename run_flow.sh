#!/bin/bash
# Train the complete-state flow (bounded momentum, see train_flow_complete.py) and test it.
DATA=muons_data_split.h5    # split_by_weight.py output (or muons_data.h5, pooled)
HI_FRAC=0.5                 # fraction of each batch from the high-weight group
NUM_SAMPLES=0   # rows used for training / muons generated for the test; 0 = all
DEVICE=cuda:0
NUM_BINS=12
TAIL_BOUND=9                # the logit momentum's 376 GeV tail sits at ~8.6 sigma
EPOCHS=150
MODEL=outputs/flow_model_complete_ploss.pt

python3 train_flow_complete.py --data $DATA --hi_frac $HI_FRAC --num_samples $NUM_SAMPLES \
    --device $DEVICE --num_bins $NUM_BINS --tail_bound $TAIL_BOUND --epochs $EPOCHS --out $MODEL &&
python3 test_flow.py --model $MODEL --num_samples $NUM_SAMPLES --device $DEVICE --load_hists
