#!/bin/bash

RAW_DATA_DIR=/nas/SNUBH-PSG_signal_extract/signal_extract_12
SAVE_DIR=./save_ours
BATCH_SIZE=4

python train.py \
    --dataset 'ours' \
    --raw_data_dir $RAW_DATA_DIR \
    --sampling_freq 250 \
    --num_nodes 10 \
    --max_seq_len 30 \
    --resolution 2500 \
    --input_dim 1 \
    --output_dim 5 \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --num_workers 8 \
    --model_name 'graphs4mer' \
    --graph_learn_metric "self_attention" \
    --dropout 0.1 \
    --g_conv 'gine' \
    --num_gcn_layers 1 \
    --hidden_dim 128 \
    --temporal_model 's4' \
    --num_temporal_layers 4 \
    --state_dim 64 \
    --bidirectional False \
    --temporal_pool 'mean' \
    --prune_method 'thresh_abs' \
    --thresh 0.1 \
    --knn 3 \
    --activation_fn 'leaky_relu' \
    --graph_pool 'sum' \
    --use_prior False \
    --residual_weight 0.6 \
    --regularizations 'feature_smoothing' 'degree' 'sparse' \
    --feature_smoothing_weight 0.2 \
    --degree_weight 0.2 \
    --sparse_weight 0.2 \
    --save_dir $SAVE_DIR \
    --metric_name 'F1' \
    --eval_metrics 'F1' 'kappa' \
    --metric_avg 'macro' \
    --lr_init 1e-3 \
    --l2_wd 5e-3 \
    --num_epochs 100 \
    --scheduler timm_cosine \
    --t_initial 100 \
    --warmup_t 5 \
    --optimizer adamw \
    --do_train True \
    --balanced_sampling True \
    --accumulate_grad_batches 1 \
    --patience 25 \
    --gpus  1\
    --gpu_id 0 \
