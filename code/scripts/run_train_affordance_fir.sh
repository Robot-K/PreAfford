CUDA_VISIBLE_DEVICES=0 python train/train_affordance.py \
    --exp_suffix 17 \
    --category_types part  \
    --model_version model_aff_fir \
    --actor_version model_actor_fir   \
    --actor_path  ../logs/actor/exp-model_actor_fir-part-15 \
    --actor_eval_epoch 35 \
    --critic_version model_critic_fir   \
    --critic_path  ../logs/critic/exp-model_critic_fir-part-15   \
    --critic_eval_epoch 64 \
    --offline_data_dir ../data/PreGrasp_train_6 \
    --offline_data_dir2 ../data/PreGrasp_train_5 \
    --val_data_dir ../data/PreGrasp_val_3 \
    --train_buffer_max_num 15000  \
    --val_buffer_max_num 2000 \
    --feat_dim 160   \
    --batch_size 50  \
    --lr 0.0015      \
    --lr_decay_every 1000 \
    --z_dim 32      \
    --rv_cnt 20    \
    --topk  20   \
    --succ_proportion 0.4 \
    --fail_proportion 0.85 \
    --coordinate_system cambase \
    --use_boxed_pc  
