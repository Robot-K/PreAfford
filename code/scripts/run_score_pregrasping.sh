CUDA_VISIBLE_DEVICES=0 python score_pregrasping.py \
    --exp_suffix 5 \
    --aff_version model_aff_sec   \
    --categories Laptop,Scissors,Cap,Keyboard2,Phones \
    --aff_path  ../logs/affordance/exp-model_aff_sec-part-8   \
    --aff_eval_epoch 64 \
    --actor_version model_actor_sec   \
    --actor_path  ../logs/actor/exp-model_actor_sec-part-8  \
    --actor_eval_epoch 40 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-model_critic_sec-part-8 \
    --critic_eval_epoch 14 \
    --offline_data_dir ../data/PreGrasp_train_5  \
    --feat_dim 160   \
    --batch_size 1  \
    --coordinate_system cambase \
    --use_boxed_pc  \
    --aff_topk 0.005 \
    --num_ctpt 20 \
    --rvs_proposal 20 \
    --critic_topk1 5 \
    --device cuda:0
    # --aff_path  ../logs/affordance/exp-model_aff_sec-part-13   \
    # --aff_eval_epoch 42 \
    # --actor_version model_actor_sec   \
    # --actor_path  ../logs/actor/exp-model_actor_sec-part-13  \
    # --actor_eval_epoch 55 \
    # --critic_version model_critic_sec   \
    # --critic_path  ../logs/critic/exp-model_critic_sec-part-13 \
    # --critic_eval_epoch 36 \