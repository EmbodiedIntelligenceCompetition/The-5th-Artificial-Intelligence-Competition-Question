


python -u point_nav_ppo_train.py \
    --root_dir "test" \
    --config_file "gibson_extension/examples/configs/turtlebot_nav.yaml" \
    --agent_config_file "../configs/agent.yaml" \
    --initial_collect_steps 512 \
    --collect_steps_per_iteration 1 \
    --batch_size 512 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 10000 \
    --num_eval_episodes 10 \
    --eval_interval 10000000 \
    --gpu_c 0 \
    --gpu_g 0 \
    --num_parallel_environments 2 \
    --model_ids "Rs_int, Benevolence_0_int" \
    2>&1 | grep -v "Created DrawableGroup" > turtle_output_clamp_igibson2.2.2_multi_envs.log \