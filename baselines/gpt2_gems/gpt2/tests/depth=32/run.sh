python -m launch --nnodes 1 \
        --node_rank 0 \
        --nproc_per_node=8 \
        main_with_runtime.py \
        --master_addr localhost \
        --module models.depth=8 \
        --train_batch_size 2 \
        --train_data_file /home/ubuntu/data/gpt2/wiki.train.raw \
        --do_train \
        --num_minibatches 200 \
        --gradient_accumulation_steps 1 \
        --config_path tests/depth=8/conf.json --recompute_step
