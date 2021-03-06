python -m launch --nnodes=1 --nproc_per_node=4 main_with_runtime.py \
        --master_addr=localhost \
        --module models.bert48.depth=4 \
        --max_seq_length 128 \
        --train_batch_size 16 \
        --train_path /home/ubuntu/data/bert/enwiki_corpus_for_bert.200K.postprocess.txt \
        --bert_config_path configs/bert_config_bert-large-uncased.json \
        --vocab_path /home/ubuntu/data/bert/vocab.txt \
        --do_train \
        --on_memory \
        --do_lower_case \
        --num_minibatches 256 \
        --gradient_accumulation_steps 1 --recompute_step --config_path tests/depth=4/conf.json
