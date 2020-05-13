set -eux
MODEL_PATH=../ERNIE/ERNIE_weight
TASK_DATA_PATH=../misspelling/data
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export PYTHONPATH=./ernie:${PYTHONPATH:-}
python -u ./ernie/run_sequence_labeling.py \
	--use_cuda true \
	--do_train false \
	--do_val true \
	--do_test true \
	--batch_size 8 \
	--init_checkpoint ./checkpoints/small_word_random/latest \
	--num_labels 100005 \
	--chunk_scheme "IOB" \
	--tokenizer "WordTokenizer" \
        --do_lower_case false \
	--label_map_config ${TASK_DATA_PATH}/full_label_map.json \
	--train_set ${TASK_DATA_PATH}/train_ernie.tsv \
	--dev_set ${TASK_DATA_PATH}/dev_ernie.tsv,${TASK_DATA_PATH}/test_ernie.tsv \
	--test_set ${TASK_DATA_PATH}/dev_ernie.tsv,${TASK_DATA_PATH}/test_ernie.tsv \
	--vocab_path ${TASK_DATA_PATH}/wordVocab.txt \
	--ernie_config_path ${MODEL_PATH}/small_config.json \
	--checkpoints ./checkpoints \
	--save_steps 1000 \
	--weight_decay 0.01 \
	--warmup_proportion 0.0 \
	--validation_steps 1000 \
	--epoch 100 \
	--learning_rate 5e-5 \
        --skip_steps 1 \
	--use_fp16 false \
	--max_seq_len 256 \
	--num_iteration_per_drop_scope 1 \
	--for_cn false \
	--test_save ./ernie_corrector_prediction/small_word_random_dev,./ernie_corrector_prediction/small_word_random_test
#	--init_checkpoint ./checkpoints/detection_best \
