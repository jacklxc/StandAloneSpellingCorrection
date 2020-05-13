set -eux
MODEL_PATH=../ERNIE/ERNIE_weight
TASK_DATA_PATH=../misspelling/new_data
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export PYTHONPATH=./ernie:${PYTHONPATH:-}
python -u ./ernie/run_misspelling.py \
	--use_cuda true \
	--do_train true \
	--do_val true \
	--do_test false \
	--batch_size 16 \
	--init_pretraining_params ${MODEL_PATH}/params \
	--num_labels 3 \
	--chunk_scheme "IOB" \
	--label_map_config ${TASK_DATA_PATH}/label_map.json \
	--train_set ${TASK_DATA_PATH}/debug_detection.tsv \
	--dev_set ${TASK_DATA_PATH}/debug_detection.tsv \
	--test_set ${TASK_DATA_PATH}/debug_detection.tsv \
	--vocab_path ${MODEL_PATH}/vocab.txt \
	--ernie_config_path ${MODEL_PATH}/ernie_config.json \
	--checkpoints ./checkpoints \
	--save_steps 100 \
	--weight_decay 0.01 \
	--warmup_proportion 0.0 \
	--validation_steps 5000 \
	--epoch 100 \
	--learning_rate 5e-5 \
    --skip_steps 1 \
	--use_fp16 false \
	--max_seq_len 256 \
	--num_iteration_per_drop_scope 1 \
	--for_cn false \
	--test_save ./ernie_detector_prediction/debug
#	--init_checkpoint ./checkpoints/detection_best \
