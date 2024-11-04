# Stage 1: training lq adapter
accelerate launch --num_processes <num_of_gpus> train_stage1_adapter.py \
    --output_dir <your/output/path> \
    --train_data_dir <your/data/path> \
    --logging_dir <your/logging/path> \
    --pretrained_model_name_or_path <your/sdxl/path> \
    --feature_extractor_path <your/dinov2/path> \
    --save_only_adapter \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 1000 \
    --lr_scheduler cosine \
    --lr_num_cycles 1 \
    --resume_from_checkpoint latest