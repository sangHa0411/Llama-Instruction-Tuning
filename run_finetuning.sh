
# Training script
## Evaluation Dataset
## MMLU : 5-shot
## ARC : 25-shot
## HellaSwag : 10-shot
## TruthfulQA : 5-shot
## GSM8k : 5-shot

python /home/sangha110495/project/Llama-Instruction-Tuning/src/run_finetune.py \
    --model_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
    --tokenizer_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
    --instruction_datasets="[openorca-multiplechoice]" \
    --dataset_sizes="[all]" \
    --evaluation_datasets="[arc,mmlu,hellaswag]" \
    --evaluation_shots="[5,5,5]" \
    --random_seed=42 \
    --padding_side="left" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=2 \
    --sequence_max_length=1024 \
    --generation_max_length=256 \
    --gradient_checkpointing=True \
    --evaluation_strategy="steps" \
    --eval_steps=500 \
    --logging_dir="/home/sangha110495/project/Llama-Instruction-Tuning/logging" \
    --output_dir="/mnt/disks-standard/persist/llm/llama-instruction-tuning/exps/checkpoints" \
    --cache_dir="/mnt/disks-standard/persist/huggingface" \
    --num_train_epochs=3 \
    --weight_decay=1e-2 \
    --warmup_ratio=0.0 \
    --learning_rate=5e-6 \
    --lr_scheduler_type="constant" \
    --logging_steps=100
