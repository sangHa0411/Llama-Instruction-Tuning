# Insturction-Tuning command example
python /home/sangha110495/project/Llama-Instruction-Tuning/src/run_finetune.py \
    --model_path="/mnt/disks/persist-standard/llama/llama-2-7b-hf" \
    --tokenizer_path="/mnt/disks/persist-standard/llama/llama-2-7b-hf" \
    --instruction_datasets="[arc,mmlu,gsm8k,alpaca,cot-collection,slimorca]" \
    --dataset_sizes="[3000,3000,3000,1000,2000,2000]" \
    --evaluation_datasets="[truthful_qa-generation,truthful_qa-multiple_choice,gsm8k,arc,mmlu]" \
    --evaluation_shots="[0,0,5,10,5]" \
    --random_seed=42 \
    --padding_side="left" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=2 \
    --sequence_max_length=512 \
    --eval_sequence_max_length=1536 \
    --generation_max_length=128 \
    --gradient_checkpointing=True \
    --evaluation_strategy="steps" \
    --eval_steps=500 \
    --logging_dir="/home/sangha110495/project/Llama-Instruction-Tuning/logging" \
    --output_dir="/mnt/disks/persist-standard/llm/llama-instruction-tuning/exps/checkpoints" \
    --cache_dir="/mnt/disks/persist-standard/huggingface" \
    --num_train_epochs=3 \
    --weight_decay=1e-2 \
    --warmup_ratio=0.0 \
    --learning_rate=3e-5 \
    --lr_scheduler_type="constant" \
    --logging_steps=100
