# Llama-Instruction-Tuning
* This repository was implemented using Jax/Flax.
* This fine-tuning code supports model parallel, so you can train Llama2-7B model on **TPU V3-8**.
  * I haven't run this code on other tpu devices like TPU V3-32, TPU V3-256 or gpu devices yet. So i am not sure this code running on those devices without errors.
* You can easly do instruction tuning Llama2 model with diverse instruction datasets and evaluation benchmarks which consist of Open LLM Leaderboard

---
<br>

## Possible instruction datasets
  * These are instruction datasets and evaluation datasets which this repository supports, If you want to add other datasets, you should change ```utils/loader.py```, ```utils/preprocessor.py``` and ```utils/eval_preprocessor.py```
  * **Instruction Tuning**
    * Alpaca(https://huggingface.co/datasets/tatsu-lab/alpaca)
    * Cot-Collection(https://huggingface.co/datasets/kaist-ai/CoT-Collection)
    * SlimOrca(https://huggingface.co/datasets/Open-Orca/SlimOrca)
    * Openorca-multiplechoice-10k(https://huggingface.co/datasets/beaugogh/openorca-multiplechoice-10k)
    * MMLU(https://huggingface.co/datasets/cais/mmlu)
    * ARC(https://huggingface.co/datasets/ai2_arc)
    * GSM8k(https://huggingface.co/datasets/gsm8k)
    * Winogrande(https://huggingface.co/datasets/winogrande)
    * Hellaswag(https://huggingface.co/datasets/Rowan/hellaswag)
  * **Evaluation**
    * MMLU(https://huggingface.co/datasets/cais/mmlu)
    * ARC(https://huggingface.co/datasets/ai2_arc)
    * GSM8k(https://huggingface.co/datasets/gsm8k)
    * Winogrande(https://huggingface.co/datasets/winogrande)
    * Hellaswag(https://huggingface.co/datasets/Rowan/hellaswag)

<br>
<br>

## Prompt
  * This is prompt i used for instruction tuning. If you want to change format, you should change prompt in ```utils/preprocessor.py``` and ```eval/eval_preprocessor.py```.
    * I used ```\n\n``` for dataset column delimiter. And after each column, i put ```\n``` and then wrote content of that column.
    * I used ```\n\n\n\n``` for few-shot delimiter.
    * You can change this delimiter in ```utils/preprocessor.py```, ```eval/eval_preprocessor.py``` and ```utils/trainer.py```.
  * These are instruction examples.
    * Simple instruction example from Alpaca dataset.
        ```
        ### INSTRUCTION:
        What would be the best type of exercise for a person who has arthritis?
        
        ### RESPONSE:
        For someone with arthritis, the best type of exercise would be low-impact activities like yoga, swimming, or walking. These exercises provide the benefits of exercise without exacerbating the symptoms of arthritis.
        ```
    * Multiple-choice instruction example from MMLU dataset.
        ```
        ### QUESTION:
        Rules in the reading room Hello, everyone. Welcome to the school reading room. We hope you have a good time here. Before you go into the reading room, there are some rules you need to keep. 1.The reading room is open from 8:00 a.m. to 5:00 p.m. from Monday to Friday. 2. Don't take your bag into the reading room. 3. Don't talk loudly in the reading room. 4. Don't take any food or drink into the reading room. 5. Take only one book at a time. After you finish reading the book, you must put it back and then you can take another one. Don't take many books to your seat. 6. Before you leave, you must the book to the bookshelf. You can't take any book out of the reading room. How long is the reading room open every day?
        
        ### CHOICES:
        (0): Ten hours. (1): Nine hours. (2): Seven hours. (3): Eight hours.
        
        ### ANSWER:
        Nine hours.
        ```
    * Binary-choice instruction example from Winogrande dataset.
        ```
        ### SENTENCE:
        While packing for hunting Chris made sure to bring his bag and not his knife because the _ was nonessential.
        
        ### OPTION1:
        bag
        
        ### OPTION2:
        knife
        
        ### ANSWER:
        knife
        ```

<br>
<br>

## Select and combine intruction datasets 
  * You can easly select and combine instruction datasets using ```instruction_datasets``` and ```dataset_sizes``` columns.
  * This is example of fine-tuning command.
    ```shell
    python /home/sangha110495/project/Llama-Instruction-Tuning/src/run_finetune.py \
        --model_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --tokenizer_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --instruction_datasets="[arc,mmlu,slimorca]" \
        --dataset_sizes="[all,10%,1000]"
    ```
    * those command means use all train data from Arc, 10% train data for MMLU, and 1000 train data from Slimorca.
  * As you can see, dataset_sizes column supports three type of dataset_sizes
    * ```all``` : Use all data.
    * ```K``` : Use random sampling and extract exactly K data. | [0 ~ Max size of dataset]
    * ```K%``` : Use random sampling and extract exactly K% of all data. | [0% ~ 100%]

<br>
<br>

## Evaluation
  * You can easly select evaluation benchmarks and select few-shot size using ```evaluation_datasets``` and ```evaluation_shots``` columns.
  * This is example of fine-tuning command.
    ```shell
    python /home/sangha110495/project/Llama-Instruction-Tuning/src/run_finetune.py \
        --model_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --tokenizer_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --instruction_datasets="[openorca-multiplechoice]" \
        --dataset_sizes="[all]" \
        --evaluation_datasets="[arc,mmlu,hellaswag]" \
        --evaluation_shots="[3,4,5]"
    ```
    * Those command means evaluate Arc, MMLU and Hellaswag benchmarks regularly while training.
    * Use 3 shot for Arc benchmark, 4 shot for MMLU benchmark, and 5 shot for Hellaswag benchmark.
  * **Notice**
    * If you use few-shot for evaluation, there are cases which length is more than sequence_max_length. In these case, ```eval/eval_preprocessor.py``` deletes truncated shots and uses only perfect shots.
    * For example, even though you give 5 shots, evaL_preprocessor can make some examples less than 5 shots.
    * 2-Shot example from MMLU dataset
      ```
      ### QUESTION:
      The International Space Station (ISS) circles the Earth approximately 410 km above the ground. Find the best estimate for the orbital speed of the ISS:
      
      ### CHOICES:
      (0): 19000 km/h (1): 21000 km/h (2): 28000 km/h (3): 32000 km/h
      
      ### ANSWER:
      28000 km/h
      
      
      
      ### QUESTION:
      World population tends to be concentrated
      
      ### CHOICES:
      (0): in continental interiors. (1): on continental margins. (2): in the desert. (3): in the tropical lowlands and river valleys.
      
      ### ANSWER:
      on continental margins.
      
      
      
      ### QUESTION:
      The rehabilitation of old, rundown inner-city neighborhoods by middle- and high-income people is called
      
      ### CHOICES:
      (0): urbanization. (1): gentrification. (2): suburbanization. (3): multiplier effect.
      
      ### ANSWER:
      ```

<br>
<br>

## Fine-tuning code
  * This is detail of fine-tuning code in ```run_finetun.py```. I made fine-tuning code like Huggingface style as possible as i can.
  * Fine-tuning code
    ```python
    # Setting Device & Model mesh
    num_tpu_device = jax.device_count()
    tpu_devices = jax.local_devices()
    devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    mesh = Mesh(devices, axis_names=('dp', 'mp'))
    logging.info(f"The number of tpu device:{num_tpu_device}")
    logging.info(f"Tpu devices:{tpu_devices}")

    # Extracting model parameters from huggingface model
    parameter_convertor = ParameterConvertor(mesh=mesh, config=config, tokenizer=tokenizer)
    params = parameter_convertor(hf_model=hf_model)
    
    # Data Collator
    data_collator = Seq2SeqCollator(tokenizer, sequence_max_length=args.sequence_max_length)

    # Model
    model = FlaxLlaMaForCausalLM(config, _do_init=False)

    # Trainer
    trainer = Trainer(
        args=args, 
        model=model, 
        params=params, 
        tokenizer=tokenizer,
        dataset=encoded_instruction_dataset, 
        eval_datasets=encoded_evaluation_datasets,
        data_collator=data_collator
    )

    trainer.train()
    ```
  * Trainer functions
    * In ```utils/trainer.py```, Trainer class has 3 functions (train, evaluate, save_model).
       * ```evaluate(self, num_trainin_step)``` : Evaluate benchmarks which given from **evaluation_datasets** argument.
       * ```save_model(self, num_training_step: int, output_dir: str)```  : Convert trained parameters to huggingface(LlamaForCausalLM) model format and save checkpoint at output_dir
       * ```train(self, )```  : Train instruction dataset. While training, evalate and save model regularly.
       
<br>
<br>

## Command example
  * Command example on my server
    ```bash
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
    ```
  * Notice
    * You should huggingface style model for **model_path**.
    * The number of **instruction_datasets** and **dataset_sizes** should be same. 
    * The number of **evaluation_datasets** and **evaluation_shots** should be same. 
    * Just **constant or linear learning rate scheudler** can be used.
    * This repository does not support **gradient accumulation step**, i am working on it.


<br>
<br>

## Further points
  * These are tasks which i am studying nowdays and i am going to implement in this repository.
    1. Update evaluation metrics in ```eval/metrics.py``` for using metrics in https://github.com/EleutherAI/lm-evaluation-harness. 
    2. Add chat datasets for instruction-tuning.
    3. Gradient accumulation steps
    3. LoRA for Llama in Jax/Flax.
