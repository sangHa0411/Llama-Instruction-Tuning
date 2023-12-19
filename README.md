# Llama-Instruction-Tuning
* This repository was implemented using Jax/Flax.
* This fine-tuning code supports model parallel, so you can train Llama2-7B model on **TPU V3-8**.
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
    * WizardLM(https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k)
    * Open-Platypus(https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
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
    * I used ```\n``` for dataset column delimiter. And after each column, i put one space and then wrote content of that column.
    * I used ```\n\n``` for few-shot delimiter.
    * You can change this delimiter in ```utils/preprocessor.py```, ```eval/eval_preprocessor.py``` and ```utils/trainer.py```.
  * These are instruction examples.
    * Simple instruction example from Alpaca dataset.
        ```
        Instruction: What would be the best type of exercise for a person who has arthritis?
        Response: For someone with arthritis, the best type of exercise would be low-impact activities like yoga, swimming, or walking. These exercises provide the benefits of exercise without exacerbating the symptoms of arthritis.
        ```
    * Multiple-choice instruction example from MMLU dataset.
        ```
        Question: Which of these statements concerning the study of identity is not applicable to social constructivism?
        Choices:
        0. Identity defines actors, and tells us who they are, what their preferences and interests are, and how those preferences might inform their actions.
        1. Identity is of fundamental importance to the social constructivist perspective because it is the basest principle in defining behaviour; it is a naturally bestowed quality comprised of certain unalterable behavioural characteristics.
        2. The identities, interests and behaviour of political agents are socially constructed by shared ideas, collective meaning, and interpretations and assumptions about the world.
        3. Actors form their identity through interaction, which in turn defines the types of relationship formed between these actors.
        Answer: Identity is of fundamental importance to the social constructivist perspective because it is the basest principle in defining behaviour; it is a naturally bestowed quality comprised of certain unalterable behavioural characteristics.
        ```
    * Binary-choice instruction example from Winogrande dataset.
        ```
        Sentence: While packing for hunting Chris made sure to bring his bag and not his knife because the _ was nonessential.
        Option1: bag
        Option2: knife
        Answer: knife
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
    * 2-Shot example for MMLU dataset
      * Input
        ```
        Question: In what ways is a fusion nuclear device different from a fission device?
        Choices:
        0. A fusion weapon is a three-stage-bomb that uses an implosion device to create a fission reaction, which then detonates a fusion reaction. It is often referred to as a thermo-nuclear or hydrogen device and has unlimited destructive potential.
        1. A fusion reaction is created when the nuclei of two light elements are combined, the reaction resulting in heavier elements fragmenting into smaller parts. This fragmentation releases energy of limited destructive capacity, the extent of energy released dependent on the mass of material used.
        2. A fusion device is a two-stage process where a gun-type device is used to trigger a series of fission reactions with the cumulative result being the production of a high energy flash explosion with unlimited thermal energy.
        3. Fusion weapons have a highly specific destructive effect. The heavier element produced from a fissions reaction, and the difference in mass from the two lighter nuclei (which is translated into an energy explosion) can be precision calculated. Bombs even within the multi-megaton range have unlimited military utility, their destructive capability can be manufactured according to the target.
        Answer: A fusion weapon is a three-stage-bomb that uses an implosion device to create a fission reaction, which then detonates a fusion reaction. It is often referred to as a thermo-nuclear or hydrogen device and has unlimited destructive potential.

        Question: Which of these statements concerning the study of identity is not applicable to social constructivism?
        Choices:
        0. Identity defines actors, and tells us who they are, what their preferences and interests are, and how those preferences might inform their actions.
        1. Identity is of fundamental importance to the social constructivist perspective because it is the basest principle in defining behaviour; it is a naturally bestowed quality comprised of certain unalterable behavioural characteristics.
        2. The identities, interests and behaviour of political agents are socially constructed by shared ideas, collective meaning, and interpretations and assumptions about the world.
        3. Actors form their identity through interaction, which in turn defines the types of relationship formed between these actors.
        Answer: Identity is of fundamental importance to the social constructivist perspective because it is the basest principle in defining behaviour; it is a naturally bestowed quality comprised of certain unalterable behavioural characteristics.
              
        Question: Which of the following best describes the balance the Supreme Court has struck between the establishment clause and the free-exercise clause?
        Choices:
        0. Freedom of speech is protected except in certain situations, such as yelling "fire" in a crowded theater.
        1. Once a church has been recognized by the federal government, its tax-exempt status can never be revoked.
        2. Once Congress has created an administrative agency, that agency can be dissolved only by a constitutional amendment.
        3. State-sponsored prayer during school hours is prohibited, but voluntary prayer by student groups before school is allowed.
        Answer: 
        ```
      * Label
        ```
        State-sponsored prayer during school hours is prohibited, but voluntary prayer by student groups before school is allowed.
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
  * Command example
    ```bash
    python /home/sangha110495/project/Llama-Instruction-Tuning/src/run_finetune.py \
        --model_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --tokenizer_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --instruction_datasets="[arc,mmlu,gsm8k,alpaca,cot-collection,slimorca,openorca-mc10k]" \
        --dataset_sizes="[3000,3000,3000,2500,2500,2500,2500]" \
        --evaluation_datasets="[truthful_qa-generation,truthful_qa-multiple_choice,gsm8k,arc,mmlu]" \
        --evaluation_shots="[0,0,5,25,5]" \
        --random_seed=42 \
        --padding_side="left" \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=2 \
        --sequence_max_length=512 \
        --eval_sequence_max_length=1536 \
        --generation_max_length=128 \
        --gradient_checkpointing=True \
        --evaluation_strategy="epoch" \
        --logging_dir="/home/sangha110495/project/Llama-Instruction-Tuning/logging" \
        --output_dir="/mnt/disks-standard/persist/llm/llama-instruction-tuning/exps/checkpoints" \
        --cache_dir="/mnt/disks-standard/persist/huggingface" \
        --num_train_epochs=3 \
        --weight_decay=1e-2 \
        --warmup_ratio=0.1 \
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
