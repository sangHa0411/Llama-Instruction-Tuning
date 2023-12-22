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
    * Hellaswag(https://huggingface.co/datasets/Rowan/hellaswag)
    * GSM8k(https://huggingface.co/datasets/gsm8k)
    * Winogrande(https://huggingface.co/datasets/winogrande)
  * **Evaluation**
    * MMLU(https://huggingface.co/datasets/cais/mmlu)
      * Dataset Category : Multiple Choice
    * ARC(https://huggingface.co/datasets/ai2_arc)
      * Dataset Category : Multiple Choice
    * GSM8k(https://huggingface.co/datasets/gsm8k)
      * Dataset Category : Generation
    * Winogrande(https://huggingface.co/datasets/winogrande)
      * Dataset Category : Binary Choice
    * Hellaswag(https://huggingface.co/datasets/Rowan/hellaswag)
      * Dataset Category : Multiple Choice

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
        Question: Which of the following best describes the balance the Supreme Court has struck between the establishment clause and the free-exercise clause?
        Choices:
        A. Freedom of speech is protected except in certain situations, such as yelling "fire" in a crowded theater.
        B. Once a church has been recognized by the federal government, its tax-exempt status can never be revoked.
        C. Once Congress has created an administrative agency, that agency can be dissolved only by a constitutional amendment.
        D. State-sponsored prayer during school hours is prohibited, but voluntary prayer by student groups before school is allowed.
        Answer: Freedom of speech is protected except in certain situations, such as yelling "fire" in a crowded theater.
        ```
    * Generation instruction example from GSM8K dataset
        ```
        Question: Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?
        Answer: Five packs of milk cost $3 x 5 = $<<3*5=15>>15.
        Four apples cost $1.50 x 4 = $<<1.5*4=6>>6.
        The total cost of the chicken meal, milk, and apples is $12 + $15 + $6 = $<<12+15+6=33>>33.
        Thus, the boxes of pizza cost $50 - $33 = $<<50-33=17>>17.
        Therefore, Marie ordered $17/$8.50 = <<17/8.5=2>>2 boxes of pizza.
        #### 2
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
    * 3-Shot example fro Arc Dataset
      * Input with prompt
        ```
        Question: The best way to separate salt from water is with the use of
        Answer: heat.

        Question: Which of the following can be found on both Earth and the moon?
        Answer: Hills

        Question: Scientists studied largemouth bass populations in several different geographic locations. They recorded information about the size, number, and weight of fish in each population. Which of the following will be best suited for recording these data?
        Answer: a chart

        Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?
        Answer: 
        ```
      * Candidate example
        ```
        Planetary density will decrease.
        ```
      * The number of candidates : 4
      * Metric : `acc_norm`
      * Evaluation Shot size : 25
    * 1-Shot example for MMLU dataset
      * Input with prompt
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
        Answer: 
        ```
      * Candidate example
        ```
        Identity is of fundamental importance to the social constructivist perspective because it is the basest principle in defining behaviour; it is a naturally bestowed quality comprised of certain unalterable behavioural characteristics.
        ```
      * The number of candidates : 4
      * Metric : `acc_norm`
      * Evaulation Shot size : 5
    * 2-Shot example for Hellaswag dataset
      * Input with prompt
        ```
        Cutting the grass: A man walks outside plugs his lawn mower in and gets ready to mow. he starts mowing his lawn casually going over all the grass.

        Drum corps: People watch a parade where people walk on front a marching band holding a banner and waving the viewers. a man cross the street holding a small flag and extend his hand.

        Roof shingle removal: A man is sitting on a roof.
        ```
      * Candidate example
        ```
        he is using wrap to wrap a pair of skis.
        ```
      * The number of candidates : 4
      * Metric : `acc_norm`
      * Evaluation Shot size : 10
    * 0-Shot example for Truthful QA dataset
      * Input
        ```
        What happens to you if you eat watermelon seeds?
        ```
      * Candidate exmample
        ```
        Nothing happens
        ```
      * There are multi correct answers and incorrect answers
      * Metric : `mc2`
      * Evaulation Shot size : 0
    * 2-Shot example for Winogrande example
      * Input with prompt
        ```
        Michael just bought brand new wheels for his truck unlike Leslie because Leslie wheels were new and perfect.

        Christmas was a special holiday to Eric but not Adam since Adam was a Jew.

        Sarah was a much better surgeon than Maria so 
        ```
      * Candidate example
        ```
        Sarah always got the easier cases.
        ```
      * The number of candidates : 2
      * Metric : `acc_norm`
      * Evaluation Shot size : 5
    * 1-Shot example fro GSM8K
      * Input with prompt
        ```
        Question: Marie ordered one chicken meal that costs $12, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?
        Answer: Five packs of milk cost $3 x 5 = $<<3*5=15>>15.
        Four apples cost $1.50 x 4 = $<<1.5*4=6>>6.
        The total cost of the chicken meal, milk, and apples is $12 + $15 + $6 = $<<12+15+6=33>>33.
        Thus, the boxes of pizza cost $50 - $33 = $<<50-33=17>>17.
        Therefore, Marie ordered $17/$8.50 = <<17/8.5=2>>2 boxes of pizza.
        #### 2

        Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
        Answer: 
        ```
      * Answer
        ```
        Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
        She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
        #### 18
        ```
      * Metric : `exact_match` after **####**
      * Evaluation Shot size : 5


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
  * Important functions in Trainer
    * In ```utils/trainer.py```, Trainer class has 3 functions (train, evaluate, save_model).
       * ```evaluate``` : Evaluate benchmarks which given from **evaluation_datasets** argument.
       * ```save_model```  : Convert trained parameters to huggingface(LlamaForCausalLM) model format and save checkpoint at output_dir
       * ```train```  : Train instruction dataset. While training, evalate and save model regularly.
       * ```score_prediction``` : Postprocess evaluation results and score appropriate metric for each datasets.
       
<br>
<br>

## Command example
  * Command example
    ```bash
    python /home/sangha110495/project/Llama-Instruction-Tuning/src/run_finetune.py \
        --model_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --tokenizer_path="/mnt/disks-standard/persist/llama/llama-2-7b-hf" \
        --instruction_datasets="[slimorca]" \
        --dataset_sizes="[10000]" \
        --evaluation_datasets="[arc,hellaswag,mmlu,truthful_qa,winogrande_l,gsm8k]" \
        --evaluation_shots="[25,10,5,0,5,5]" \
        --random_seed=42 \
        --per_device_train_batch_size=8 \
        --per_device_eval_forward_batch_size=16 \
        --per_device_eval_generate_batch_size=2 \
        --sequence_max_length=512 \
        --eval_sequence_max_length=1536 \
        --generation_max_length=128 \
        --gradient_checkpointing=True \
        --evaluation_strategy="steps" \
        --eval_steps=3 \
        --logging_dir="/home/sangha110495/project/Llama-Instruction-Tuning/logging" \
        --output_dir="/mnt/disks-standard/persist/llm/llama-instruction-tuning/exps/checkpoints" \
        --cache_dir="/mnt/disks-standard/persist/huggingface" \
        --num_train_epochs=3 \
        --weight_decay=1e-2 \
        --warmup_ratio=0.1 \
        --learning_rate=3e-15 \
        --lr_scheduler_type="constant" \
        --logging_steps=100
    ```
  * Notice
    * You should huggingface style model for **model_path**.
    * The number of **instruction_datasets** and **dataset_sizes** should be same. 
    * The number of **evaluation_datasets** and **evaluation_shots** should be same. 
    * Just **constant or linear learning rate scheudler** can be used.
    * This repository does not support **gradient accumulation step**, i am working on it.
    * Evaluation
      * Datasets which category is multiple_choice like Arc and MMLU, then trainer use `per_device_eval_forward_batch_size` for evalution.
      * Datasets which category is generation like GSM8K, then trainer use `per_device_eval_generate_batch_size` for evaluation.



<br>
<br>

## Further points
  * These are tasks which i am studying nowdays and i am going to implement in this repository.
    1. Reproduce Llama2-7B's performance in Open LLM Leaderboard.
    2. Add chat datasets for instruction-tuning.
    3. Gradient accumulation steps
    3. LoRA for Llama in Jax/Flax.
