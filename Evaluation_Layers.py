from peft import PeftModel, PeftConfig
from datasets import load_dataset
import random
from codebleu import calc_codebleu
from rouge import Rouge
from bert_score import score
import torch
import subprocess
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    )

def getOutput(tokenizer,model,testPrompt,hparam,size=0):
    input = tokenizer(testPrompt, return_tensors="pt").input_ids
    input = input.to('cuda')
    if hparam == "vanilla":
        # Vanilla decoding output generation
        outputs = model.generate(input, max_length = 450)
    elif hparam == "topK":
        # Top-K output sampling
        outputs = model.generate(input,
                                 max_length = 450,
                                 do_sample=True,
                                 top_k=size)
    elif hparam == "beam":
        # Beam Search output generation
        outputs = model.generate(input,
                                 max_length = 450,
                                 num_beams=size,
                                 early_stopping=True)
    elif hparam == "temp":
         # Temperature sampling output
         outputs = model.generate(input,
                                 max_length = 450,
                                 do_sample=True,
                                 top_k=0,
                                 temperature = size)
    # Generate output at different model layers
    elif hparam == "layer":
        logits_processor = LogitsProcessorList(
        [
        MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ]
        )
        # instantiate logits processors
        logits_warper = LogitsProcessorList(
        [
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7),
        ]
        )

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=50)])

        torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        outputs = model._dola_decoding(
            input,
            dola_layers=[size],
            max_length = 450,
            repetition_penalty=1.2,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            # output_scores=True,
            # return_dict_in_generate=True,
        )


    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


modelList = ["./Llama"]

outputType = [
    "vanilla",
    "topK",
    "beam",
    "temp",
    "layer"
]

topKsize = [
    2,
    4,
    6,
    8
]

beamsize = [
    2,
    3,
    4,
    5
]

tempSize = [
    .1,
    .25,
    .5,
    .75
]

layernum = [
    8,
    16,
    24,
]

datapath = "flytech/python-codes-25k"

# Load dataset
dataset = load_dataset("flytech/python-codes-25k", split='train')
numInputs = 50

randrows = []
for i in range(numInputs):
    randrows.append(random.randint(0,len(dataset)))

# Select random rows from the dataset
dataset = dataset.select(randrows)

for modelpath in modelList:
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(modelpath)
    model = PeftModel.from_pretrained(model, modelpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    for hparam in outputType:
        sizes = []
        if hparam == "vanilla":
            sizes = [1]
        elif hparam == "topK":
            sizes = topKsize.copy()
        elif hparam == "beam":
            sizes = beamsize.copy()
        elif hparam == "temp":
            sizes = tempSize.copy()
        elif hparam == "layer":
            sizes = layernum.copy()

        for size in sizes:
            referencelist = []
            predictionlist = []
            for i in range(numInputs):
                print("Getting output for: " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size) + "...Instruction:" + str(i+1))
                testPrompt = dataset[i]["instruction"]
                
                text = getOutput(tokenizer,model,testPrompt,hparam,size)
                
                referencelist.append(dataset[i]["output"])
                predictionlist.append(text)
            
            print("Results for " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            print('-' * 80)
            ##codebleu##
            codebleuResult = calc_codebleu(referencelist, predictionlist, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
            print("CodeBleu Scrore: " + str(codebleuResult["codebleu"]))
            ##rouge##
            rouge = Rouge()
            scores = rouge.get_scores(predictionlist, referencelist, avg=True)
            print("Rouge-L score: " + str(scores["rouge-l"]))
            ##BERTscore##
            P, R, F1 = score(predictionlist, referencelist, lang="en", verbose=True)
            recall = R.mean().item()
            precision = P.mean().item()
            f1_score = F1.mean().item()
            print("BERTScore:")
            print(P, R, F1)
            bert_score_str = f"r: {recall}, p: {precision}, f: {f1_score}"
            print("BertScore (P, R, F1)")
            print(bert_score_str)

            print('-' * 80)
            print("")

            print("For Human Evaluation on : " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            
            if hparam in ["vanilla", "layer"]:
                with open("evaluation_metrics.txt", "a") as file:
                    file.write(f"Model: {modelpath}, Output Type: {hparam}, Size: {size}\n")
                    file.write(f"CodeBleu Score: {codebleuResult['codebleu']}\n")
                    file.write(f"Rouge-L score: {scores['rouge-l']}\n")
                    file.write(f"BertScore: {bert_score_str}\n")
                    file.write("-" * 80 + "\n")
                    
            #only need 20 for human evaluation
            if numInputs > 20:
                numHumanEval = 20
            else:
                numHumanEval = numInputs

            for i in range(numHumanEval):
                print("Instruction " + str(i))
                
                print(dataset[i]["instruction"])
                print("***")
                print(str(modelpath) + " output:")
                print(predictionlist[i])
                print('-' * 80)
            # If hparam is "layer", execute additional code                
            
            if hparam == "vanilla":
                # Run dolaTest.py script with the current layer_value as argument
                subprocess.run(["python", "dolaTest.py"])
                print('Layer 32')
            if hparam == "layer":
                # Run dolaTest.py script with the current layer_value as argument
                subprocess.run(["python", "dolaTest.py", str(size)])
                print("NEXT LAYER")