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
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no value is passed, set layernum_value to 32
        layernum_value = 32
    elif len(sys.argv) == 2:
        # Retrieve layernum value from command-line arguments
        layernum_value = int(sys.argv[1])
    else:
        print("Usage: python dolaTest.py [<layernum_value>]")
        sys.exit(1)

    print("Running dolaTest.py with layernum:", layernum_value)
print("DOLA layernum_value")
print(layernum_value)
modelpath = "./Llama/"
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForCausalLM.from_pretrained(modelpath)
# model = model.to('cuda')

# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
model.config.pad_token_id = model.config.eos_token_id
model.generation_config.pad_token_id = model.config.eos_token_id

input_prompt = "Write hello world in Python"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda')
# instantiate logits processors
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
    input_ids,
    dola_layers=[layernum_value],
    repetition_penalty=1.2,
    logits_processor=logits_processor,
    logits_warper=logits_warper,
    stopping_criteria=stopping_criteria,
    output_scores=True,
    return_dict_in_generate=True,
)

text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)

input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
tokens= []
probabilities=[]

for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | log probability | probability
    # print(f"| {tokenizer.decode(tok):8s} | {np.exp(score.cpu().numpy()):.2%}")
    tokens.append(tokenizer.decode(tok))
    probabilities.append(np.exp(score.cpu().numpy()))

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(tokens, probabilities)
ax.set_xlabel('Tokens')
ax.set_ylabel('Probability')
ax.set_title('Token Probability at layer ' +str(layernum_value))
ax.set_xticklabels(tokens, rotation=90, fontsize=6)

filename = "token_probability_layer_" +str(layernum_value)+".png"
plt.savefig(filename, bbox_inches='tight')

print(text)