# CS6263_Assignment2_Decoder - Aiden Garcia-Rubio
## Instructions
### Envrionment Setup
First load the environment from the YML file with:

`conda env create -f environment.yml`

Then use this to activate it:

`conda activate unsloth_env`

You may need to install the transformers library manually since a forked version is being used:

`pip uninstall transformers`
`pip install -e transformers-dola-main/`

After that use:

`pip install -r requirements.txt` or `pip install -r requirements1.txt` if the former doesn't work

## NOTE
The code was done using python 3.10 and code pytorch 12.1. If you want to use a differnet version of pytorch or cuda I recommond making a new a enviorment using this directory https://github.com/unslothai/unsloth. If you experience errors with installing the xformers I also suggest using that site to intsall unsloth.

If you have any other problems I suggest this looking at this issue thread as I was able to generally able to find solutions from them https://github.com/unslothai/unsloth/issues/221.


### Execution

You can use this code to find the infersnces for the various layer outputs on the model (before running I suggest checking this file to make sure that any unwanted hparams have been cut):

`python Evaluation_Layers.py`

## Assignment Discussion

**1)	We would like for you to present and visualize the probabilities of each token in the vocabulary from early exit layers (premature vocabulary distribution layers) vs. mature layer (last layer – Layer 32).**

### Layer 8
![token_probability_layer_8](https://github.com/Agarciahunter/CS6263_Assignment_2_Decoder/blob/main/token_probability_layer_8.png)

### Layer 16
![token_probability_layer_16](https://github.com/Agarciahunter/CS6263_Assignment_2_Decoder/blob/main/token_probability_layer_16.png)

### Layer 24
![token_probability_layer_24](https://github.com/Agarciahunter/CS6263_Assignment_2_Decoder/blob/main/token_probability_layer_24.png)

### Layer 32
![token_probability_layer_32](https://github.com/Agarciahunter/CS6263_Assignment_2_Decoder/blob/main/token_probability_layer_32.png)

**2)	If you recall the paper we reviews on consistency checking used several models, do you think we can use consistency check method between these layers for factuality analysis? Present your approach and results including discussion.**
Yes I believe consistency checking would be somewhat effective by looking at the different layers.  Intuitively, if a token had a high probability in all observered layers, one could expect that that token is solid and not being hallucinated.  The token would have "survived" so many layers, meaning that the model always had high confidence in it.

**3)	Write another discussion explaining the how the layers effect on the different metrics on your trained model from assignment 1.c.**




Interestingly the model seems to improve slightly from Layer 8 to Layer 16, but then regess a bit in Layer 24.  At the top layer, 32,  it performs the best across all metrics.  BERTScore shows the least amount ofchange from layer to layer, probably because it is a more robust to words that are similar.  Rouge and CodeBLEU followed a pattern that I mentioned above more decidedly.  At the final layer, CodeBLEU score was more than double the score of any of the other layers.

## Special Thanks
DOLA: https://github.com/voidism/DoLa

DOLA Transformers: https://github.com/voidism/transformers-dola/
# CS6263_Assignment_2_Decoder
# CS6263_Assignment_2_Decoder
# CS6263_Assignment_2_Decoder
# CS6263_Assignment_2_Decoder
