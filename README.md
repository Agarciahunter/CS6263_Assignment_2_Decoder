# CS6263_Assignment2_Decoder - Aiden Garcia-Rubio
## Instructions
### Envrionment Setup
First load the environment from the YML file with:

`conda env create -f environment.yml`

Then use this to activate it:

`conda activate unsloth_env`

You may need to install the transformers library manually since a forked version is being used:

`pip uninstall transformers` <br />
`pip install -e transformers-dola-main/`

After that use:

`pip install -r requirements.txt` or `pip install -r requirements1.txt` if the former doesn't work

# NOTE
The code was done using python 3.10 and code pytorch 12.1. If you want to use a differnet version of pytorch or cuda I recommond making a new a enviorment using this directory https://github.com/unslothai/unsloth. If you experience errors with installing the xformers I also suggest using that site to intsall unsloth.

If you have any other problems I suggest that you look at this issue thread as I was able to find most solutions to my errors from them: https://github.com/unslothai/unsloth/issues/221.

The python version can be very finicky when it comes to unsloth so I don’t recommend changing it.

### Execution
Once it's all set up run the code with:

`llamaTrain.py`

Afterwards you can use this code to find the infersnces for the various layer outputs on the model (before running I suggest checking this file to make sure that any unwanted hparams have been cut):

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
Consistency checking can be effective by viewing the different layers. It stands to reason that, if a token shows consistent high probability in all observed layers, it can be expected that the given token is not being hallucinated. The token surviving a lot of layers, means that the model always had high confidence in it.

**3)	Write another discussion explaining the how the layers effect on the different metrics on your trained model from assignment 1.c.**

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">Llama2</th>
    <th class="tg-9wq8">CodeBleu</th>
    <th class="tg-9wq8" colspan="2">Rouge-L</th>
    <th class="tg-9wq8">Rouge-L Average</th>
    <th class="tg-9wq8" colspan="2">BERTScore</th>
    <th class="tg-9wq8">BERTScore Average</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="3">Layer 8</td>
    <td class="tg-9wq8" rowspan="3">0.1369</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.2445</td>
    <td class="tg-9wq8" rowspan="3">0.2579</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.8180</td>
    <td class="tg-9wq8" rowspan="3">0.8387</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.2863</td>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.8601</td>
  </tr>
  <tr>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.2430</td>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.8381</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Layer 16</td>
    <td class="tg-9wq8" rowspan="3">0.1379</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.2454</td>
    <td class="tg-9wq8" rowspan="3">0.2595</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.8210</td>
    <td class="tg-9wq8" rowspan="3">0.8410</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.2888</td>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.8616</td>
  </tr>
  <tr>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.2443</td>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.8403</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Layer 24</td>
    <td class="tg-9wq8" rowspan="3">0.1209</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.2380</td>
    <td class="tg-9wq8" rowspan="3">0.2567</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.8162</td>
    <td class="tg-9wq8" rowspan="3">0.8374</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.2922</td>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.8593</td>
  </tr>
  <tr>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.2400</td>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.8367</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Layer 32</td>
    <td class="tg-9wq8" rowspan="3">0.3479</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.5384</td>
    <td class="tg-9wq8" rowspan="3">0.3453</td>
    <td class="tg-9wq8">Recall: </td>
    <td class="tg-9wq8">0.8806</td>
    <td class="tg-9wq8" rowspan="3">0.8476</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.2168</td>
    <td class="tg-9wq8">Precision: </td>
    <td class="tg-9wq8">0.8159</td>
  </tr>
  <tr>
    <td class="tg-9wq8">F1&nbsp;&nbsp;&nbsp;Score: </td>
    <td class="tg-9wq8">0.2807</td>
    <td class="tg-9wq8">F1 Score: </td>
    <td class="tg-9wq8">0.8464</td>
  </tr>
</tbody>
</table>

An improvement, albeit a small one, can be seen from Layer 8 to Layer 16, however it then proceeds to regress a bit in Layer 24. However, once it reaches layer 32, it performs the best across all metrics.  BERT Score experiences the least amount of change as it goes from layer to layer, which is likely due to it being more robust when it comes to words that are similar. Rouge and CodeBLEU followed the pattern of increase-decrease-increase that was mentioned. CodeBLEU was the most affected by the layers as the layer 32 score was more than double the layer 8 score.

## Special Thanks
DOLA: https://github.com/voidism/DoLa

DOLA Transformers: https://github.com/voidism/transformers-dola/

Unsloth AI: https://github.com/unslothai/unsloth
