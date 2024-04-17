# Router Visualization for Mixtral
This is a straightforward project that provides a visual representation of the expert choices made by the Mixtral router for text generation.

(Re-implementation of the Figure 8 from [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf) paper, see reference below)

In the visualization, each token in a text sample is colored with the first expert choice.

The code is kept simplistic for further customization.

### Requirements

~~~
# Python Library
transformers 4.39.3
~~~

### Example

- Prompt

~~~
# prompt (instruction + response)
<s> [INST] Act as Superman and give me a greeting [/INST] Up, up, and away! Greetings, citizen! It's a bird, it's a plane, no, it's Superman here to bring some super smiles to your day! How can I assist you today?</s>
~~~

- Output

The main.py results in n_layers (8 in Mixtral's case) html files that visualize the colorized text spans.

~~~
images/router_choice_layer_0.html
images/router_choice_layer_1.html
...
~~~

For the first 2 layers of [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1):

| Layer 0                              | Layer 1                              |
|--------------------------------------|--------------------------------------|
| ![layer_0.png](images%2Flayer_0.png) | ![layer_1.png](images%2Flayer_1.png) |

### Reference

Inspired by the Figure 8 from [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf) paper.

![reference.png](images%2Freference.png)
