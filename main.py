import numpy as np
from transformers import AutoTokenizer, MixtralForCausalLM
from base import CustomGenerationMixin
from viz import format_html


class MixtralEncode(MixtralForCausalLM, CustomGenerationMixin):
    """Only encode prompt (containing both instruction and response)"""
    pass


if __name__ == '__main__':
    model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MixtralEncode.from_pretrained(model_path, device_map="auto")

    prompt = """<s> [INST] Act as Superman and give me a greeting [/INST] Up, up, and away! Greetings, citizen! It's a bird, it's a plane, no, it's Superman here to bring some super smiles to your day! How can I assist you today?</s>"""
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.compute_router_logits(inputs["input_ids"], max_new_tokens=5, output_router_logits=True)

    for i, layer_router_logits in enumerate(outputs.router_logits):
        tokens = []
        for j, token_router_logits in enumerate(layer_router_logits.cpu().tolist()):
            token = tokenizer.decode([inputs["input_ids"].tolist()[0][j]], skip_special_tokens=False)
            expert_choice = np.argmax(token_router_logits) + 1
            tokens.append([expert_choice, token])

        format_html(tokens, output_file=f"images/router_choice_layer_{i}.html")
