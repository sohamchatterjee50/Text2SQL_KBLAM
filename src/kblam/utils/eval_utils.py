from typing import Optional
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import torch
import transformers

from kblam.models.kblam_config import KBLaMConfig
from kblam.models.sql_coder_model import KblamLlamaForCausalLM
#from kblam.models.phi4_model import KBLaMPhi3ForCausalLM

instruction_prompts = """
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
"""

instruction_prompts_multi_entities = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

zero_shot_prompt = """
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
"""

zero_shot_prompt_multi_entities = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


# def _prune_for_llama(S: str) -> str:
#     S = S.replace("<|eot_id|>", "")
#     S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "")
#     S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
#     S = S.replace("<|end_of_text|>", "")
#     return S
def _prune_for_llama(S: str) -> str:
    S = S.replace("<|assistant|>", "")
    S = S.replace("<|end|>", "")
    S = S.replace("<|user|>", "")
    S = S.replace("<|end_of_text|>", "")
    return S

def _prune_for_phi3(S: str) -> str:
    S = S.replace("<|end|>", "")
    S = S.replace("<|assistant|>", "")
    S = S.replace("<|user|>", "")
    return S


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


# def _format_Q_llama(Q: str):
#     return (
#         "<|start_header_id|>user<|end_header_id|>"
#         + Q
#         + "<|eot_id|>"
#         + "<|start_header_id|>assistant<|end_header_id|>"
#     )

# def truncate_to_32_words(text):
#     words = text.split()
#     truncated = " ".join(words)
#     #print("Truncated length:",truncated)
#     return truncated

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        # table_metadata_string =  truncate_to_32_words(f.read())
        table_metadata_string = f.read()

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )
    # prompt = prompt.format(
    #     user_question=question
    # )
    return prompt


def _format_Q_llama(Q: str,db_id:str):
    metadata_file = os.path.join("/home/schatterjee1/Datasets/BirdSQL/train/train_databases/train_databases", db_id, db_id + '_schema.sql')
    if not os.path.exists(metadata_file):
        metadata_file = os.path.join("/home/schatterjee1/Datasets/BirdSQL/dev/dev_databases", db_id, db_id + '_schema.sql')

    prompt = generate_prompt(Q, "/home/schatterjee1/KBLAM/sqlcoder/prompt.md",metadata_file)
    #prompt = generate_prompt(Q, "/home/schatterjee1/KBLAM/sqlcoder/prompt.md")
    return "<|user|>" + prompt + "<|end|>" + "<|assistant|>" 

def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


model_question_format_mapping = {
    KblamLlamaForCausalLM: _format_Q_llama,
   #KBLaMPhi3ForCausalLM: _format_Q_phi3,
}
model_prune_format_mapping = {
   KblamLlamaForCausalLM: _prune_for_llama,
  # KBLaMPhi3ForCausalLM: _prune_for_phi3,
}
def _format_QA_llama(Q: str, A: str):
    return (
        "<|start_header_id|>user<|end_header_id|>"
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )

def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    # model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
   # model: KBLaMPhi3ForCausalLM,
     model: KblamLlamaForCausalLM,
    Q: str,
    A:str,
    db_id:str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
):  
    input_str = []
    # for m in model_question_format_mapping:
    #     if isinstance(model, m):
    #         input_str = model_question_format_mapping[m](Q,db_id)
    input_str = model_question_format_mapping[KblamLlamaForCausalLM](Q,db_id)
    print("Input STR:",input_str)       
    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to(
        "cuda"
    )
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=300,
            tokenizer=tokenizer,
            output_attentions=True,
            kb_config=kb_config,
        ).squeeze()
   
    
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)
    #print("Outputs after decode:",outputs)
    # for m in model_prune_format_mapping:
    #     if isinstance(model, m):
    #         pruned_output = model_prune_format_mapping[m](outputs)
    pruned_output = model_prune_format_mapping[KblamLlamaForCausalLM](outputs)
    #print("Pruned Output:",pruned_output)
    return pruned_output
