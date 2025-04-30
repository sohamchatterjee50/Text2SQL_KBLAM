import argparse
import json
import logging
import os
import pathlib
import re
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Optional
from transformers import LlamaTokenizer
import numpy as np
import torch
import transformers
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from kblam.kb_encoder import KBEncoder
from kblam.models.kblam_config import KBLaMConfig
from kblam.models.sql_coder_model_same_embed import KblamLlamaForCausalLM
#from kblam.models.phi4_model import KBLaMPhi3ForCausalLM
from kblam.utils.data_utils import (
    augment_row,
    generate_multi_entity_qa,
    get_i_dont_know_ans,
)
from kblam.utils.train_utils import (
    context_set_size_scheduler,
    get_kb_embd,
    setup_scheduler_and_optimizer,
)
from torch.nn.utils.rnn import pad_sequence

LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

# setup logging
# Create a custom theme for Rich
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)

# Create a Rich console with the custom theme
console = Console(theme=custom_theme)

# Configure the root logger to WARNING
logging.basicConfig(
    level=logging.WARNING,  # Set the root logger to WARNING
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--train_dataset",type=str,default="train_dev_bird_DB_Schemas_100",choices=["train_dev_bird_DB_Schemas_100","train_dev_bird_same_key_val_100","train_dev_bird_ST_100","train_dev_bird_naive_100", "dev_bird_naive"])
#parser.add_argument("--N", type=int, default=120000, help="Size of training set, select the first N samples for training")
parser.add_argument("--B", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, help="Train a separate query head")
parser.add_argument("--use_oai_embd", action="store_true", help="Use OpenAI embedding")
parser.add_argument("--use_cached_embd", action="store_true", help="Choose to use pre-computed KV embeddings")
#parser.add_argument("--total_steps", type=int, default=20000, help="Total steps")
parser.add_argument("--encoder_spec", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--key_embd_src", type=str, default="key", choices=["key", "answer", "questions", None], help="Source of key embedding")
parser.add_argument("--use_data_aug", action="store_true", help="Randomly pick templates for the question")
parser.add_argument("--use_lr_decay", action="store_true")
parser.add_argument("--dataset_dir", type=str, default="/home/schatterjee1/KBLAM/KBLaM/datasets")
parser.add_argument("--model_dir_to_resume", type=str, default=None, help="Checkpoint directory to resume training")
parser.add_argument("--hf_model_spec", type=str, default="defog/sqlcoder-7b-2", choices=["defog/sqlcoder-7b-2","microsoft/phi-4","microsoft/Phi-4-mini-instruct","meta-llama/Meta-Llama-3-8B", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-3.2-1B-Instruct"])
parser.add_argument("--hf_token", type=str,default="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",help="Huggingface token")
parser.add_argument("--model_save_dir", type=str, default="/scratch-shared/chatty/sql-coder-SameEmbedSpace", help="Place to save the checkpoints")
parser.add_argument("--kb_size", type=int, default=None, help="The size of the KB set size")
parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None, help="The size of the KB set size. Set a dynamic range for the kbsize specify min and max")
parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True, help="Duplicate true entity's KB token")
parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False, help="Scale the raw attention score")
parser.add_argument("--outlier_num", type=int, default=1, help="Introduce questions without correct KB entites")
parser.add_argument("--multi_entities", type=int, default=None, help="Introduce questions involving multiple entities")
parser.add_argument("--use_extended_qa", action="store_true", help="Introduce QA with extended open-ended parts")
parser.add_argument("--kb_token_layer_frequency", type=int, default=3, help="Introduce QA with extended open-ended parts")
parser.add_argument("--gradient_accm_step", type=int, default=20, help="Introduce QA with extended open-ended parts")
parser.add_argument("--verbose", action="store_true", help="Set logging to debug")
parser.add_argument("--log_to_file", action="store_true", help="Log to file as well as stdout")
parser.add_argument("--llm_type",type=str,default="llama3",choices=["llama3", "phi3"])
parser.add_argument('--outlier_ratio', type=float, default=-1, help="Outlier ratio")
# fmt: on
def truncate_to_32_words(text):
    words = text.split()
    truncated = " ".join(words[:32])
    print("Truncated length:",truncated)
    return truncated

def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string =  truncate_to_32_words(f.read())

    prompt = prompt.format(
        user_question=question, table_metadata_string=table_metadata_string
    )

    #prompt = prompt.format(user_question=question)
    return prompt


def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
) -> Progress:
    """
    Create a custom progress bar using Rich, optionally including loss reporting.

    :param description: Description of the task
    :param total: Total number of steps
    :param console: Rich Console object (if None, a new one will be created)
    :param color: Color of the progress bar
    :param show_time: Whether to show the time remaining
    :param show_spinner: Whether to show a spinner
    :param spinner_style: Style of the spinner (e.g., "dots", "dots12", "line", "arrow")
    :param show_loss: Whether to show loss information
    :return: A Rich Progress object and task ID
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend(
        [
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
        ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=False)
    return progress

def _format_QA_llama(Q: str, A: str,db_id:str):
    metadata_file = os.path.join("/home/schatterjee1/Datasets/BirdSQL/train/train_databases/train_databases", db_id, db_id + '_schema.sql')
    if not os.path.exists(metadata_file):
        metadata_file = os.path.join("/home/schatterjee1/Datasets/BirdSQL/dev/dev_databases", db_id, db_id + '_schema.sql')

    prompt = generate_prompt(Q, "/home/schatterjee1/KBLAM/sqlcoder/prompt.md",metadata_file)
    
    #prompt = generate_prompt(Q, "/home/schatterjee1/KBLAM/sqlcoder/prompt.md")
    print("Training Prompt:",prompt)
    return "<|user|>" + prompt + "<|end|>" + "<|assistant|>" + A + "<|end|>"


# def _format_QA_llama(Q: str, A: str):
#     return (
#         "<|start_header_id|>user<|end_header_id|>"
#         + Q
#         + "<|eot_id|>"
#         + "<|start_header_id|>assistant<|end_header_id|>"
#         + A
#         + "<|eot_id|>"
#     )


# def _format_QA_phi3(Q: str, A: str):
#     return (
#         "<|im_start|>user<|im_sep|>"
#         + Q
#         + "<|im_end|>"
#         + "<|im_start|>assistant<|im_sep|>"
#         + A
#         + "<|im_end|>"
#     )

def _format_QA_phi3(Q: str, A: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"


def _create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    # Not sure this is correct. This method simply masks the <|start_header_id|>user<|end_header_id|> then leaves the rest in the labels
    # Possibly what they want is to mask out the query. To do that swap the index from the tokenizer below from 1 to 2
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|assistant|>")["input_ids"][1]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 1)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels








def _create_labels_for_phi3(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    # We just want to mask out the starting token.
    # The tokenized values are left padded so we want to know where our Q/A pairs start
    # Not 100% this is correct
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|assistant|>")["input_ids"][0]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 1)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels


# import torch
# from typing import List
# from transformers import AutoTokenizer

# def format_QA_and_create_labels(Q: str, A: str, tokenizer):
#     # Step 1: Format the input
#     formatted_text = (
#         "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"
#     )
#     print("Formatted Text:")
#     print(formatted_text)
    
#     # Step 2: Tokenize the formatted text
#     tokenized = tokenizer(formatted_text, return_tensors="pt", padding=True)
#     input_ids = tokenized["input_ids"]
#     print("\nTokenized input_ids:", input_ids.tolist())
    
#     # Step 3: Find the start of the assistant's response
#     answer_indices = torch.argmax(
#         (input_ids == tokenizer("<|assistant|>")["input_ids"][0]).long(), -1
#     )
#     print("\nAssistant starts at index:", answer_indices.item())
    
#     # Step 4: Create a mask to keep only the assistant’s response
#     answer_mask = torch.ones_like(input_ids)
#     for b in range(len(input_ids)):
#         answer_mask[b, : (answer_indices[b].item() + 1)] = 0  # Mask everything before assistant
    
#     labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
#     print("\nFinal labels tensor:", labels.tolist())
    
#     return input_ids, labels

# # Example usage
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
# Q = "What is the capital of France?"
# A = "The capital of France is Paris."

# input_ids, labels = format_QA_and_create_labels(Q, A, tokenizer)


def get_batch(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    start_index: int = 0,
    B: int = 20,
    random_sample=False,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
):
    """
    dataset: List of dictionary, denoting the KB, used to extract QA pairs
    model: The LLM, used to provide the embedding
    kb_embedding: KB embedding (differentiable)
    B: Batchsize
    include_outlier : Create a batch of question without answer in the KB.
    multi_entities : Create a batch of question that involves more than one entities.
    """
    labels = []
    # if multi_entities is not None:
    #     assert not include_outlier

    # if random_sample:
    #     if multi_entities is not None:
    #         batch_indices = np.random.choice(
    #             len(dataset), (B, multi_entities), replace=False
    #         )
    #     else:
    #         batch_indices = np.random.choice(len(dataset), B, replace=False)
    # else:

    #     # batch_indices = np.arange(B)
    batch_indices = np.arange(start_index, min(9586,start_index + B))

    def get_question_and_answer(idx: int) -> tuple[str, str]:
        # if use_extended_qa:
        #     Q, A = dataset[idx]["extended_Q"], dataset[idx]["extended_A"]

        # elif multi_entities is not None:
        #     Q, A = generate_multi_entity_qa(
        #         [dataset[i]["name"] for i in idx],
        #         [dataset[i]["description_type"] for i in idx],
        #         [dataset[i]["description"] for i in idx],
        #     )
        # else:
        #     Q = augment_row(dataset[idx]) if use_data_aug else dataset[idx]["Q"]
        #     A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]
        Q = dataset[idx]["Q"]
        A = dataset[idx]["A"]
        db_id = dataset[idx]["db_id"]
        return Q, A, db_id

    with torch.autograd.no_grad():
        input_strs = []
        for idx in batch_indices:
            Q, A, db_id = get_question_and_answer(idx)
            input_strs.append(qa_format_func(Q, A, db_id))
        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(
            device
        )
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )

        labels = label_func(input_ids, input_strs, tokenizer)
    # if include_outlier:
    #     # Generate a new set of indices, such that the KB does not contain the entity where the question comes from
    #     batch_indices = np.random.choice(len(dataset), B, replace=False)

    print("Indices:",input_ids)
    print("Labels:",labels)
    print("ATtention masks:",attention_masks)
    return input_ids, attention_masks, labels, batch_indices


def get_prefix_str(args):
    use_data_aug = args.use_data_aug
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size

    if dynamic_kb_size is not None:
        kb_size = "dynamic"  # Random size

    duplicate_true_kb = args.duplicate_true_kb
    length_invariance = args.length_invariance
    outlier_ratio = args.outlier_ratio
    use_outlier = outlier_ratio != -1
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    lr = args.lr

    prefix_string = f"stage1_lr_{lr}"
    if kb_token_layer_frequency is not None:
        prefix_string += f"KBTokenLayerFreq{kb_token_layer_frequency}"
    if use_extended_qa:
        prefix_string += "UseExtendedQA"
    if multi_entities is not None:
        prefix_string += f"MultiEntities{multi_entities}"
    if use_outlier:
        prefix_string += f"UseOutlier{outlier_ratio}"
    if length_invariance:
        prefix_string += "LengthInvariant"
    if not duplicate_true_kb:
        prefix_string += "NoDuplicate"
    if kb_size is not None:
        prefix_string += f"KBSize{kb_size}"
    if sep_query_head:
        prefix_string += "SepQueryHead"
    if use_data_aug:
        prefix_string += "UseDataAug"
    return prefix_string


def _load_cached_embeddigns(
    encoder_model_spec: str, dataset_dir: str, dataset_name: str, key_embd_src: str
):
    if encoder_model_spec == "OAI":
        encoder_model_spec_str = "oai"
    else:
        encoder_model_spec_str = encoder_model_spec
    key_embds = np.load(
        os.path.join(
            dataset_dir,
            f"{dataset_name}_{encoder_model_spec_str}_embd_{key_embd_src}.npy",
        )
    ).astype("float32")
    if key_embd_src == "answer":
        # If we are using the answer string as the key, we also use it as the value string
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"{dataset_name}_{encoder_model_spec_str}_embd_answer.npy",
            )
        ).astype("float32")
    else:
        value_embds = np.load(
            os.path.join(
                dataset_dir,
                f"{dataset_name}_{encoder_model_spec_str}_embd_value.npy",
            )
        ).astype("float32")
    return key_embds, value_embds


def get_step_config(
    current_accum_step: int,
    total_accum_step: int,
    use_data_aug: bool,
    outlier_num: int,
    multi_entities: int | None,
    use_extended_qa: bool,
):
    """
    Our instruction tuning dataset is composed of different types of instructions.
    Strategies:
    Outlier QA takes the last `outlier_num` accum steps;
    Multiple entites QA (if included) takes 1/3 of the rest accum_steps;
    Extended QA (if included) takes 1/3 of the rest accum_steps;
    Standard QA takes the rest.
    """
    config = {}
    config["use_data_aug"] = use_data_aug
    config["include_outlier"] = False
    config["multi_entities"] = None
    config["use_extended_qa"] = False
    # include_outlier = current_accum_step >= total_accum_step - 1 - outlier_num
    # Decide to include outlier and has reached the time
    # if include_outlier:
    #     config["include_outlier"] = True
    #     return config
    # if current_accum_step % 3 == 0:
    #     # multi_entities could be None,
    #     # in which case we just use standard QA
    #     config["multi_entities"] = multi_entities
    #     return config
    # if current_accum_step % 3 == 1:
    #     config["use_extended_qa"] = use_extended_qa
    #     return config
    return config


def _get_parameter_count(encoder):
    param_count = 0.0
    for p in encoder.parameters():
        if p.requires_grad:
            param_count += p.numel()
    return param_count


def _get_phi3_query_head_parameters(
    #model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
  # model: KBLaMPhi3ForCausalLM,
    model:KblamLlamaForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            # For phi3
            if "qkv_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight[: model.config.hidden_size, :])  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


def _get_llama3_query_head_parameters(
   # model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
   # model: KBLaMPhi3ForCausalLM,
     model:KblamLlamaForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:  # TODO: this is different for each model type
            # For llama3
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


class KBRetriever:
    def __init__(
        self,
        encoder: KBEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray],
        value_embds: Optional[np.ndarray],
    ):
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices, batch_size, step, kb_size):
        if self._use_cached_embd():
            train_set_key, train_set_val = get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            print("Here embeddings are retrieved")
            # train_set_key, train_set_val = get_kb_embd(
            #     self.encoder, batch_indices, kb_dict=self.dataset
            # )
            train_set_key = []
            for index in batch_indices:
                print("Index:",index)
                chunk_no = int(np.floor(index / 8))  # Calculate the chunk number
                print("Chunk no:",chunk_no)
                chunk_path = f"/scratch-shared/chatty/chunked_embeddings/embd_chunk_{chunk_no}.npy"
                embd_chunk = np.load(chunk_path)
                embd_index_in_chunk = index % 8
                print("embd_index_inside_chunk:",embd_index_in_chunk)
                selected_embedding = embd_chunk[embd_index_in_chunk]
                print("Embeddings shape:",selected_embedding.shape)
                train_set_key.append(selected_embedding)


        # if len(train_set_key.shape) == 2:
        #     # Add comment on why we need this line
        #     train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
        #     train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        # context_set_size = context_set_size_scheduler(step, kb_size)
        # context_set_index = np.random.choice(
        #     len(self.dataset), context_set_size, replace=False
        # )  # type: ignore
        # if self._use_cached_embd():
        #     context_set_key, context_set_val = get_kb_embd(
        #         self.encoder,
        #         context_set_index,
        #         precomputed_embd=(self.key_embds, self.value_embds),
        #     )
        # else:
        #     context_set_key, context_set_val = get_kb_embd(
        #         self.encoder, context_set_index, kb_dict=self.dataset
        #     )
        # context_set_key = context_set_key.unsqueeze(0).expand(
        #     batch_size, *context_set_key.shape
        # )
        # context_set_val = context_set_val.unsqueeze(0).expand(
        #     batch_size, *context_set_val.shape
        # )
        # # context_set_val = torch.randn_like(context_set_val)
        # # Idea: Try torch.randn here context_set_tokens??
        # true_kb_copy = 1
        # kb_embedding = (
        #     torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
        #     torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        # )
        
        # train_set_key_combined = np.array(train_set_key) 
         
        

        # train_set_key_combined = np.empty(len(train_set_key), dtype=object)
    
        # # Fill the object array with the individual sequences (they can have different shapes)
        # for i, seq in enumerate(train_set_key):
        #     train_set_key_combined[i] = np.array(seq)
        # train_set_key_combined_tensor = torch.tensor(train_set_key_combined) 

        train_set_key_combined_tensor = pad_sequence([torch.tensor(seq) for seq in train_set_key], batch_first=True)
        kb_embedding = (train_set_key_combined_tensor,train_set_key_combined_tensor)
        return kb_embedding


class Trainer:
    def __init__(
        self,
        #llm_model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
      #  llm_model: KBLaMPhi3ForCausalLM,
        llm_model:KblamLlamaForCausalLM,
        kbretriever: KBRetriever,
        tokenizer: transformers.PreTrainedTokenizer,
        kb_token_layer_frequency: int,
        num_steps: int,
        lr: float,
        device: torch.device,
        use_lr_decay: bool,
        kb_size: int | List[int],
        llm_savename: str,
        output_dir: str,
        sep_query_head: bool = False,
    ):
        self.logger = logging.getLogger("training")
        self.tokenizer = tokenizer
        self.sep_query_head = sep_query_head
        self.kb_token_layer_frequency = kb_token_layer_frequency
        self.num_steps = num_steps
        self.lr = lr
        self.model = llm_model
        self.device = device
        self.kbretriever = kbretriever
        self.kb_size = kb_size
        self.use_lr_decay = use_lr_decay
        self.llm_savename = llm_savename
        self.output_path = pathlib.Path(output_dir)
        print("Total no of steps:",self.num_steps)
        if isinstance(llm_model, KblamLlamaForCausalLM):  # llama
            self._get_batch = partial(
                get_batch, _format_QA_llama, _create_labels_for_llama
            )
        elif isinstance(llm_model, KBLaMPhi3ForCausalLM):  # Phi3
            self._get_batch = partial(
                get_batch, _format_QA_phi3, _create_labels_for_phi3
            )
            self._get_params = _get_phi3_query_head_parameters
      
            self._get_params = _get_llama3_query_head_parameters
       
        
        else:
            raise ValueError(f"{llm_model} not recognised")

        self.scheduler, self.optim = self.setup_scheduler_and_optim()

    def setup_scheduler_and_optim(self):
        if self.sep_query_head:
            self.logger.info("Query head being fine tuned!")
            llm_q_params = self._get_params(
                self.model, self.sep_query_head, self.kb_token_layer_frequency
            )
            scheduler, optim = setup_scheduler_and_optimizer(
                chain(self.kbretriever.encoder.parameters(), llm_q_params),
                self.lr,
                self.num_steps,
            )
            self.logger.info("Optimizer recreated")
        else:
            scheduler, optim = setup_scheduler_and_optimizer(
                self.kbretriever.encoder.parameters(), self.lr, self.num_steps
            )
            self.logger.info("Optimizer recreated")
        return scheduler, optim

    def train(
        self,
        training_set: List[Dict],
        batch_size,
        grad_accum_steps: int,
        outlier_num: int,
        use_data_aug: bool = False,
        multi_entities: bool = False,
        use_extended_qa: bool = False,
        save_period: int = 50,
        resumed_step: int = 0,
        kb_config: KBLaMConfig = None,
    ):
        train_losses = []
        start_step = 0

        loss_fct = CrossEntropyLoss(reduction="none")
        prev_start_index = 0
        print("No of steps:",self.num_steps)
        with create_custom_progress_bar(console=console) as pbar:
            task = pbar.add_task("Training", total=self.num_steps, loss=100)
            for step in range(start_step, self.num_steps, 1):
                self.optim.zero_grad()
                losses = []

                # Accumulate gradients
                for a_step in range(grad_accum_steps):
                    step_config = get_step_config(
                        a_step,
                        grad_accum_steps,
                        use_data_aug,
                        outlier_num,
                        multi_entities,
                        use_extended_qa,
                    )
                    print("Batch start index:",prev_start_index)
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(
                        training_set,
                        self.tokenizer,
                        self.device,
                        start_index = prev_start_index,
                        B=batch_size,
                        random_sample=False,
                        **step_config,
                    )
                    print("Batch list of indices:",batch_indices)
                    print("Batch size:",len(batch_indices))
                    prev_start_index += len(batch_indices)
                    if prev_start_index == 9586:
                        prev_start_index = 0
                    kb_embedding = self.kbretriever.get_key_embeddings(
                        batch_indices, len(batch_indices), step, self.kb_size
                    )
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        kb_kvs=kb_embedding,
                        output_attentions=True,
                        kb_config=kb_config,
                    )
                    logits = out["logits"]

                    # display ground truth and model prediction to quickly check model
                    if step:
                        batch_index = 0  # Which example in the batch to select
                        max_logits = logits.argmax(axis=2)
                        decoded_pred = self.tokenizer.decode(
                            max_logits[batch_index, :-1],skip_special_tokens=True
                        )
                        sel_labels = labels[batch_index, :]
                        sel_labels = sel_labels[
                            sel_labels >= 0
                        ]  # Remove padding token -100
                        decoded_gt = self.tokenizer.decode(sel_labels,skip_special_tokens=True)
                        self.logger.info(f"Decoded GT:{decoded_gt}")
                        self.logger.info(f"Decoded Pred:{decoded_pred}")

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = (
                        (shift_labels > 0)
                        .sum(-1, keepdim=True)
                        .expand(-1, shift_labels.shape[1])
                        .contiguous()
                    )
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    weights = weights.view(-1)

                    

                    # Print GT vs Preds
                    print("Labels:",shift_labels.max())  # Check the maximum token ID
                    print("Logits:",shift_logits.max())  # Check the maximum token ID
                    valid_shift_labels = shift_labels[shift_labels < self.tokenizer.vocab_size]
                    valid_shift_labels = valid_shift_labels.to(dtype=torch.int64)  # Ensure correct dtype
                    # valid_shift_labels_list = valid_shift_labels.cpu().numpy().astype(int).tolist()
                    valid_shift_labels = valid_shift_labels[valid_shift_labels != -100]
                    valid_shift_labels_list = valid_shift_labels.cpu().numpy().astype(int).tolist()
                    # print("valid_shift_labels shape:", valid_shift_labels.shape)
                    # print("valid_shift_labels dtype:", valid_shift_labels.dtype)
                    # print("valid_shift_labels min:", valid_shift_labels.min().item())
                    # Ensure valid_shift_labels is a tensor on CPU and explicitly int64
                    valid_shift_labels_list = np.where(valid_shift_labels_list != -100, valid_shift_labels_list, self.tokenizer.pad_token_id)

                    # Convert safely without dtype issues
                    valid_shift_labels_list = list(map(int, valid_shift_labels.tolist()))

                    # Decode
                    shift_labels_text = self.tokenizer.decode(valid_shift_labels_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    print(self.tokenizer.vocab_size)  # Check the vocabulary size of the tokenizer

                    print("******************************************************")
                    print("Ground-Truth Text:", shift_labels_text)

                    # Get predicted token IDs from shift_logits (with the highest probability for each token)
                    predicted_token_ids = shift_logits.argmax(dim=-1)

                    # Decode the predicted token IDs into text
                    predicted_text = self.tokenizer.decode(predicted_token_ids.tolist(), skip_special_tokens=True)
                    print("Predicted Text:", predicted_text)
                    print("#################################################################")

                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = (
                        loss_fct(shift_logits, shift_labels) * weights.max() / weights
                    ).mean()  # Make sure each sample is equally weighted

                    loss.backward()
                    losses.append(loss.item())

                self.optim.step()
                if self.use_lr_decay:
                    self.scheduler.step()

                self.logger.info(f"step: {step}, loss: {np.mean(losses)}")
                print("self.output_path:",self.output_path)
                print("self.llm_savename",self.llm_savename)
                train_losses.append(np.mean(losses))
                pbar.update(task, advance=1, loss=np.mean(losses))

                if (step % save_period) == 0 and (step != start_step):
                    # Save the full model with query head, and the encoder (adapter) and the kb_config
                    model_ckpt_name = (
                        self.output_path / f"{self.llm_savename}_step_{step}"
                    )
                    self.model.save_pretrained(model_ckpt_name)

                    encoder_ckpt_name = model_ckpt_name / "encoder.pt"
                    torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)

                    kb_config.save_pretrained(model_ckpt_name / "kb_config.json")
                if step == self.num_steps-1:
                    # Save the full model with query head, and the encoder (adapter) and the kb_config
                    model_ckpt_name = (
                        self.output_path / f"{self.llm_savename}_step_{step}"
                    )
                    self.model.save_pretrained(model_ckpt_name)

                    encoder_ckpt_name = model_ckpt_name / "encoder.pt"
                    torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)

                    kb_config.save_pretrained(model_ckpt_name / "kb_config.json")

                    


def main():
    logger = logging.getLogger("training")

    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    dataset_name = args.train_dataset
    seed = args.seed
    #N = args.N
    B = args.B

    #total_steps = args.total_steps
    encoder_spec = args.encoder_spec
    key_embd_src = args.key_embd_src
    use_data_aug = args.use_data_aug
    use_lr_decay = args.use_lr_decay
    use_cached_embd = args.use_cached_embd
    dataset_dir = args.dataset_dir
    model_dir_to_resume = args.model_dir_to_resume
    model_save_dir = args.model_save_dir
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size

    if kb_size is not None and dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")

    kb_size = kb_size if kb_size is not None else dynamic_kb_size

    gradient_accm_step = args.gradient_accm_step

    length_invariance = args.length_invariance
    outlier_num = args.outlier_num
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    llm_type = args.llm_type
    hf_model_spec = args.hf_model_spec
    hf_token = args.hf_token

    torch.manual_seed(seed)
    np.random.seed(seed)

    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if args.log_to_file:
        formatter = logging.Formatter(LOGFORMAT)
        f_handler = logging.FileHandler(model_save_dir / "log.txt")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.info(f"Running on {device}")

    logger.info("🚨 Started training 🚨")
    logger.info(f"💽 Saving to  {model_save_dir}💽")
    if sep_query_head:
        os.environ["SEP_QUERY_HEAD"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    if length_invariance:
        os.environ["LENGTH_INVARIANCE"] = "TRUE"
        logger.info("Having seperate query head for KB!")

    os.environ["SCALE_FACTOR"] = ""

    if use_cached_embd:
        # We load the pre-computed version stored on the disk rather
        # than computing them on the fly to make things faster
        logger.info(f"Using pre-computed {encoder_spec} embedding")
        # key_embds, value_embds = _load_cached_embeddigns(
        #     encoder_spec, dataset_dir, dataset_name, key_embd_src
        # )
        key_embds, value_embds = None, None

    prefix_string = get_prefix_str(args)
    logger.info(f"Experiment prefix {get_prefix_str(args)}")

    if use_extended_qa:
        dataset = json.load(
            open(os.path.join(dataset_dir, f"{dataset_name}_augmented.json"))
        )
    else:
        dataset = json.load(open(os.path.join(dataset_dir, f"{dataset_name}.json")))

    #training_set = dataset[:N]
    training_set = dataset


    # Set up the LLM
    llm_model_spec = model_dir_to_resume if model_dir_to_resume else hf_model_spec

    resumed_step = (
        0 if not model_dir_to_resume else int(model_dir_to_resume.split("_")[-1])
    )

    if llm_model_spec is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    if hf_token is None and args.llm_type == "llama3":
        raise ValueError(
            "Please supply HuggingFace token(hf_token) when loading model Llama weights from HuggingFace"
        )

    #Tokenizer comes from the base model
    # tokenizer = AutoTokenizer.from_pretrained(
    #     hf_model_spec,
    #     trust_remote_code=True,
    #     token=hf_token if hf_token is args.llm_type == "llama3" else None,
    # )
    # tokenizer.pad_token = tokenizer.eos_token

    tokenizer = AutoTokenizer.from_pretrained("C:\Users\SCEE\Desktop\Code_CleanUp\KBLaM\sql-coder-special-Tokens")
    tokenizer.pad_token = tokenizer.eos_token
    
    # num_added_toks = tokenizer.add_tokens(['<|begin_db_id|>','<|end_db_id|>','<|begin_table_id|>','<|end_table_id|>','<|begin_col_id|>','<|end_col_id|>','<|begin_select|>','<|end_select|>','<|begin_from|>','<|end_from|>'], special_tokens=True) 
    # print("No of special tokens added:",num_added_toks)
    #print("Test check for special tokens:",tokenizer.convert_ids_to_tokens(tokenizer( '<|user|>[QUESTION]Start of Question[/QUESTION][SQL]<|end|><|assistant|>Start of SQL<|end|>', return_tensors="pt").to('cpu')['input_ids'].squeeze(0).tolist()))
    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token,
        )
        model.resize_token_embeddings(len(tokenizer))
    elif args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        ValueError(f"LLM type {args.llm_type} not recognised")

    logger.info(model.config)  # type: ignore

    model.train()  # type: ignore
    # freeze model
    for _, param in model.named_parameters():  # type: ignore
        param.requires_grad = True

    # Set up the encoder
    encoder = KBEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size  # type: ignore
        * (model.config.num_hidden_layers // kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=device,
    )

    if model_dir_to_resume:
        encoder.load_state_dict(
            torch.load(os.path.join(model_dir_to_resume, "encoder.pt"))
        )
        kb_config = KBLaMConfig.from_pretrained(
            os.path.join(model_dir_to_resume, "kb_config.json")
        )
    else:
        kb_config = KBLaMConfig(
            sep_query_head=sep_query_head,
            kb_layer_frequency=kb_token_layer_frequency,
        )

    encoder.train()

    kbretriever = KBRetriever(
        encoder,
        training_set,
        key_embds=key_embds,  # type: ignore
        value_embds=value_embds,  # type: ignore
    )

    logger.info("Model ready 🚀")

    # Get the training started
    llm_ckpt_name = (
        f"{prefix_string}KeyFrom{key_embd_src}_{encoder_spec}_{dataset_name}_{llm_type}"
    )
    total_steps = np.ceil(len(training_set) / B).astype(int)
    print("Total steps for training:",total_steps)
    trainer = Trainer(
        model,  # type: ignore
        kbretriever,
        tokenizer,
        kb_token_layer_frequency,
        total_steps,
        args.lr,
        device,
        use_lr_decay,
        kb_size,  # type: ignore
        llm_ckpt_name,
        model_save_dir,
        sep_query_head=sep_query_head,
    )

    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    trainer.train(
        training_set,
        B,
        gradient_accm_step,
        outlier_num,
        use_data_aug=use_data_aug,
        multi_entities=multi_entities,
        use_extended_qa=use_extended_qa,
        save_period=110,
        resumed_step=resumed_step,
        kb_config=kb_config,
    )


if __name__ == "__main__":
    main()
