import argparse
import json
import os

import glob
from torch import nn
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from kblam.gpt_session import GPT
from kblam.utils.data_utils import DataPoint
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/home/schatterjee1/KBLAM/sql-coder-special-Tokens")
tokenizer.pad_token = tokenizer.eos_token
embedding_model = nn.Embedding(
           32016, 4096,None
        )
embedding_model = embedding_model.to('cuda')


# def load_all_chunks(chunk_dir, prefix="embd"):
#     files = sorted(glob.glob(f"{chunk_dir}/{prefix}_chunk_*.npy"))
#     all_chunks = [np.load(f) for f in files]
#     key_embeds = np.concatenate(all_chunks, axis=0)
#     np.save(
#         "/home/schatterjee1/KBLAM/KBLaM/datasets/train_dev_bird_DB_Schemas_SAME_EMBED_embd_key.npy",
#         np.array(key_embeds),
#     )
    



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings"],
    )
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument(
        "--dataset_path",
        default="/home/schatterjee1/KBLAM/KBLaM/datasets/random_dev_bird_DB_Schemas_90.json",
        type=str,
        required=False,
        help="Path to the dataset in JSON format.",
    )
    parser.add_argument("--output_path", type=str, default="dataset")

    args = parser.parse_args()
    return args


# def compute_embeddings(
#     encoder_model_spec: str, dataset: list[DataPoint], part: str, batch_size: int = 4
# ) -> np.array:
#     """Compute embeddings for the given dataset in batches using the encoder model spec."""
#     embeddings = []
#     all_elements = []
#     for entity in dataset:
#         if part == "key_string":
#             all_elements.append(entity.key_string)
#         elif part == "description":
#             all_elements.append(entity.description)
#         else:
#             raise ValueError(f"Part {part} not supported.")
#     chunks = [
#         all_elements[i : i + batch_size]
#         for i in range(0, len(all_elements), batch_size)
#     ]

#     #model = SentenceTransformer(encoder_model_spec, device="cuda")
#     chunk_index=0
#     for chunk in tqdm(chunks):
#         tokenizer_output = tokenizer(chunk, return_tensors="pt", padding=True).to(
#             'cuda'
#         )
#         embd = embedding_model(tokenizer_output["input_ids"])
        
#         #embd = embedding_model(chunk)
#         # print("shape of embedding:",embd.shape)
        
#         embeddings.append(embd)
#         print("Chunk index:",chunk_index)
#         chunk_index+=1
        

#     embeddings = np.concatenate(embeddings, 0)
#     # assert len(embeddings) == len(all_elements)
#     # args.output_path="/home/schatterjee1/KBLAM/KBLaM/datasets"
#     # args.dataset_name="train_dev_bird_DB_Schemas_100"
#     # np.save(
#     #     f"{args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy",
#     #     np.array(key_embeds),
#     # )
#     return embeddings


import torch
import numpy as np
from tqdm import tqdm
import os

def compute_embeddings(
    encoder_model_spec: str,
    dataset: list,
    part: str,
    batch_size: int = 8,
    save_dir: str = "/scratch-shared/chatty/chunked_embeddings_inference",
    save_prefix: str = "embd"
) -> list:

    """Compute embeddings and save each chunk to disk to avoid CUDA OOM."""
    os.makedirs(save_dir, exist_ok=True)

    all_elements = []
    for entity in dataset:
        if part == "key_string":
            all_elements.append(entity.key_string)
        elif part == "description":
            all_elements.append(entity.description)
        else:
            raise ValueError(f"Part {part} not supported.")
    batch_size = 8
    chunks = [
        all_elements[i : i + batch_size]
        for i in range(0, len(all_elements), batch_size)
    ]

    saved_chunk_paths = []

    for chunk_index, chunk in enumerate(tqdm(chunks)):
        # Tokenize and send to CUDA
        tokenizer_output = tokenizer(chunk, return_tensors="pt", padding=True).to('cuda')

        # Encode using your embedding model
        with torch.no_grad():
            embd = embedding_model(tokenizer_output["input_ids"]).detach().cpu().numpy()

        # Save chunk to disk
        chunk_path = os.path.join(save_dir, f"{save_prefix}_chunk_{chunk_index}.npy")
        np.save(chunk_path, embd)
        saved_chunk_paths.append(chunk_path)

        # Free up memory
        del embd, tokenizer_output
        torch.cuda.empty_cache()

    return saved_chunk_paths  # paths of saved chunks


if __name__ == "__main__":
    args = parser_args()
    print(args.dataset_path)
    with open(args.dataset_path, "r") as file:
        data_list = json.load(file) 
        #print(data_list)
        dataset = [DataPoint(**item) for item in data_list]

    if args.model_name == "all-MiniLM-L6-v2":
        #pass
        key_embeds = compute_embeddings(args.model_name, dataset, "key_string")
        #value_embeds = compute_embeddings(args.model_name, dataset, "description")
    elif args.model_name in ["ada-embeddings", "text-embedding-3-large"]:
        gpt = GPT(args.model_name, args.endpoint_url)

        key_embeds = []
        value_embeds = []

        for entity in tqdm(dataset):
            key_embeds.append(gpt.generate_embedding(entity.key_string))
            value_embeds.append(gpt.generate_embedding(entity.description))
    else:
        raise ValueError(f"Model {args.model_name} not supported.")

    os.makedirs(args.output_path, exist_ok=True)

    if args.model_name == "all-MiniLM-L6-v2":
        save_name = "all-MiniLM-L6-v2"
    elif args.model_name == "ada-embeddings":
        save_name = "OAI"
    else:
        save_name = "BigOAI"

    args.output_path="/home/schatterjee1/KBLAM/KBLaM/datasets"
    args.dataset_name="random_dev_bird_DB_Schemas_90"
    # np.save(
    #     f"{args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy",
    #     np.array(key_embeds),
    # )
    # np.save(
    #     f"{args.output_path}/{args.dataset_name}_{save_name}_embd_value.npy",
    #     np.array(value_embeds),
    # )

    #load_all_chunks("/scratch-shared/chatty/chunked_embeddings")
