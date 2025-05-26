# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Literal, TypedDict, List
from functools import partial

import torch
import random

import transformers
import webdataset as wds
from llava.multifile_tariterators import tarfile_to_samples

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import SHARD_SHUFFLE_BUFSIZE, SHARD_SHUFFLE_INITIAL, SAMPLE_SHUFFLE_BUFSIZE, SAMPLE_SHUFFLE_INITIAL
from llava.constants import IMG_CONTEXT_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.taisu2_preprocess import load_image as dynres_load_image
from llava.taisu2_preprocess import taisu2_wds_map

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="OpenGVLab/InternVL3-2B")
    version: Optional[str] = field(default="internvl2_5")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_vit_pos_embedding: bool = field(default=False)
    tune_vision_tower: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    # data arguments for dynamic resolution strategy (InternVL-1.5 and later)
    dynamic_resolution: Optional[bool] = field(default=True)
    base_img_size: Optional[int] = field(default=448)
    min_subimg_num: Optional[int] = field(default=1)
    max_subimg_num: Optional[int] = field(default=12)
    use_thumbnail: Optional[bool] = field(default=True)
    # data arguments for tokenizer
    padding: Optional[str] = field(default="do_not_pad")
    padding_side: Optional[str] = field(default=None)
    return_tensors: Optional[str] = field(default=None)
    return_attention_mask: Optional[bool] = field(default=None)
    # data arguments for webdataset
    wds_shards_folder: Optional[str] = field(default=None)
    wds_shards_subfolder: Optional[str] = field(default="rename_and_rearchive")
    wds_nsamples_per_epoch: Optional[int] = field(default=None)
    wds_last_batch: Optional[bool] = field(default=True)
    wds_shuffle_seed: Optional[int] = field(default=42)
    # data arguments for image-text preprocessing
    txts_separator: Optional[str] = field(default="\n")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default="Taisu2")
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default=None)
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'mlp1']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # also save pos embedding
        if getattr(trainer.args, "tune_vit_pos_embedding", False):
            keys_to_match.extend(['vision_tower.embeddings.position_embedding', 'vision_model.embeddings.position_embedding'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        print("weight to save:", weight_to_save.keys())
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.name:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:  # compatible with transformers==4.32.0
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

            if i != 0 and not tokenizer.legacy:  # compatible with transformers==4.32.0
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_internvl2_5(
    sources: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer, 
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()

    # Mask targets
    sep = conv.sep + conv.roles[1]  # <|im_end|>\n<|im_start|>assistant\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx: conv_idx + 2]))    # user + gpt
        cur_len = 0
        target[: cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if "v1" in conversation_lib.default_conversation.name:
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if "internvl2_5" in conversation_lib.default_conversation.name or "internvl3" in conversation_lib.default_conversation.name:
        return preprocess_internvl2_5(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.MPT:
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
                 self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments
                ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        flag = False
        while not flag:
            try:
                sources = self.list_data_dict[i]
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
                if 'image' in sources[0]:
                    image_file = self.list_data_dict[i]['image']
                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor
                    if not self.data_args.dynamic_resolution:
                        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                        if self.data_args.image_aspect_ratio == 'pad':
                            def expand2square(pil_img, background_color):
                                width, height = pil_img.size
                                if width == height:
                                    return pil_img
                                elif width > height:
                                    result = Image.new(pil_img.mode, (width, width), background_color)
                                    result.paste(pil_img, (0, (width - height) // 2))
                                    return result
                                else:
                                    result = Image.new(pil_img.mode, (height, height), background_color)
                                    result.paste(pil_img, ((height - width) // 2, 0))
                                    return result
                            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        else:
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        img_file = pathlib.Path(image_folder) / image_file
                        input_size = self.data_args.base_img_size
                        min_num = self.data_args.min_subimg_num
                        max_num = self.data_args.max_subimg_num
                        use_thumbnail = self.data_args.use_thumbnail
                        image = dynres_load_image(
                                                  image_file=img_file, input_size=input_size, 
                                                  min_num=min_num, max_num=max_num, 
                                                  use_thumbnail=use_thumbnail
                                                 )
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                # elif "video" in sources[0]:
                #     pass
                else:
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(
                    sources,
                    self.tokenizer,
                    has_image=('image' in self.list_data_dict[i]))
                if isinstance(i, int):
                    data_dict = dict(
                                     input_ids=data_dict["input_ids"][0],
                                     labels=data_dict["labels"][0]
                                    )

                # image exist in the data
                if 'image' in self.list_data_dict[i]:
                    data_dict['image'] = image
                # elif 'video' in self.list_data_dict[i]:
                #     pass
                elif self.data_args.is_multimodal:
                    # image does not exist in the data, but the model is multimodal
                    if not self.data_args.dynamic_resolution:
                        crop_size = self.data_args.image_processor.crop_size
                    else:
                        crop_size = self.data_args.base_img_size
                    data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                flag = True
            except Exception as e:
                print(e)
                i = random.randint(0, len(self.list_data_dict) - 1)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
                                                    input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id
                                                   )
        labels = torch.nn.utils.rnn.pad_sequence(
                                                 labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX
                                                )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
                     input_ids=input_ids,
                     labels=labels,
                     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                    )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if not self.data_args.dynamic_resolution:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
            else:
                if all(x is not None and x.shape[1: ] == images[0].shape[1: ] for x in images):
                    batch["images"] = torch.cat(images)
                else:
                    batch["images"] = images
        # if 'video' in instances[0]:
        #     pass

        return batch


def make_supervised_data_module(
                                tokenizer: transformers.PreTrainedTokenizer,
                                data_args
                               ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
                                          tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args
                                         )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator
               )


class WdsCollatorOutput(TypedDict, total=False):
    pixel_values: Optional[torch.Tensor]
    input_ids: Optional[torch.LongTensor]
    attention_mask: Optional[torch.LongTensor]
    image_flags: Optional[torch.Tensor]
    labels: Optional[torch.LongTensor]
    data_names: Optional[List[str]]


@dataclass
class DataCollatorForWebDataset(object):
    """Collate image-text data for webdataset IterableDataset"""
    tokenizer: transformers.PreTrainedTokenizer
    pad_token_id: int
    conv_name: str
    is_train: bool = True

    def __call__(
                 self, 
                 img_text_data: Sequence[Dict]
                ) -> WdsCollatorOutput:
        native_input_ids = [data["input_ids"] for data in img_text_data]
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                                                          native_input_ids, 
                                                          batch_first=True, 
                                                          padding_value=self.pad_token_id
                                                         )
        if self.is_train:
            batch_labels = (data["labels"] for data in img_text_data)
            batch_labels = torch.nn.utils.rnn.pad_sequence(
                                                           list(batch_labels), 
                                                           batch_first=True, 
                                                           padding_value=IGNORE_INDEX
                                                          )
        else:
            max_input_len = batch_input_ids.shape[-1]
            native_attn_mask = []
            for input_ids in native_input_ids:
                attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device, requires_grad=False)
                pad_mask = torch.zeros((max_input_len - input_ids.shape[0], ), dtype=torch.long, device=input_ids.device, requires_grad=False)
                native_attn_mask.append(torch.cat((pad_mask, attn_mask), dim=0))
            batch_attn_mask = torch.stack(native_attn_mask, dim=0)
            batch_names = [data["data_name"] for data in img_text_data]
        batch_no_imgs = all("pixel_values" not in data for data in img_text_data)
        if batch_no_imgs:
            if self.is_train:
                return dict(
                            input_ids=batch_input_ids, 
                            labels=batch_labels, 
                           )
            else:
                return dict(
                            input_ids=batch_input_ids, 
                            attention_mask=batch_attn_mask, 
                            data_names=batch_names, 
                           )
        else:
            batch_imgs = (data["pixel_values"] for data in img_text_data)
            batch_imgs = torch.cat(tuple(batch_imgs), dim=0)
            if "internvl2_5" in self.conv_name or "internvl3" in self.conv_name:
                if self.is_train:
                    image_flags = torch.ones((batch_imgs.shape[0], 1), dtype=torch.int)
                    return dict(
                                pixel_values=batch_imgs, 
                                input_ids=batch_input_ids, 
                                image_flags=image_flags, 
                                labels=batch_labels, 
                               )
                else:
                    return dict(
                                pixel_values=batch_imgs, 
                                input_ids=batch_input_ids, 
                                attention_mask=batch_attn_mask, 
                                data_names=batch_names, 
                               )


def make_wds_data_module(
                         tokenizer: transformers.PreTrainedTokenizer, 
                         data_args: DataArguments = None
                        ) -> Dict:
    """Make training and evaluation webdataset iterable for pretraining & supervised fine-tuning"""
    data_root_dir = pathlib.Path(os.getenv("HOME", "")) / "datasets" / "Taisu2_datasets"
    wds_shards_p = data_root_dir / f"{data_args.wds_shards_folder}" / f"{data_args.wds_shards_subfolder}" / "image-text-pairs"
    if not wds_shards_p.exists():
        raise FileNotFoundError(f"sharded tar files directory - {wds_shards_p}, does not exist!")

    def get_first_and_last_tarnames() -> Tuple[str]:
        tar_names = sorted(os.listdir(wds_shards_p))
        return tar_names[0].split(".")[0], tar_names[-1].split(".")[0]

    first_tarname, last_tarname = get_first_and_last_tarnames()
    wds_train_urls = f"{wds_shards_p}/" + "{" + f"{first_tarname}.." + f"{last_tarname}" + "}.tar"
    wds_train_pipeline = [wds.SimpleShardList(urls=wds_train_urls)]
    wds_train_pipeline.append(
        wds.detshuffle(bufsize=SHARD_SHUFFLE_BUFSIZE, initial=SHARD_SHUFFLE_INITIAL, seed=data_args.wds_shuffle_seed)
    )
    wds_train_pipeline.append(wds.split_by_node)
    wds_train_pipeline.append(wds.split_by_worker)
    wds_train_pipeline.append(tarfile_to_samples())
    wds_train_pipeline.append(wds.detshuffle(bufsize=SAMPLE_SHUFFLE_BUFSIZE, initial=SAMPLE_SHUFFLE_INITIAL, seed=data_args.wds_shuffle_seed))
    wds_train_map = partial(taisu2_wds_map, is_train=True, tokenizer=tokenizer, data_args=data_args)
    wds_train_pipeline.append(wds.map(wds_train_map))
    train_web_dataset = wds.DataPipeline(*wds_train_pipeline)
    if data_args.wds_nsamples_per_epoch:
        train_web_dataset.with_epoch(nsamples=data_args.wds_nsamples_per_epoch)
        train_web_dataset.with_length(data_args.wds_nsamples_per_epoch)

    wds_collator = DataCollatorForWebDataset(
                                             tokenizer=tokenizer, 
                                             pad_token_id=tokenizer.pad_token_id, 
                                             conv_name=conversation_lib.default_conversation.name
                                            )

    return dict(
        train_dataset=train_web_dataset, 
        eval_dataset=None, 
        data_collator=wds_collator, 
    )


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # params check
    if data_args.return_tensors == "":
        data_args.return_tensors = None
    if data_args.txts_separator == "\\n":
        data_args.txt_separator = "\n"
    if model_args.vision_tower == "":
        model_args.vision_tower = None
    if training_args.cache_dir == "":
        training_args.cache_dir = None
    if training_args.lora_weight_path == "":
        training_args.lora_weight_path = None

    local_rank = training_args.local_rank
    training_args._frozen = False  # compatible with transformers==4.32.0
    data_args._frozen = False  # compatible with transformers==4.32.0
    model_args._frozen = False  # compatible with transformers==4.32.0
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    internvl_flag = "internvl2_5" in model_args.model_name_or_path.lower() or "internvl3" in model_args.model_name_or_path.lower()
    if internvl_flag:
        model_args.mm_use_im_start_end = False
        model_args.mm_use_im_patch_token = False

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    elif internvl_flag:
        model_args.vision_tower = None
        if attn_implementation == "flash_attention_2":
            use_flash_attn = True
        else:
            use_flash_attn = False
        model = InternVLChatModel.from_pretrained(
                                                  model_args.model_name_or_path, use_flash_attn=use_flash_attn, trust_remote_code=False, 
                                                  cache_dir=training_args.cache_dir, 
                                                  torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)), 
                                                  **bnb_model_from_pretrained_args
                                                 )
        data_args.context_token_per_img = model.num_image_token
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        if not internvl_flag:
            model.model.requires_grad_(False)
        else:
            model.vision_model.requires_grad_(False)
            model.mlp1.requires_grad_(False)
            model.language_model.requires_grad_(False)
            model.language_model.lm_head.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
                                 r=training_args.lora_r,
                                 lora_alpha=training_args.lora_alpha,
                                 target_modules=find_all_linear_names(model),
                                 lora_dropout=training_args.lora_dropout,
                                 bias=training_args.lora_bias,
                                 task_type="CAUSAL_LM",
                                )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif internvl_flag:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                                                               model_args.model_name_or_path, 
                                                               cache_dir=training_args.cache_dir, 
                                                               model_max_length=training_args.model_max_length, 
                                                               padding_side="right", 
                                                               trust_remote_code=True, 
                                                               use_fast=False
                                                              )
        if not model.img_context_token_id:
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["internvl2_5"]

    if model_args.vision_tower:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        # freeze llm: only tune mlp layer
        if model_args.tune_mm_mlp_adapter or training_args.freeze_llm:
            print("Only tune mlp adapter.")
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # tune vit position embedding
        model.config.tune_vit_pos_embedding = training_args.tune_vit_pos_embedding = model_args.tune_vit_pos_embedding
        if model_args.tune_vit_pos_embedding:
            print("Tuning ViT position embedding.")
            for name, p in model.get_model().vision_tower.named_parameters():
                if "position_embedding" in name:
                    p.requires_grad = True
                    print("\tvit pos embedding name: ", name)

        # tune vision tower
        model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower
        for vision_p in vision_tower.parameters():
            if model_args.tune_vision_tower:
                vision_p.requires_grad = True
            else:
                vision_p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    elif internvl_flag:
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter or training_args.freeze_llm:
            print("Only tune mlp adapter")
            model.requires_grad_(False)
            for p in model.mlp1.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            model.requires_grad_(True)
            for p in model.mlp1.parameters():
                p.requires_grad = False

        # tune Intern-ViT positional embedding
        model.config.tune_vit_pos_embedding = training_args.tune_vit_pos_embedding = model_args.tune_vit_pos_embedding
        if model_args.tune_vit_pos_embedding:
            print("Tuning ViT position embedding")
            for name, p in model.vision_model.named_parameters():
                if "position_embedding" in name:
                    p.requires_grad = True
                    print("\tIntern-ViT pos embedding name: ", name)

        # tune Intern-ViT
        model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower
        for vision_p in model.vision_model.parameters():
            if model_args.tune_vision_tower:
                vision_p.requires_grad = True
            else:
                vision_p.requires_grad = False

        if training_args.bits in (4, 8):
            model.mlp1.to(dtype=compute_dtype, device=training_args.device)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if internvl_flag:
                for layer in model.mlp1:
                    if isinstance(layer, torch.nn.LayerNorm):
                        layer = layer.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if data_args.wds_shards_folder:
        data_module = make_wds_data_module(
                                           tokenizer=tokenizer, 
                                           data_args=data_args
                                          )
    else:
        data_module = make_supervised_data_module(
                                                  tokenizer=tokenizer, 
                                                  data_args=data_args
                                                 )

    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    trainer = LLaVATrainer(
                           model=model,
                           processing_class=tokenizer,
                           args=training_args,
                           **data_module
                          )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
