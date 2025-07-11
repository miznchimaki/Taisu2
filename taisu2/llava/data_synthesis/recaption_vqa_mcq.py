import argparse
from argparse import Namespace
import math
import os
import shutil
import json
from datetime import timedelta
from tqdm import tqdm
from functools import partial
from typing import Union, Tuple, TypedDict, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import deepspeed
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import ModelOutput
import webdataset as wds
from llava.constants import SHARD_SHUFFLE_BUFSIZE, SHARD_SHUFFLE_INITIAL
from llava.constants import SAMPLE_SHUFFLE_BUFSIZE, SAMPLE_SHUFFLE_INITIAL
from llava.constants import IMG_CONTEXT_TOKEN
from llava.conversation import conv_templates, set_default_conv_template
from llava.model import LlavaMptForCausalLM, LlavaLlamaForCausalLM
from llava.model import InternVLChatModel
from llava.multifile_tariterators import tarfile_to_samples
from llava.taisu2_preprocess import taisu2_wds_map
from llava.train import DataCollatorForWebDataset


def init_distributed(args: Namespace = None):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise KeyError(f"either environmental variables `RANK` or `WORLD_SIZE` does not exist, cannot execute DDP initialization normally")
    num_gpu_per_node = torch.cuda.device_count()
    args.rank = str(os.environ["RANK"])
    args.world_size = str(os.environ["WORLD_SIZE"])
    args.local_rank = str(int(args.rank) % num_gpu_per_node)
    deepspeed.init_distributed(
                               dist_backend="nccl", 
                               timeout=timedelta(days=2), 
                               init_method="env://", 
                               rank=int(args.rank), 
                               world_size=int(args.world_size), 
                              )

    if int(args.rank) == 0:
        print(f"DDP initialization finished for {args.world_size} processes")
    torch.cuda.set_device(int(args.local_rank))
    deepspeed.comm.barrier()
    return


def set_conv_tempalte(args: Namespace = None):
    if args.conv_template_name is None:
        if int(args.rank) == 0:
            print(f"conversation template name is None, set conversation tempalte to the default one")
        from llava.conversation import default_conversation
        conv = default_conversation.copy()
    elif args.conv_template_name in conv_templates:
        if int(args.rank) == 0:
            print(f"get conversation name: {args.conv_template_name}")
        set_default_conv_template(args.conv_template_name)
        conv = conv_templates[args.conv_template_name].copy()
    else:
        raise KeyError(f"get a wrong conversation name: {args.conv_template_name}, which does not exist!")
    args.conversation = conv
    return


class TokenizerAndModel(TypedDict):
    tokenizer: transformers.PreTrainedTokenizer
    model: transformers.PreTrainedModel


def create_tokenizer_and_model(args: Namespace = None) -> TokenizerAndModel:
    model_name_or_path = args.model_name_or_path
    mpt_flag = "mpt" in model_name_or_path
    internvl_flag = "internvl2_5" in model_name_or_path.lower() or "internvl3" in model_name_or_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(
                                              model_name_or_path, 
                                              use_fast=args.use_fast, 
                                              trust_remote_code=args.trust_remote_code, 
                                              model_max_length=args.model_max_length, 
                                              padding_side=args.padding_side if not mpt_flag else "right", 
                                             )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    cuda_device_str = f"cuda:{args.local_rank}"
    if not internvl_flag:
        if "mpt" in model_name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            if args.mpt_attn_impl is not None:
                config.attn_config["attn_impl"] = args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                                                        model_name_or_path, 
                                                        config=config, 
                                                        cache_dir=args.cache_dir, 
                                                       ).to(device=cuda_device_str)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                                                          model_name_or_path, 
                                                          cache_dir=args.cache_dir, 
                                                          torch_dtype=torch.bfloat16 if args.data_type == "bfloat16" else (torch.float16 if args.data_type == "float16" else torch.float32), 
                                                          attn_implementation="flash_attention_2" if args.use_flash_attn else None, 
                                                         ).to(device=cuda_device_str)
    else:
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model = InternVLChatModel.from_pretrained(
                                                  model_name_or_path, 
                                                  cache_dir=args.cache_dir, 
                                                  trust_remote_code=args.trust_remote_code, 
                                                  use_flash_attn=args.use_flash_attn, 
                                                  torch_dtype=torch.bfloat16 if args.data_type == "bfloat16" else (torch.float16 if args.data_type == "float16" else torch.float32), 
                                                 ).to(device=cuda_device_str)
        model.img_context_token_id = img_context_token_id
        args.context_token_per_img = model.num_image_token
    model = model.eval()
    model.config.use_cache = False

    return dict(
                tokenizer=tokenizer, 
                model=model
               )


def create_dataloader(
                      tokenizer: transformers.PreTrainedTokenizer, 
                      args: Namespace = None
                     ) -> DataLoader:
    root_dir = Path(os.getenv("HOME", None)) / "datasets" / "Taisu2_datasets"
    output_dir = root_dir / args.tars_folder / args.tars_subfolder / f"{args.recaption_idx}th_recaption"
    if int(args.rank) == 0:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(output_dir)
    tars_p = root_dir / args.tars_folder / args.tars_subfolder / "image-text-pairs"
    if not tars_p.exists():
        raise FileNotFoundError(f"tar files directory - {tars_p}, does not exist!")

    if args.drop_last:
        raise ValueError(f"when doing inference for recaption, no data should be discarded, not even single one")

    def get_first_and_last_tarname() -> Tuple[str, str]:
        tar_names = sorted(os.listdir(tars_p))
        return tar_names[0].split(".")[0], tar_names[-1].split(".")[0]

    first_tarname, last_tarname = get_first_and_last_tarname()
    tar_urls = f"{tars_p}/" + "{" + f"{first_tarname}.." + f"{last_tarname}" + "}.tar"
    wds_pipeline = [wds.SimpleShardList(urls=tar_urls)]
    if args.wds_shuffle_seed is not None:
        wds_pipeline.append(wds.detshuffle(bufsize=SHARD_SHUFFLE_BUFSIZE, initial=SHARD_SHUFFLE_INITIAL, seed=args.wds_shuffle_seed))
    wds_pipeline.append(tarfile_to_samples())
    wds_pipeline.append(wds.split_by_node)
    wds_pipeline.append(wds.split_by_worker)
    if args.wds_shuffle_seed is not None:
        wds_pipeline.append(wds.detshuffle(bufsize=SAMPLE_SHUFFLE_BUFSIZE, initial=SAMPLE_SHUFFLE_INITIAL, seed=args.wds_shuffle_seed))
    recaption_map_func = partial(
                                 taisu2_wds_map, 
                                 is_train=False, 
                                 inference_recaption=True, 
                                 tokenizer=tokenizer, 
                                 data_args=args
                                )
    wds_pipeline.append(wds.map(recaption_map_func))
    recaption_wds = wds.DataPipeline(*wds_pipeline)

    assert args.total_samples is not None, f"number of samples for dataset based on webdataset must be applied!"
    if args.num_workers:
        total_samples_per_worker = math.ceil(args.total_samples / (int(args.world_size) * args.num_workers))
        total_samples_per_rank = total_samples_per_worker * args.num_workers
    else:
        total_samples_per_worker = math.ceil(args.total_samples / int(args.world_size))
        total_samples_per_rank = total_samples_per_worker
    args.total_samples_per_worker = total_samples_per_worker
    args.total_samples_per_rank = total_samples_per_rank
    recaption_wds.with_epoch(nsamples=total_samples_per_worker)
    recaption_wds.with_length(n=total_samples_per_rank, silent=True)

    recaption_data_collator = DataCollatorForWebDataset(
                                                        tokenizer=tokenizer, 
                                                        pad_token_id=tokenizer.pad_token_id, 
                                                        conv_name=args.conv_template_name, 
                                                        is_train=False
                                                       )

    data_loader = DataLoader(
                             recaption_wds, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers, 
                             collate_fn=recaption_data_collator, 
                             pin_memory=args.pin_memory, 
                             drop_last=args.drop_last
                            )
    batch_num_per_rank = math.ceil(total_samples_per_rank / args.batch_size)
    total_batch_num = batch_num_per_rank * int(args.world_size)
    args.total_batch_num = total_batch_num
    args.batch_num_per_rank = batch_num_per_rank
    return data_loader


@torch.inference_mode()
def recaption(
              tokenizer: transformers.PreTrainedTokenizer, 
              model: transformers.PreTrainedModel, 
              data_loader: DataLoader, 
              args: Namespace = None
             ):
    eos_token_id = tokenizer.convert_tokens_to_ids(args.conversation.sep.strip())
    generation_cfg = dict()
    if args.max_new_tokens is not None:
        generation_cfg.update({"max_new_tokens": args.max_new_tokens})
    elif args.max_length is not None:
        generation_cfg.update({"max_length": args.max_length})
    remained_cfg = dict(
        pad_token_id=eos_token_id, 
        eos_token_id=eos_token_id, 
        min_length=args.min_length, 
        do_sample=args.do_sample, 
        num_beams=args.num_beams, 
        temperature=args.temperature, 
        top_k=args.top_k, 
        top_p=args.top_p, 
        repetition_penalty=args.repetition_penalty, 
        length_penalty=args.length_penalty, 
        num_return_sequences=args.num_return_sequences, 
        return_dict_in_generate=args.return_dict_in_generate, 
        output_attentions=args.output_attentions, 
        output_hidden_states=args.output_hidden_states, 
        output_scores=args.output_scores, 
        output_logits=args.output_logits, 
    )
    generation_cfg.update(remained_cfg)
    recaption_res = dict() # per rank recaption result
    recaption_p = Path(args.output_dir) / f"{args.recaption_idx}th_recaption_rank_{args.rank}.json"
    recaption_p.unlink(missing_ok=True)

    for batch_idx, batch_data in enumerate(tqdm(data_loader, 
                                                desc=f"{args.recaption_idx}th_recaption", 
                                                total=args.batch_num_per_rank + 5, 
                                                disable=int(args.rank) != 0, 
                                                dynamic_ncols=True)):
        pixel_values: torch.Tensor = batch_data["pixel_values"].to(dtype=model.dtype, device=model.device)
        input_ids: torch.LongTensor = batch_data["input_ids"].to(device=model.device)
        attention_mask: torch.LongTensor = batch_data["attention_mask"].to(device=model.device)
        data_names: List[str] = batch_data["data_names"]
        batch_recaption: Union[torch.Tensor | ModelOutput] = model.generate(
                                                                            pixel_values=pixel_values, 
                                                                            input_ids=input_ids, 
                                                                            attention_mask=attention_mask, 
                                                                            **generation_cfg
                                                                           )
        if args.return_dict_in_generate:
            recaption_strs = tokenizer.batch_decode(batch_recaption["sequences"], skip_special_tokens=True)
        else:
            recaption_strs = tokenizer.batch_decode(batch_recaption, skip_special_tokens=True)
        if len(data_names) != len(recaption_strs):
            raise ValueError(f"the {batch_idx}th batch on process with rank {args.rank} encounter a length inequality "
                             f"between input data_names({len(data_names)}) and outupt recaption strings ({len(recaption_strs)})!")
        for data_name, recaption_str in zip(data_names, recaption_strs):
            recaption_res.update({data_name: recaption_str})

    with open(recaption_p, mode="w", encoding="utf-8") as recaption_fp:
        json.dump(recaption_res, recaption_fp, ensure_ascii=False)
    print(f"process with rank {args.rank} has completed recaption results saving, into {recaption_p}")
    deepspeed.comm.barrier()
    return


def args_save(args: Namespace = None):
    args_p = Path(args.output_dir) / "arguments.json"
    args_dict = vars(args)
    _ = args_dict.pop("conversation")
    with open(args_p, mode="w", encoding="utf-8") as args_fp:
        json.dump(args_dict, args_fp, ensure_ascii=False)
    return


def recaption_res_aggregation(args: Namespace = None):
    root_dir = Path(os.getenv("HOME", None)) / "datasets" / "Taisu2_datasets"
    output_dir = root_dir / args.tars_folder / args.tars_subfolder / f"{args.recaption_idx}th_recaption"
    assert output_dir.exists(), f"recaption result directory: {output_dir} does not exist!"
    args.output_dir = str(output_dir)
    all_recaption_res = {}
    path_generator = Path(args.output_dir).glob("*th_recaption_rank_*.json")
    while True:
        try:
            recaption_res_p = next(path_generator)
            with open(recaption_res_p, mode="r", encoding="utf-8") as recaption_res_fp:
                recaption_res = json.load(recaption_res_fp)
                all_recaption_res.update(recaption_res)
        except StopIteration as _:
            break
    all_recaption_res_p = Path(args.output_dir) / f"{args.recaption_idx}th_recaption.json"
    with open(all_recaption_res_p, mode="w", encoding="utf-8") as res_fp:
        json.dump(all_recaption_res, res_fp, ensure_ascii=False)

    return


def parse_args():

    def eval_arg(x):
        try:
            return eval(x)
        except NameError as _:
            return str(x)

    parser = argparse.ArgumentParser()
    parser.add_argument("--recaption-idx", type=int, default=None, help="recaption iteration index")
    parser.add_argument("--conv-template-name", type=str, default=None, help="conversation template name")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None, 
                        help="remained argument for distribution")

    # dataloader params
    parser.add_argument("--num-workers", type=int, default=8, help="number workers for DataLoader")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for recaption DataLoader")
    parser.add_argument("--pin-memory", type=eval_arg, default=True, help="pin_memory param for DataLoader")
    parser.add_argument("--drop-last", type=eval_arg, default=False, help="drop_last param for DataLoader")

    # tokenizer params
    parser.add_argument("--use-fast", type=eval_arg, default=False, help="whether or not to use fast text tokenizer")
    parser.add_argument("--trust-remote-code", type=eval_arg, default=False, 
                        help="whether or not to allow for custom defined tokenizer and model code")
    parser.add_argument("--cache-dir", type=eval_arg, default=None, help="path where a downloaded pretrained model is cached")
    parser.add_argument("--model-max-length", type=int, default=12288, help="maximum length for tokenizer and model")
    parser.add_argument("--padding", type=eval_arg, default="do_not_pad", choices=("longest", "max_length", "do_not_pad", None), 
                        help="padding strategy for text tokenizer")
    parser.add_argument("--padding-side", type=eval_arg, default="left", choices=("left", "right"), help="padding side for text tokenizer")
    parser.add_argument("--return-tensors", type=eval_arg, default=None, choices=("tf", "pt", "np", None), help="returned tensors type for text tokenizer")
    parser.add_argument("--return-attention-mask", type=eval_arg, default=True, help="whether text tokenizer returns attention mask")

    # dynamic resolution params
    parser.add_argument("--base-img-size", type=int, default=448, help="base image size for dynamic resolution strategy")
    parser.add_argument("--min-subimg-num", type=int, default=1, help="minimum sub-image number for dynamic resolution")
    parser.add_argument("--max-subimg-num", type=int,default=12, help="maximum sub-image number for dynamic resolution")
    parser.add_argument("--use-thumbnail", type=eval_arg, default=True, help="whether using thumbnail image for dynamic resolution")

    # webdataset params
    parser.add_argument("--tars-folder", type=str, default=None, help="webdataset tar file root folder")
    parser.add_argument("--tars-subfolder", type=eval_arg, default=None, help="webdataset tar file sub-root folder")
    parser.add_argument("--total-samples", type=eval_arg, default=None, help="number of image-alttext pairs for recaptioning in total")
    parser.add_argument("--wds-shuffle-seed", type=eval_arg, default=None, help="random seed for webdataset shuffling")

    # model params
    parser.add_argument("--model-name-or-path", type=str, default="OpenGVLab/InternVL3-2B", 
                        help="model remote repository name or local directory")
    parser.add_argument("--data-type", type=str, choices=("float32", "bfloat16", "float16"), default="bfloat16", 
                        help="Tensor data type for model and input data")
    parser.add_argument("--mpt-attn-impl", type=str, default="triton")
    parser.add_argument("--use-flash-attn", type=eval_arg, default=True, help="whether model using flash atention")

    # generation params
    parser.add_argument("--max-length", type=eval_arg, default=None, help="maximum length for both prompt and generated tokens")
    parser.add_argument("--max-new-tokens", type=eval_arg, default=None, help="maximum generated new tokens number")
    parser.add_argument("--min-length", type=eval_arg, default=None, help="minimum length for both prompt and generated tokens")
    parser.add_argument("--do-sample", type=eval_arg, default=False, help="whether or not to use sampling generation, use greedy decoding otherwise")
    parser.add_argument("--num-beams", type=int, default=5, help="beam number for beam search based generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value used for moduling generation probabilites")
    parser.add_argument("--top-k", type=int, default=50, help="highest probability tokens number for top-k filtering")
    parser.add_argument("--top-p", type=float, default=1.0, help="low bound of accumulated probability for top-p filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="parameter for repetition penalty, 1.0 means no penalty")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="exponential penalty to the length when using beam-based generation")
    parser.add_argument("--num-return-sequences", type=int, default=1, help="independently computed returned sequence for each element in a batch")
    parser.add_argument("--return-dict-in-generate", type=eval_arg, default=False, help="whether or not to return a ModelOutput")
    parser.add_argument("--output-attentions", type=eval_arg, default=False, help="whether or not to return attention tensors")
    parser.add_argument("--output-hidden-states", type=eval_arg, default=False, help="whether or not to return hidden states of all layers")
    parser.add_argument("--output-scores", type=eval_arg, default=False, help="whether or not to return prediction scores")
    parser.add_argument("--output-logits", type=eval_arg, default=False, help="whether or not to return unprocessed logit scores")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_distributed(args=args)
    set_conv_tempalte(args=args)

    tokenizer_and_model = create_tokenizer_and_model(args=args)
    tokenizer = tokenizer_and_model["tokenizer"]; model = tokenizer_and_model["model"]
    data_loader = create_dataloader(tokenizer, args=args)

    recaption(tokenizer, model, data_loader, args=args)
    if int(args.rank) == 0:
        args_save(args=args)
        recaption_res_aggregation(args=args)
