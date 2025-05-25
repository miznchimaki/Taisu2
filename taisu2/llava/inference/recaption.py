import argparse
from argparse import Namespace
import math
import os
import json
import torch.distributed
from tqdm import tqdm
from functools import partial
from typing import Union, Tuple, TypedDict
from pathlib import Path, PosixPath

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import webdataset as wds
from llava.constants import SHARD_SHUFFLE_BUFSIZE, SHARD_SHUFFLE_INITIAL
from llava.constants import SAMPLE_SHUFFLE_BUFSIZE, SAMPLE_SHUFFLE_INITIAL
from llava.constants import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN
from llava.conversation import conv_templates, set_default_conv_template
from llava.model import LlavaMptForCausalLM, LlavaLlamaForCausalLM
from llava.model import InternVLChatConfig, InternVLChatModel
from llava.multifile_tariterators import tarfile_to_samples
from llava.taisu2_preprocess import taisu2_wds_map
from llava.utils import disable_torch_init
from llava.train import DataCollatorForWebDataset


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def set_conv_tempalte(args: Namespace = None):
    if args.conv_template_name is None:
        print(f"conversation template name is None, set conversation tempalte to the default one")
        from llava.conversation import default_conversation
        conv = default_conversation.copy()
    elif args.conv_template_name in conv_templates:
        print(f"get conversation name: {args.conv_template_name}")
        set_default_conv_template(args.conv_template_name)
    else:
        raise KeyError(f"get a wrong conversation name: {args.conv_template_name}, which does not exist!")
    return


def init_distributed(args: Namespace = None):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise KeyError(f"either environmental variables `RANK` or `WORLD_SIZE` does not exist, cannot execute DDP initialization normally")
    num_gpu_per_node = torch.cuda.device_count()
    args.rank = str(os.environ["RANK"])
    args.world_size = str(os.environ["WORLD_SIZE"])
    args.local_rank = str(int(args.rank) & num_gpu_per_node)
    torch.distributed.init_process_group(
                                         backend="nccl", 
                                         init_method="env://", 
                                         world_size=int(args.world_size), 
                                         rank=int(args.rank)
                                        )
    print(f"DDP initialized completed at process with rank {args.rank}")
    torch.distributed.barrier()
    return


class OutputDict1(TypedDict):
    tokenizer: transformers.PreTrainedTokenizer
    model: transformers.PreTrainedModel


def create_tokenizer_and_model(args: Namespace = None) -> OutputDict1:
    model_name_or_path = args.model_name_or_path
    mpt_flag = "mpt" in model_name_or_path
    internvl_flag = "internvl2_5" in model_name_or_path or "internvl3" in model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
                                              model_name_or_path, 
                                              cache_dir=args.cache_dir, 
                                              use_fast=args.use_fast, 
                                              trust_remote_code=args.trust_remote_code, 
                                              model_max_length=args.model_max_length, 
                                              padding=args.padding, 
                                              truncation=args.truncation, 
                                              padding_side=args.padding_side if not mpt_flag else "right", 
                                              return_tensors=args.return_tensors, 
                                              return_attention_mask=args.return_attention_mask, 
                                             )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    device_map = {"": f"cuda:{args.local_rank}"}
    if not internvl_flag:
        if "mpt" in model_name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            if args.mpt_attn_impl is not None:
                config.attn_config["attn_impl"] = args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                                                        model_name_or_path, 
                                                        config=config, 
                                                        cache_dir=args.cache_dir, 
                                                        device_map=device_map, 
                                                       )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                                                          model_name_or_path, 
                                                          cache_dir=args.cache_dir, 
                                                          torch_dtype=torch.bfloat16 if args.data_type == "bfloat16" else (torch.float16 if args.data_type == "float16" else torch.float32), 
                                                          attn_implementation="flash_attention_2" if args.use_flash_attn else None, 
                                                          device_map=device_map, 
                                                         )
    else:
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model = InternVLChatModel.from_pretrained(
                                                  model_name_or_path, 
                                                  cache_dir=args.cache_dir, 
                                                  trust_remote_code=args.trust_remote_code, 
                                                  use_flash_attn=args.use_flash_attn, 
                                                  torch_dtype=torch.bfloat16 if args.data_type == "bfloat16" else (torch.float16 if args.data_type == "float16" else torch.float32), 
                                                  device_map=device_map, 
                                                 )
        model.img_context_token_id = img_context_token_id
        args.context_token_per_img = model.num_image_token
    model.eval()
    model.config.use_cache = False

    return dict(tokenizer=tokenizer, model=model)


def create_dataloader(
                      tokenizer: transformers.PreTrainedTokenizer, 
                      args: Namespace = None
                     ) -> DataLoader:
    root_dir = Path(os.getenv("HOME", None)) / "datasets" / "Taisu2_datasets"
    tars_p = root_dir / args.tars_folder / args.tars_subfolder / "image-text-pairs"
    if not tars_p.exists():
        raise FileNotFoundError(f"tar files directory - {tars_p}, does not exist!")

    def get_first_and_last_tarname() -> Tuple[str, str]:
        tar_names = sorted(os.listdir(tars_p))
        return tar_names[0].split(".")[0], tar_names[-1].split(".")[0]

    first_tarname, last_tarname = get_first_and_last_tarname()
    tar_urls = f"{tars_p}/" + "{" + f"{first_tarname}.." + f"{last_tarname}" + "}.tar"
    wds_pipeline = [wds.SimpleShardList(urls=tar_urls)]
    if args.wds_shuffle_seed is not None:
        wds_pipeline.append(wds.detshuffle(bufsize=SHARD_SHUFFLE_BUFSIZE, initial=SHARD_SHUFFLE_INITIAL, seed=args.wds_shuffle_seed))
    wds_pipeline.append(wds.split_by_node)
    wds_pipeline.append(wds.split_by_worker)
    wds_pipeline.append(tarfile_to_samples())
    if args.wds_shuffle_seed is not None:
        wds_pipeline.append(wds.detshuffle(bufsize=SAMPLE_SHUFFLE_BUFSIZE, initial=SAMPLE_SHUFFLE_INITIAL, seed=args.wds_shuffle_seed))
    recaption_map_func = partial(taisu2_wds_map, is_train=False, tokenizer=tokenizer, data_args=args)
    wds_pipeline.append(wds.map(recaption_map_func))
    recaption_wds = wds.DataPipeline(*wds_pipeline)
    assert args.num_samples is not None, f"number of samples for dataset based on webdataset must be applied!"
    recaption_wds.with_epoch(nsamples=args.num_samples)
    recaption_wds.with_length(n=args.num_samples)

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
    batch_num = math.ceil(args.num_samples // args.batch_size)
    args.batch_num = batch_num
    return data_loader


@torch.inference_mode()
def recaption(
              tokenizer: transformers.PreTrainedTokenizer, 
              model: transformers.PreTrainedModel, 
              data_loader: DataLoader, 
              args: Namespace = None
             ):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None, help="output directory of recaption json file")
    parser.add_argument("--conv-template-name", type=str, default=None, help="conversation template name")

    # dataloader params
    parser.add_argument("--num-workers", type=int, default=8, help="number workers for DataLoader")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for recaption DataLoader")
    parser.add_argument("--pin-memory", type=bool, default=True, help="pin_memory param for DataLoader")
    parser.add_argument("--drop-last", type=bool, default=False, help="drop_last param for DataLoader")

    # tokenizer params
    parser.add_argument("--use-fast", type=bool, default=False, help="whether or not to use fast text tokenizer")
    parser.add_argument("--trust-remote-code", type=bool, default=False, 
                        help="whether or not to allow for custom defined tokenizer and model code")
    parser.add_argument("--cache-dir", type=str, default=None, help="path where a downloaded pretrained model is cached")
    parser.add_argument("--model-max-length", type=int, default=12288, help="maximum length for tokenizer and model")
    parser.add_argument("--padding", type=str, default="longest", choices=("longest", "max_length", "do_not_pad"), 
                        help="padding strategy for text tokenizer")
    parser.add_argument("--padding-side", type=str, default="right", choices=("left", "right"), help="padding side for text tokenizer")
    parser.add_argument("--return-tensors", type=str, default=None, choices=("tf", "pt", "np"), help="returned tensors type for text tokenizer")
    parser.add_argument("--return-attention-mask", type=bool, default=True, help="whether text tokenizer returns attention mask")

    # dynamic resolution params
    parser.add_argument("--base-img-size", type=int, default=448, help="base image size for dynamic resolution strategy")
    parser.add_argument("--min-subimg-num", type=int, default=1, help="minimum sub-image number for dynamic resolution")
    parser.add_argument("--max-subimg-num", type=int,default=12, help="maximum sub-image number for dynamic resolution")
    parser.add_argument("--use-thumbnail", type=bool, default=True, help="whether using thumbnail image for dynamic resolution")

    # webdataset params
    parser.add_argument("--tars-folder", type=str, default=None, help="webdataset tar file root folder")
    parser.add_argument("--tars-subfolder", type=str, default=None, help="webdataset tar file sub-root folder")
    parser.add_argument("--num-samples", type=int, default=None, help="number of image-alttext pairs for recaptioning in total")
    parser.add_argument("--wds-shuffle-seed", type=int, default=None, help="random seed for webdataset shuffling")

    # model params
    parser.add_argument("--model-name-or-path", type=str, default="OpenGVLab/InternVL3-2B", 
                        help="model remote repository name or local directory")
    parser.add_argument("--data-type", type=str, choices=("bfloat16", "float16"), default="bfloat16", 
                        help="Tensor data type for model and input data")
    parser.add_argument("--mpt-attn-impl", type=str, default="triton")
    parser.add_argument("--use-flash-attn", type=bool, default=True, help="whether model using flash atention")

    # generation params
    parser.add_argument("--max-length", type=int, default=None, help="maximum length for both prompt and generated tokens")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="maximum generated new tokens number")
    parser.add_argument("--min-length", type=int, default=None, help="minimum length for both prompt and generated tokens")
    parser.add_argument("--do-sample", type=bool, default=False, help="whether or not to use sampling generation, use greedy decoding otherwise")
    parser.add_argument("--num-beams", type=int, default=5, help="beam number for beam search based generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value used for moduling generation probabilites")
    parser.add_argument("--top-k", type=int, default=50, help="highest probability tokens number for top-k filtering")
    parser.add_argument("--top-p", type=float, default=1.0, help="low bound of accumulated probability for top-p filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="parameter for repetition penalty, 1.0 means no penalty")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="exponential penalty to the length when using beam-based generation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_conv_tempalte(args=args)
    init_distributed(args=args)

    create_res = create_tokenizer_and_model(args=args)
    tokenizer = create_res["tokenizer"]; model = create_res["model"]
    data_loader = create_dataloader(tokenizer, args=args)

    recaption(tokenizer, model, data_loader, args=args)
