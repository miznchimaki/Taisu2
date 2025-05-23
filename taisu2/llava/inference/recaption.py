import argparse
from argparse import Namespace
import os
import json
import torch.distributed
from tqdm import tqdm
from pathlib import Path, PosixPath

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import webdataset as wds
from llava.constants import SHARD_SHUFFLE_BUFSIZE, SHARD_SHUFFLE_INITIAL
from llava.constants import SAMPLE_SHUFFLE_BUFSIZE, SAMPLE_SHUFFLE_INITIAL
from llava.constants import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN
from llava.conversation import conv_templates
from llava.model import InternVLChatConfig, InternVLChatModel
from llava.multifile_tariterators import tarfile_to_samples
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
        conv = conv_templates[args.conv_template_name].copy()
    else:
        raise KeyError(f"get a wrong conversation name: {args.conv_template_name}, which does not exist!")
    args.conversation = conv
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


def create_tokenizer_and_model(args: Namespace = None):
    # TODO: Now here
    pass


def create_wds_and_collator(
                            tokenizer: transformers.PreTrainedTokenizer, 
                            args: Namespace = None
                           ):
    pass


@torch.inference_mode()
def recaption(model_name, questions_file, answers_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()

    ques_file = open(os.path.expanduser(questions_file), "r")
    ans_file = open(os.path.expanduser(answers_file), "w")
    for i, line in enumerate(tqdm(ques_file)):
        idx = json.loads(line)["question_id"]
        qs = json.loads(line)["text"]
        cat = json.loads(line)["category"]
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stopping_criteria = KeywordsStoppingCriteria([conv.sep], tokenizer, input_ids)
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            use_cache=True,
            temperature=0.7,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


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
    parser.add_argument("--model-max-length", type=int, default=12288, help="maximum length for tokenizer and model")
    parser.add_argument("--padding", type=str, default="longest", choices=("longest", "max_length", "do_not_pad"), 
                        help="padding strategy for text tokenizer")
    parser.add_argument("--truncation", type=str, default="do_not_truncate", help="truncation strategy for text tokenizer")
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
