import io, os, re
import random
from typing import Union, List, Tuple, Dict, Set
from functools import partial
from pathlib import Path, PosixPath
import llava.conversation as conversation_lib
from llava.conversation import default_conversation
from llava.constants import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN
from llava.constants import IGNORE_INDEX
from llava.constants import IMAGENET_MEAN, IMAGENET_STD
from PIL import Image
from PIL.ImageFile import ImageFile
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import transformers


TASKS_TYPE = (
              "caption", "visual_grounding", "ocr", 
              "visual_reasoning", "vqa", "multi_image", "text"
             )


__all__ = ["taisu2_img_preprocess", "load_image", "taisu2_text_preprocess", "taisu2_wds_map"]


# All functions below are image preprocess
def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    return transform


def find_closest_aspect_ratio(
                              aspect_ratio: float, 
                              target_ratios: Set[Tuple[int, int]], 
                              width: int, 
                              height: int, 
                              image_size: int
                             ) -> Tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
                       image: Union[ImageFile | Image.Image], 
                       min_num: int = 1, 
                       max_num: int = 12, 
                       image_size=448, 
                       use_thumbnail: bool = False
                      ) -> List[Image.Image | ImageFile]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(
               image_file: Union[str | PosixPath], 
               input_size: int = 448, 
               min_num: int = 1, 
               max_num: int = 12, 
               use_thumbnail: bool = True
              ) -> torch.Tensor:
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, min_num=min_num, max_num=max_num, image_size=input_size, use_thumbnail=use_thumbnail)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values


def taisu2_img_preprocess(
                          pil_img: Union[ImageFile | Image.Image], 
                          input_size: int = 448, 
                          min_num: int = 1, 
                          max_num: int = 12, 
                          use_thumbnail: bool = True
                         ) -> Tuple[torch.Tensor, int]:
    transform = build_transform(input_size=input_size)
    pil_images = dynamic_preprocess(pil_img, min_num=min_num, max_num=max_num, image_size=input_size, use_thumbnail=use_thumbnail)
    pixel_values = [transform(pil_image) for pil_image in pil_images]
    pixel_values = torch.stack(pixel_values)

    return dict(pixel_values=pixel_values, sub_img_num=len(pil_images))


# All functions below are text preprocess
def taisu2_preprocess_internvl2_5(
                                  anns: str, 
                                  data_stem_name: str, 
                                  task_type: str, 
                                  context_token_per_img: int, 
                                  sub_img_num: Union[int | List[int] | Tuple[int]], 
                                  tokenizer: transformers.PreTrainedTokenizer, 
                                  padding: Union[str | bool] = "do_not_pad", 
                                  padding_side: Union[str | None] = "right", 
                                  return_tensors: Union[str | None] = "pt", 
                                  return_attention_mask: Union[bool | None] = True, 
                                  is_train: bool = True, 
                                  inference_recaption: bool = False, 
                                 ) -> Dict[str, torch.Tensor]:
    if task_type.lower() not in TASKS_TYPE:
        raise ValueError(f"task for Taisu2 preprocessing function could only be `{TASKS_TYPE}`, but get {task_type}")
    if task_type.lower() == "caption":
        prompt_file = "caption_prompt.txt"
    elif task_type.lower() == "visual_grounding":
        prompt_file = "grounding_prompt.txt"
    elif task_type.lower() == "ocr":
        prompt_file = "ocr_prompt.txt"
    elif task_type.lower() == "visual_reasoning":
        prompt_file = "reasoning_prompt.txt"
    elif task_type.lower() == "multi_image":
        prompt_file = "multi_image_prompt.txt"
    elif task_type.lower() == "text":
        prompt_file = None
    prompt_dir = Path(os.path.abspath(__file__)).parent.parent / "prompts"
    prompt_p = prompt_dir / prompt_file if prompt_file is not None else None
    if prompt_p is not None:
        if not prompt_p.exists():
            raise FileNotFoundError(f"prompt file for task {task_type.lower()} - {prompt_p}, doesn't exist!")
        all_prompts = []
        with open(prompt_p, mode="r", encoding="utf-8") as prompt_fp:
            for prompt in prompt_fp:
                all_prompts.append(prompt.strip())

    conv = default_conversation.copy()
    conv.messages = []
    if conv.sep_style != conversation_lib.SeparatorStyle.MPT:
        raise ValueError(f"the separator style of InternVL2_5/InternVL3 must be SeparatorStyle.MPT")
    roles = conv.roles
    roles_map = {"user": roles[0], "gpt": roles[1]}

    if task_type.lower() == "caption":
        if all_prompts[0] != "native_prompts:":
            raise ValueError(f"first line of caption prompt file shoud be `native_prompts:`")
        if "recaption_prompts:" not in all_prompts:
            raise ValueError(f"`recaption_prompts:` should be in caption prompt file")
        if "native_and_recaption_prompts:" not in all_prompts:
            raise ValueError(f"`native_and_recaption_prompts:` should be in caption prompt file")
        native_caption_prompts = all_prompts[1: all_prompts.index("recaption_prompts:")]
        re_caption_prompts = all_prompts[all_prompts.index("recaption_prompts:") + 1: all_prompts.index("native_and_recaption_prompts:")]
        two_caption_prompts = all_prompts[all_prompts.index("native_and_recaption_prompts:") + 1: ]

        pat_strs = (
                    r"(<\|native_caption_start\|>)([\s\S]*)(<\|native_caption_end\|>)", 
                    r"(<\|recaption_start\|>)([\s\S]*)(<\|recaption_end\|>)", 
                    r"(<\|native_caption_start\|>)([\s\S]*)(<\|native_caption_end\|>)(<\|recaption_start\|>)([\s\S]*)(<\|recaption_end\|>)"
                   )
        native_caption = None
        re_caption = None
        match_res = re.fullmatch(pat_strs[2], anns)
        if match_res is not None:
            native_caption = match_res.group(2)
            re_caption = match_res.group(5)
        else:
            match_res = re.fullmatch(pat_strs[1], anns)
            if match_res is not None:
                re_caption = match_res.group(2)
            else:
                match_res = re.fullmatch(pat_strs[0], anns)
                if match_res is not None:
                    native_caption = match_res.group(2)
        if native_caption is None and re_caption is None:
            native_caption = anns
        if (not native_caption) and re_caption is None:
            raise ValueError(f"image-alttext pairs with stem name {data_stem_name} does not have an effective native caption and re-caption")
        if inference_recaption:
            user_prompt = random.choice(re_caption_prompts)
            conv.append_message(roles[0], "<image>\n" + user_prompt)
            conv.append_message(roles[1], None)
        else:
            if native_caption and re_caption:
                prompt_str = random.choice(two_caption_prompts)
                user_prompt = prompt_str.format(native_caption=native_caption)
                conv.append_message(roles[0], "<image>\n" + user_prompt)
                if is_train:
                    conv.append_message(roles[1], re_caption)
                else:
                    conv.append_message(roles[1], None)
            if native_caption and (not re_caption):
                user_prompt = random.choice(native_caption_prompts)
                conv.append_message(roles[0], "<image>\n" + user_prompt)
                if is_train:
                    conv.append_message(roles[1], native_caption)
                else:
                    conv.append_message(roles[1], None)
            if (not native_caption) and re_caption:
                user_prompt = random.choice(re_caption_prompts)
                conv.append_message(roles[0], "<image>\n" + user_prompt)
                if is_train:
                    conv.append_message(roles[1], re_caption)
                else:
                    conv.append_message(roles[1], None)
        conv_prompt = conv.get_prompt()

    if isinstance(sub_img_num, int):
        conv_prompt = conv_prompt.replace("<image>", IMG_START_TOKEN + sub_img_num * context_token_per_img * IMG_CONTEXT_TOKEN + IMG_END_TOKEN, 1)
    elif isinstance(sub_img_num, (list, tuple)):
        for sub_img_num_per in sub_img_num:
            conv_prompt = conv_prompt.replace("<image>", IMG_START_TOKEN + sub_img_num_per * context_token_per_img * IMG_CONTEXT_TOKEN + IMG_END_TOKEN, 1)
    remained_img_token_num = conv_prompt.count("<image>")
    if remained_img_token_num:
        raise ValueError(f"after replacing all `<image>` into image context tokens, there're still `<image>` left: \n{conv_prompt}")
    tokenized_res = tokenizer(
                              conv_prompt, 
                              padding=padding, 
                              padding_side=padding_side, 
                              return_tensors=return_tensors, 
                              return_attention_mask=return_attention_mask
                             )
    if return_tensors == "pt":
        input_ids = tokenized_res.input_ids[0]
    elif return_tensors is None:
        input_ids = torch.tensor(tokenized_res.input_ids)
    else:
        raise TypeError(f"got an unspported `return_tensors` param: {return_tensors}")

    if is_train:
        targets = input_ids.clone()
        # Mask targets
        sep = conv.sep + roles[1]
        total_len = targets.ne(tokenizer.pad_token_id).sum().item()
        rounds = conv_prompt.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[: 3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx: conv_idx + 2]))
        cur_len = 0
        targets[: cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou, padding=False, padding_side=None, return_tensors=None, return_attention_mask=False).input_ids)
            round_len += len(tokenizer(conv.sep, padding=False, padding_side=None, return_tensors=None, return_attention_mask=False).input_ids)
            instruction_len = len(tokenizer(parts[0], padding=False, padding_side=None, return_tensors=None, return_attention_mask=False).input_ids)
            targets[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        targets[cur_len: ] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                targets[:] = IGNORE_INDEX
                print(
                      f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                      f" (ignored)"
                     )

        return {
                "input_ids": input_ids, 
                "labels": targets
               }

    return {"input_ids": input_ids}


def taisu2_text_preprocess(
                           sources: Union[str | List[str]], 
                           data_stem_name: str, 
                           task_type: str, 
                           sub_img_num: Union[int | List[int]], 
                           is_train: bool = True, 
                           inference_recaption: bool = False, 
                           tokenizer: transformers.PreTrainedTokenizer = None, 
                           data_args = None
                          ) -> Dict[str, torch.Tensor]:
    if isinstance(sources, list):
        sources = data_args.txts_separator.join(sources)
    if "internvl2_5" in conversation_lib.default_conversation.name or "internvl3" in conversation_lib.default_conversation.name:
        return taisu2_preprocess_internvl2_5(
                                             data_stem_name=data_stem_name, 
                                             anns=sources, 
                                             task_type=task_type, 
                                             context_token_per_img=data_args.context_token_per_img, 
                                             sub_img_num=sub_img_num, 
                                             tokenizer=tokenizer, 
                                             padding=data_args.padding, 
                                             padding_side=data_args.padding_side, 
                                             return_tensors=data_args.return_tensors, 
                                             return_attention_mask=data_args.return_attention_mask, 
                                             is_train=is_train, 
                                             inference_recaption=inference_recaption, 
                                            )


def taisu2_wds_map(
                   wds_sample: Dict[str, str | bytes | Dict[str, bytes]], 
                   is_train: bool = True, 
                   inference_recaption: bool = False, 
                   tokenizer: transformers.PreTrainedTokenizer = None, 
                   data_args = None
                  ):
    no_img = "jpg" not in wds_sample and "jpeg" not in wds_sample and "png" not in wds_sample
    no_txt = "txt" not in wds_sample; no_json = "json" not in wds_sample
    if no_img or no_txt:
        raise KeyError(f"each sample dictionary must have image and text keys")
    if "image-alttext" in wds_sample["__url__"]:
        task_type = "caption"
    if not (isinstance(wds_sample["jpg"], (bytes, dict)) and isinstance(wds_sample["txt"], (bytes, dict))):
        raise TypeError(f"image/text gotten from webdataset could only be bytes or dictionary, "
                        f"but recieved image: {type(wds_sample["jpg"])}, "
                        f"and text: {type(wds_sample["txt"])}")
    wds_img_map = partial(
                          taisu2_img_preprocess, 
                          input_size=data_args.base_img_size, 
                          min_num=data_args.min_subimg_num, 
                          max_num=data_args.max_subimg_num, 
                          use_thumbnail=data_args.use_thumbnail, 
                         )
    if isinstance(wds_sample["jpg"], bytes):
        pil_img = Image.open(io.BytesIO(wds_sample["jpg"])).convert("RGB")
        img_map_res = wds_img_map(pil_img)
        pixel_values = img_map_res["pixel_values"]
        sub_img_num = img_map_res["sub_img_num"]
    else:
        imgs_list = []
        sub_img_num = []
        for img_idx in sorted(wds_sample["jpg"].keys()):
            img_bytes = wds_sample["jpg"][img_idx]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_map_res = wds_img_map(pil_img)
            imgs_list.append(img_map_res["pixel_values"]); sub_img_num.append(img_map_res["sub_img_num"])
        pixel_values = torch.cat(imgs_list, dim=0)
    data_stem_name = wds_sample["__key__"]

    wds_text_map = partial(
                           taisu2_text_preprocess, 
                           is_train=is_train, 
                           inference_recaption=inference_recaption, 
                           tokenizer=tokenizer, 
                           data_args=data_args
                          )
    if isinstance(wds_sample["txt"], bytes):
        src_txt = wds_sample["txt"].decode(encoding="utf-8")
    else:
        src_txt = []
        for txt_idx in sorted(wds_sample["txt"].keys()):
            txt_bytes = wds_sample["txt"][txt_idx]
            src_txt.append(txt_bytes.decode(encoding="utf-8"))
    text_map_res = wds_text_map(src_txt, data_stem_name, task_type, sub_img_num)
    input_ids = text_map_res["input_ids"]

    if is_train:
        labels = text_map_res["labels"]
        return dict(
                    input_ids=input_ids, 
                    pixel_values=pixel_values, 
                    labels=labels
                   )

    # Inference/Evaluation
    return dict(
                input_ids=input_ids, 
                pixel_values=pixel_values, 
                data_name=data_stem_name
               )
