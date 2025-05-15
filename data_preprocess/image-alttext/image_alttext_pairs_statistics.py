# The second step for image-text pairs preprocessing -- summarizing the statistics of sampled image-text pairs
import re
import os
from io import BytesIO
import json
import shutil
import logging
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
import tarfile
import argparse
import multiprocessing
import subprocess
from concurrent import futures as futures
from datetime import datetime
from functools import partial
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer


logger = None
proc_lock_for_all_proc = None
sub_proc_tokenizer = None


def parse_args():
    image_statistic_types = ["image_shorter_edge", "image_longer_edge", "image_aspect_ratio", "image_num_pixels"]
    text_statistic_types = ["text_num_words", "text_num_tokens"]
    parser = argparse.ArgumentParser(description="Summarize the statistics of sampled iamge-text pairs from the raw ones")
    parser.add_argument("--base-taisu2-dir", type=lambda x: Path(x), default="/mnt/lustre/lizongshu/datasets/Taisu2_datasets", 
                        help="The basic directory of Taisu2 dataset")
    parser.add_argument("--specific-data-folder", type=str, default="init-image-text-pairs-total-100.00M-at-2025-01-21-13:09:29", 
                        help="The specific image-text pairs folder after randomly sampling for Taisu2")
    parser.add_argument("--image-statistic-types", type=str, nargs="+", default=image_statistic_types, 
                        help="statistic analyzation type for raw image")
    parser.add_argument("--text-statistic-types", type=str, nargs="+", default=text_statistic_types, 
                        help="statistic analyzation type for raw alt-text")
    parser.add_argument("--tokenizer-repo-id", type=str, default="Qwen/Qwen2-VL-7B-Instruct",  
                        help="the repository ID of text tokenizer, either remote or local")
    parser.add_argument("--max-workers-for-tars", type=int, default=8, help="The `max_workers` parameter for tar files reading")
    parser.add_argument("--max-workers-for-data", type=int, default=2, help="The `max_workers` parameter for image-text pairs reading")
    parser.add_argument("--logging-level", type=lambda x: eval(x), default=DEBUG, choices=[INFO, DEBUG, WARNING, ERROR, FATAL], 
                        help="The logging level for both file and stream handlers of the logging module")
    args = parser.parse_args()

    # time stamp & corresponding string settings
    cur_timestamp = datetime.now(); timestamp_str = cur_timestamp.strftime("%Y-%m-%d-%H:%M:%S")
    args.time_stamp = timestamp_str

    # output directory settings
    if not args.base_taisu2_dir.exists():
        raise ValueError(f"The basic data directory of Taisu2 does not exist: {args.base_taisu2_dir}!")
    if not (args.base_taisu2_dir / args.specific_data_folder).exists():
        raise ValueError(f"The specific folder which is to be analysis statistically does not exist: {args.specific_data_folder}!")
    args.specific_data_dir = args.base_taisu2_dir / args.specific_data_folder
    result_dir = args.specific_data_dir / f"statistics-at-{timestamp_str}"
    args.result_dir = result_dir
    if result_dir.exists():
        for prev_res_ele in result_dir.iterdir():
            if prev_res_ele.isdir():
                shutil.rmtree(prev_res_ele)
            else:
                prev_res_ele.unlink(missing_ok=False)
    result_dir.mkdir(parents=False, exist_ok=True)  # param `parents` default to False; param `exist_ok` default to True

    return args


def init_logger(args):
    global logger
    logger = logging.getLogger(f"image-text pairs statistics"); logger.setLevel(args.logging_level)
    fmt = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)-10s] --- %(message)s")

    stream_handler = logging.StreamHandler(); logger.addHandler(stream_handler)
    output_log_file = "statistics_output.log"; output_log_p = args.result_dir / output_log_file
    args.output_log_p = output_log_p
    if args.output_log_p.exists():
        args.output_log_p.unlink()
    file_handler = logging.FileHandler(output_log_p); logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(args.logging_level); handler.setFormatter(fmt)


def single_proc_for_data(img_txt_data_tuple, args={}):
    tar_file_name, member_name, img_txt_bytes_dict = img_txt_data_tuple
    global logger, proc_lock_for_all_proc
    return_dict = {}

    img_read_error_flag = False; txt_read_error_flag = False
    for member_type, member_bytes in img_txt_bytes_dict.items():
        if member_type == "image":
            try:
                img = Image.open(BytesIO(member_bytes))
            except Exception as e:
                with proc_lock_for_all_proc:
                    logger.info(f"When reading image {member_name}.jpg from archive file {tar_file_name}, encounter error: {e}")
                img_read_error_flag = True; break

            img_h, img_w = img.height, img.width
            if "image_shorter_edge" in args["image_statistic_types"]:
                return_dict["image_shorter_edge"] = min(img.size)
            if "image_longer_edge" in args["image_statistic_types"]:
                return_dict["image_longer_edge"] = max(img.size)
            if "image_aspect_ratio" in args["image_statistic_types"]:
                return_dict["image_aspect_ratio"] = round(img_w / img_h, 4)
            if "image_num_pixels" in args["image_statistic_types"]:
                return_dict["image_num_pixels"] = img.size[0] * img.size[1]
        elif member_type == "text":
            try:
                alt_text = member_bytes.decode("utf-8")
                if "text_num_words" in args["text_statistic_types"]:
                    punc_and_space_pat = r"[，。！？“”‘’：；、【】《》\s,.!?\"':;[\]<>]"
                    return_dict["text_num_words"] = len(alt_text.strip()) - len(re.findall(punc_and_space_pat, alt_text))
                if os.getenv("TOKENIZE", None) is not None and bool(int(os.getenv("TOKENIZE", None))):
                    global sub_proc_tokenizer
                    return_dict["text_num_tokens"] = len(sub_proc_tokenizer(alt_text, add_special_tokens=False).input_ids)
            except Exception as e:
                with proc_lock_for_all_proc:
                    logger.info(f"When reading alt-text file {member_name}.txt from archive file {tar_file_name}, encounter error: {e}")
                txt_read_error_flag = True; break

    if img_read_error_flag or txt_read_error_flag:
        tar_file_path = str(args["specific_data_dir"] / "image-text-pairs" / tar_file_name)
        delete_cmd = f"tar --delete -f {tar_file_path} {member_name}"
        for suffix in (".jpg", ".txt", ".json"):
            try:  # delete the file with specific suffix(.jpg, .txt, or .json) in the archive file with `.tar` suffix
                del_cmd = delete_cmd + suffix
                _ = subprocess.run(del_cmd, shell=True, text=True, capture_output=True, check=True)
                with proc_lock_for_all_proc:
                    logger.info(f"Delete the erroneously read file {member_name}{suffix} from {tar_file_name} successfully")
            except subprocess.CalledProcessError as e:
                with proc_lock_for_all_proc:
                    logger.info(f"When deleting file {member_name}{suffix} that failed to read from an archive file, "
                                f"encounter the following error: {e}")
        return None, "", {}

    # (1) name (with suffix) of a `tar` archive file (When encoutering error, return `None`)
    # (2) prefix name of image_alt-text pairs (When encoutering error, return `""`)
    # (3) dictionary stored statistics of image_alt-text pairs (When encoutering error, return `{}`)
    return tar_file_name, member_name, return_dict


def proc_pool_init_for_data(pid_to_subrank_dict, lock_for_data_proc_pool_init, tokenizer_repo_id, alttext_tokenize):
    with lock_for_data_proc_pool_init:
        sub_pid = os.getpid(); sub_rank = str(len(pid_to_subrank_dict))
        os.environ["SUBRANK"] = sub_rank; pid_to_subrank_dict[sub_pid] = sub_rank
    rank = os.environ.get("RANK", None)
    if rank is None:
        raise RuntimeError(f"Process-{sub_pid} cannot get the rank index of its parent process, "
                           f"please check and debug codes")

    os.environ["TOKENIZE"] = str(1) if alttext_tokenize else str(0)
    global sub_proc_tokenizer
    if alttext_tokenize:
        sub_proc_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo_id)


def single_proc_for_tar(tar_files_for_cur_proc, args={}):
    sub_mp_manager = multiprocessing.Manager()
    subrank_dict_for_data = sub_mp_manager.dict()
    lock_for_data_proc_pool_init = sub_mp_manager.Lock()
    data_initargs = (subrank_dict_for_data, lock_for_data_proc_pool_init, args["tokenizer_repo_id"], 
                     args["alttext_tokenize"])

    global logger, proc_lock_for_all_proc

    def generator_for_data_in_tar():
        tar_file_obj = tarfile.open(tar_file_p, "r"); 
        tar_file_name = os.path.basename(tar_file_obj.name)

        while True:
            tar_sub_members = (tar_file_obj.next(), tar_file_obj.next(), tar_file_obj.next())
            if None not in tar_sub_members:
                tar_sub_member_names = [sub_member.name.split(".")[0] for sub_member in tar_sub_members]
                if len(set(tar_sub_member_names)) != 1:
                    raise ValueError(f"The base name of sub tar members should be identical, but got {tar_sub_member_names}")
                else:
                    tar_sub_member_name = tar_sub_member_names[0]
                img_txt_bytes_dict = {"image": None, "text": None}
                for tar_sub_member in tar_sub_members:
                    if tar_sub_member.name.endswith(".jpg"):
                        img_txt_bytes_dict["image"] = tar_file_obj.extractfile(tar_sub_member).read()
                    elif tar_sub_member.name.endswith(".txt"):
                        img_txt_bytes_dict["text"] = tar_file_obj.extractfile(tar_sub_member).read()
                yield tar_file_name, tar_sub_member_name, img_txt_bytes_dict
            else:
                break

        tar_file_obj.close()
        return None

    def img_txt_result_filter(img_txt_result):
        if img_txt_result[0] is None or img_txt_result[1] == "" or img_txt_result[2] == {}:
            return False
        else:
            return True

    def statistic_post_process():
        cur_tar_result_file = args["result_dir"] / (tar_file_p.stem + ".jsonl")
        with open(cur_tar_result_file, "w", encoding="utf-8") as cur_tar_result_fp:
            for data_res in filtered_data_results:
                json.dump(data_res, cur_tar_result_fp, ensure_ascii=False)
                cur_tar_result_fp.write("\n")

        rank = os.getenv("RANK", "None")
        with proc_lock_for_all_proc:
            logger.info(f"Statistics of image-text tarfile {tar_file_p.stem} has been completed at rank {rank}")

        return cur_tar_result_file

    chunksize_for_data = max(args["max_workers_for_data"], 16)
    tar_files_iter_for_cur_proc = iter(tar_files_for_cur_proc)
    partial_data_func = partial(single_proc_for_data, args=args)
    returned_list = []  # returned variable
    with futures.ProcessPoolExecutor(max_workers=args["max_workers_for_data"], initializer=proc_pool_init_for_data, 
                                     initargs=data_initargs) as data_executor:
        while True:
            try:
                tar_file_p = next(tar_files_iter_for_cur_proc)
            except StopIteration as _:
                break
            data_results = data_executor.map(partial_data_func, generator_for_data_in_tar(), chunksize=chunksize_for_data)
            filtered_data_results = filter(img_txt_result_filter, data_results)
            res_file_p = statistic_post_process()
            returned_list.append(res_file_p)

    sub_mp_manager.shutdown()
    return returned_list


def proc_pool_init_for_tar(pid_to_rank_dict):
    global proc_lock_for_all_proc
    with proc_lock_for_all_proc:
        pid = str(os.getpid()); proc_rank = str(len(pid_to_rank_dict))
        pid_to_rank_dict[pid] = proc_rank; os.environ["RANK"] = proc_rank


def parallel_statistic_analyzation(args):
    tar_files_list = list((args.specific_data_dir / "image-text-pairs").glob("*.tar"))
    total_num_tar_files = len(tar_files_list)
    if args.max_workers_for_tars > total_num_tar_files:
        logger.warning(f"Number of parallel processes for tar reading is {args.max_workers_for_tars}, "
                       f"which is greater than the total number of `*.tar` files - {total_num_tar_files}. "
                       f"Hence set the parallel processes number for tar reading equal to the total number of `*.tar` files")
        args.max_workers_for_tars = total_num_tar_files

    # split `*.tar` files for parallel process pool
    if total_num_tar_files % args.max_workers_for_tars == 0:
        split_num_tar_files = total_num_tar_files // args.max_workers_for_tars
    else:
        split_num_tar_files = total_num_tar_files // args.max_workers_for_tars + 1
    split_tar_files_list = []
    for idx in range(args.max_workers_for_tars):
        start_idx = idx * split_num_tar_files
        if idx < args.max_workers_for_tars - 1:
            end_idx = start_idx + split_num_tar_files
        else:  # the last tar files list for process in the process pool
            end_idx = total_num_tar_files
        try:
            split_tar_files_list.append(tar_files_list[start_idx: end_idx])
        # for the situation --- number of tar files is smaller than `max_workers_for_tars` argument
        except IndexError as _:
            break

    with multiprocessing.Manager() as main_mp_manager:
        rank_dict_for_tar = main_mp_manager.dict()
        global proc_lock_for_all_proc; proc_lock_for_all_proc = main_mp_manager.Lock()
        alttext_tokenize = True if "text_num_tokens" in args.text_statistic_types else False
        args.alttext_tokenize = alttext_tokenize

        partial_func_for_tar = partial(single_proc_for_tar, args=vars(args))
        tar_initargs = (rank_dict_for_tar, )

        with futures.ProcessPoolExecutor(max_workers=args.max_workers_for_tars, initializer=proc_pool_init_for_tar, 
                                         initargs=tar_initargs) as tar_executor:
            _ = tar_executor.map(partial_func_for_tar, split_tar_files_list)


def main():
    args = parse_args(); init_logger(args)

    global logger
    logger.info(f"The specific Taisu2 image-text pairs folder is {args.specific_data_folder}")
    logger.info(f"The output statistically sub-folder for the analyzation of Taisu2 image-text pairs is {args.result_dir.name}")

    # Check the attribution (remote or local) of the huggingface tokenizer repository id argument
    repo_id_pat = r"^[^\s/]+/[^/\s]+$"
    if re.match(repo_id_pat, args.tokenizer_repo_id):
        logger.warning(f"Get a remote huggingface repository id: {args.tokenizer_repo_id}, which has potential network problems." 
                       f"We suggest to change to local disk repository")
    else:
        args.tokenizer_repo_id = Path(args.tokenizer_repo_id); assert args.tokenizer_repo_id.exists()
    if type(args.tokenizer_repo_id) is str:
        tokenizer_name = args.tokenizer_repo_id.split("/")[-1]
    else:
        tokenizer_name = args.tokenizer_repo_id.name
    logger.info(f"Using tokenizer of `{tokenizer_name}` to tokenize the alt-texts")

    parallel_statistic_analyzation(args)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
