# image-alttext pairs randomly sampling script
import os, re, sys
import json
import time
from datetime import datetime
import shutil
import tarfile
import random
import argparse
from argparse import Namespace
from typing import Dict, List
import logging
from pathlib import Path, PosixPath
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL
import multiprocessing
from multiprocessing.managers import DictProxy, ValueProxy
from multiprocessing.synchronize import Barrier
from concurrent import futures as futures
from functools import partial
import math
from braceexpand import braceexpand


cpu_cnts = os.cpu_count()
logger: logging.Logger = None
pid_to_rank: DictProxy = None
proc_lock = None
proc_barrier = None
total_tar_num: int = None
cur_sampled_data_num: ValueProxy = None
all_ranks_to_sampled_tarnames: DictProxy = None

cur_proc_sample_tar_num: int = None
cur_proc_sample_data_num: int = None


def parse_args():
    parser = argparse.ArgumentParser(description="command line arguments parser for image-alttext archive files (tars) parallelise sampling")

    # image-alttext data randomly sampling arguments
    parser.add_argument("--raw-data-dir", type=lambda x: Path(x), default="/home/yidongyi/ImageDataSets/bdbk_citiao_sougou/output", 
                        help="The directory for whole set of Taisu2")
    parser.add_argument("--base-out-dir", type=lambda x: Path(x), default="/mnt/lustre/lizongshu/datasets/Taisu2_datasets", 
                        help="the base image-alttext pairs directory of all Taisu2 datasest experiments")
    parser.add_argument("--total-sample-data-num", type=lambda x: int(eval(x)), default=5e6, help="Total number of raw image-text pairs for Taisu2")
    parser.add_argument("--tar-patterns", type=str, default="{00000..15079}.tar", help="the brace expanding expression of tar files for sampling")
    parser.add_argument("--preserved-file-types", type=str, nargs="+", default=["tar"], 
                        help="file types which should be copied")
    parser.add_argument("--base-random-seed", type=lambda x: eval(x), default=None, help="base random seed for all image-alttext pairs sampling processes")
    parser.add_argument("--sample-num-per-proc", type=lambda x: eval(x), default=None, help="number of tar files for each worker the randomly sample")
    parser.add_argument("--sample-max-workers", type=int, default=40, help="max workers for the process pool to sample image-alltext data randomly")

    # logging arguments
    parser.add_argument("--logging-level", type=lambda x: eval(x), default=DEBUG, choices=[NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL], 
                        help="The output level for logger object of the logging module")
    args = parser.parse_args()

    # output folder & directory settings
    time_str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    sub_out_folder = f"image-alttext-total-{args.total_sample_data_num / 1e6:.2f}M-at-{time_str}"
    args.sub_out_folder = sub_out_folder
    args.sub_out_dir = args.base_out_dir / sub_out_folder
    if args.sub_out_dir.exists():
        shutil.rmtree(args.sub_out_dir)
    else:
        args.sub_out_dir.mkdir(parents=False, exist_ok=False)
    img_alttext_dir: PosixPath = args.sub_out_dir / "image-text-pairs"
    args.img_alttext_dir = img_alttext_dir
    args.img_alttext_dir.mkdir(parents=False, exist_ok=False)

    return args


def init_logger(logging_level: int, out_dir: PosixPath):
    global logger
    logger = logging.getLogger("Taisu2 raw image-alttext data sampling")
    logger.setLevel(logging_level)

    fmt = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")
    stream_hndlr = logging.StreamHandler(); stream_hndlr.setFormatter(fmt); stream_hndlr.setLevel(logging_level)
    logger.addHandler(stream_hndlr)

    # logger output path & FileHandler
    log_path: PosixPath = out_dir / f"image-alttext-pairs-random-sampling.log"; log_path.touch(exist_ok=False)
    file_hndlr = logging.FileHandler(log_path); file_hndlr.setFormatter(fmt); file_hndlr.setLevel(logging_level)
    logger.addHandler(file_hndlr)


def args_check_and_log(args: Namespace = None):
    global logger
    if not args.raw_data_dir.exists():
        logger.error(f"raw image-alttext pairs directory - {args.raw_data_dir} does not exist!")
        sys.exit(1)
    logger.info(f"raw image-alttext data directory for randomly sampling (args.raw_data_dir): {args.raw_data_dir}")
    logger.info(f"basic image-alttext data directory for all Taisu2 experiments (args.base_out_dir): {args.base_out_dir}")
    logger.info(f"total sampling image-alttext pairs number (args.total_sample_data_num): {args.total_sample_data_num}")
    logger.info(f"sub folder stored current Taisu2 image-alttext pairs data (args.sub_out_folder): {args.sub_out_folder}")
    logger.info(f"native tar patterns for expanding (args.tar_patterns): {args.tar_patterns}")

    sample_res_file = "multiproc_sampling_results.jsonl"
    sample_res_p: PosixPath = args.sub_out_dir / sample_res_file
    args.sample_results_p = sample_res_p
    logger.info(f"results file for randomly sampling via multiprocessing pool (args.sample_results_p): {args.sample_results_p}")

    args.preserved_file_types = set(args.preserved_file_types)
    if not len(args.preserved_file_types):
        logger.error(f"no file types are specified in argument `preserved_file_types`: {args.preserved_file_types}")
        sys.exit(1)
    if "tar" not in args.preserved_file_types:
        logger.error(f"tar file must be included in argument `preserved_file_types`, but get {args.preserved_file_types}")
        sys.exit(1)
    remained_preserved_file_types = args.preserved_file_types.difference(["tar", "json", "parquet"])
    if remained_preserved_file_types:
        logger.error(f"there are some file types that should not be specified and copied: {remained_preserved_file_types}")
        sys.exit(1)
    logger.info(f"file types to sample and copy (args.preserved_file_types): {args.preserved_file_types}")

    if args.base_random_seed is None or type(args.base_random_seed) not in (int, float):
        args.base_random_seed = int(time.time())
    else:
        args.base_random_seed = int(args.base_random_seed)
    logger.info(f"basic random seed for image-alttext sampling (args.base_random_seed): {args.base_random_seed}")
    global cpu_cnts
    if args.sample_max_workers >= cpu_cnts // 5 * 4:
        logger.warning(f"the value of argument `args.sample_max_workers` is greater than 4 / 5 of total cpu counts, set it to {cpu_cnts // 5 * 4}")
        args.sample_max_workers = int(cpu_cnts // 5 * 4)
    else:
        logger.info(f"the maximum workers for the randomly sampling process pool (args.sample_max_workers): {args.sample_max_workers}")
    if args.sample_num_per_proc is not None:
        logger.warning(f"argument `args.sample_num_per_proc` will be determined by the total number of tar files and total number of workers in a process pool, "
                       f"hence set it to `None`")
        args.sample_num_per_proc = None

    return


def proc_init_func(base_random_seed: int):
    global pid_to_rank, proc_lock, logger
    global cur_proc_sample_tar_num, cur_proc_sample_data_num

    pid = str(os.getpid())
    with proc_lock:
        rank = str(len(pid_to_rank))
        pid_to_rank[pid] = rank
    os.environ["RANK"] = rank
    random.seed(base_random_seed + int(rank))
    with proc_lock:
        logger.info(f"image-alttext randomly sampling process (pid {pid}) initializing, set process rank to {rank}")

    if cur_proc_sample_tar_num is None or cur_proc_sample_tar_num != 0:
        cur_proc_sample_tar_num = 0
    if cur_proc_sample_data_num is None or cur_proc_sample_data_num != 0:
        cur_proc_sample_data_num = 0
    with proc_lock:
        logger.info(f"initialize the global variable `cur_proc_sample_tar_num` to 0 at process with rank {rank}")
        logger.info(f"initialize the global variable `cur_proc_sample_data_num` to 0 at process with rank {rank}")


def update_sampled_tarnames_dict(proc_rank: str, sampled_tarnames: List):
    global all_ranks_to_sampled_tarnames
    has_rank = False
    for rank_key in all_ranks_to_sampled_tarnames.keys():
        if rank_key == proc_rank:
            has_rank = True
            all_ranks_to_sampled_tarnames[rank_key].extend(sampled_tarnames)
            break
    if not has_rank:
        all_ranks_to_sampled_tarnames[proc_rank] = sampled_tarnames


def random_sample_task_func(tar_names_list: List[str], args: Dict = None, proc_barrier: Barrier = None):
    global logger, proc_lock, cur_sampled_data_num
    global cur_proc_sample_tar_num, cur_proc_sample_data_num

    proc_rank = os.getenv("RANK", None)
    pid = os.getpid()
    if proc_rank is None:
        with proc_lock:
            logger.error(f"cannot get environmental variable `RANK` at process with pid {pid}, exit abnormally")
        sys.exit(1)

    sampled_tarnames = []
    if not tar_names_list:
        with proc_lock:
            update_sampled_tarnames_dict(proc_rank, [])
            logger.info(f"get an empty tar names list at process with rank {proc_rank}, hence return directly")
        return
    else:
        # duplicated tar names / files are not allowed
        if len(tar_names_list) != len(set(tar_names_list)):
            with proc_lock:
                logger.error(f"there're duplicated tar names/files in the tar names list dispending to process with rank {proc_rank}, exit abnormally")
            sys.exit(1)
        with proc_lock:
            logger.info(f"process with rank {proc_rank} gets a tar names list with length {len(tar_names_list)} for randomly sampling")

        while True:
            if cur_sampled_data_num.value < args["total_sample_data_num"] and tar_names_list:
                cur_proc_sample_tar_num += 1
                cur_tar_name = random.choice(tar_names_list)
                tar_names_list.remove(cur_tar_name)
                tar_file_p: PosixPath = args["raw_data_dir"] / cur_tar_name
                parquet_file_p = tar_file_p.with_suffix(".parquet")
                json_file_p = tar_file_p.with_name(f"{tar_file_p.stem}_stats.json")
                if not parquet_file_p.exists() or not json_file_p.exists():
                    logger.warning(f"the json or parquet files of current randomly sampled tar file ({cur_tar_name}) does not exist, "
                                   f"hence skip this tar file and continue next one")
                    continue
                if not tar_file_p.exists():
                    with proc_lock:
                        logger.error(f"tar file with name - {cur_tar_name} does not exist at process with rank {proc_rank}, exit abnormally")
                    sys.exit(1)

                for file_suffix in args["preserved_file_types"]:
                    if file_suffix == "json":
                        cur_file_p = tar_file_p.with_name(f"{tar_file_p.stem}_stats.json")
                    else:  # tar files & parquet files
                        cur_file_p = tar_file_p.with_suffix(f".{file_suffix}")
                    dst_p: PosixPath = args["img_alttext_dir"] / f"{cur_file_p.name}"
                    try:
                        _ = shutil.copy(cur_file_p, dst_p)
                    except Exception as cp_error:
                        with proc_lock:
                            logger.error(f"when copy the randomly sampling file {cur_file_p.name} from {args['raw_data_dir']} to {args['img_alttext_dir']}, "
                                         f"encounter error ({type(cp_error)}): {cp_error}")
                        sys.exit(1)

                with tarfile.open(tar_file_p, "r", encoding="utf-8") as cur_tar_fp:
                    cur_tar_file_names = cur_tar_fp.getnames()
                cur_tar_img_names = [file_name for file_name in cur_tar_file_names if file_name.endswith(".jpg")]
                cur_tar_txt_names = [file_name for file_name in cur_tar_file_names if file_name.endswith(".txt")]
                cur_tar_json_names = [file_name for file_name in cur_tar_file_names if file_name.endswith(".json")]
                if len(set([len(cur_tar_img_names), len(cur_tar_txt_names), len(cur_tar_json_names)])) != 1:
                    with proc_lock:
                        logger.error(f"the number of images, txt files, and json files in current tar file `{cur_tar_name}` aren't equal, "
                                     f"get {len(cur_tar_img_names)}, {len(cur_tar_txt_names)}, and {len(cur_tar_json_names)} for image, txt, and json")
                    sys.exit(1)

                with proc_lock:
                    cur_sampled_data_num.value += len(cur_tar_img_names)
                sampled_tarnames.append(cur_tar_name)
                cur_proc_sample_data_num += len(cur_tar_img_names)
                with proc_lock:
                    logger.info(f"process with rank {proc_rank} has randomly sampled a tar file named {cur_tar_name}, "
                                f"this process has sampled {cur_proc_sample_tar_num} tar files, {cur_proc_sample_data_num} image-alttext pairs in total")
            else:
                if cur_sampled_data_num.value >= args["total_sample_data_num"]:  # sampled image-alttext pairs number is satisfied
                    with proc_lock:
                        logger.info(f"has randomlly sampled {cur_sampled_data_num.value} image-alttext pairs in total, "
                                    f"process with rank {proc_rank} exit normally")
                else:  # tar files (tar names list) for current process is exhausted
                    with proc_lock:
                        logger.info(f"the tar names in the list dispensed for process with rank {proc_rank} has been exhausted exit normally")
                with proc_lock:
                    logger.info(f"process with rank {proc_rank} has finished randomly sampling")
                break
        proc_barrier.wait()
        with proc_lock:
            update_sampled_tarnames_dict(proc_rank, sampled_tarnames)
            logger.info(f"having sampled tar files number - {cur_proc_sample_tar_num} at process with rank {proc_rank} cumulatively")
            logger.info(f"having sampled image-alttext data number - {cur_proc_sample_data_num} at process with rank {proc_rank} cumulatively")
        proc_barrier.wait()
        return


def main(args: Namespace = None):
    global logger
    global total_tar_num
    all_tar_names = [tar_p.name for tar_p in args.raw_data_dir.glob("*.tar")]
    total_tar_num = len(all_tar_names)
    args.sample_num_per_proc = math.ceil(total_tar_num / args.sample_max_workers)

    def tar_names_list_generator(tar_names_list: List, native_tar_patterns: str, sample_num_per_proc: int):
        pat_str = r"\{\d+\.\.\d+\}\.tar"
        tar_patterns_iter = re.finditer(pat_str, native_tar_patterns)
        ret_tar_names_list = []
        while True:
            try:
                tar_pattern: re.Match = next(tar_patterns_iter)
                try:
                    cur_tar_files_iter = braceexpand(tar_pattern.group(0))
                except Exception as e:
                    logger.error(f"when using `braceexpand` function on a tar pattern `{tar_pattern.group(0)}`, encounter an error ({type(e)}): {e}")
                    sys.exit(1)
                for cur_tar_name in cur_tar_files_iter:
                    if cur_tar_name not in tar_names_list:
                        logger.error(f"a tar file named {cur_tar_name} generated from tar pattern string `{tar_pattern.group(0)}` "
                                     f"does not exist in the raw image-alttext data directory")
                        sys.exit(1)
                    ret_tar_names_list.append(cur_tar_name)
                    if len(ret_tar_names_list) == sample_num_per_proc:
                        yield ret_tar_names_list
                        ret_tar_names_list = []
            except StopIteration as _:
                break
        if ret_tar_names_list:
            yield ret_tar_names_list
        return

    # parallel proceesses in the process pool to sample imagel-alttext pair tar files
    with multiprocessing.Manager() as mp_manager:
        global pid_to_rank, proc_lock, proc_barrier
        global cur_sampled_data_num, all_ranks_to_sampled_tarnames
        pid_to_rank = mp_manager.dict()
        proc_lock = mp_manager.Lock()
        proc_barrier = mp_manager.Barrier(args.sample_max_workers)
        cur_sampled_data_num = mp_manager.Value("i", 0)
        all_ranks_to_sampled_tarnames = mp_manager.dict()

        random_sample_func = partial(random_sample_task_func, args=vars(args), proc_barrier=proc_barrier)
        args.sample_num_per_proc = math.ceil(total_tar_num  / args.sample_max_workers)
        logger.info(f"maximum total tar numbers for each worker to randomly sample (args.sample_num_per_proc): {args.sample_num_per_proc}")

        with futures.ProcessPoolExecutor(max_workers=args.sample_max_workers, initializer=proc_init_func, 
                                         initargs=(args.base_random_seed, )) as sample_proc_pool:
            _ = sample_proc_pool.map(random_sample_func, tar_names_list_generator(all_tar_names, args.tar_patterns, args.sample_num_per_proc), chunksize=1)

        with open(args.sample_results_p, mode="a", encoding="utf-8") as sample_res_fp:
            for rank_key, rank_sampled_tarnames in all_ranks_to_sampled_tarnames.items():
                rank_sampled_dict = {rank_key: rank_sampled_tarnames}
                json.dump(rank_sampled_dict, sample_res_fp, ensure_ascii=False)
                sample_res_fp.write("\n")
        logger.info(f"write the randomly sampling results of process pool into file - {args.sample_results_p}")

    # save command line arguments
    args_save_p = args.sub_out_dir / "arguments.json"
    args_for_saving = vars(args)
    for arg_name, arg_val in args_for_saving.items():
        if isinstance(arg_val, PosixPath):
            args_for_saving[arg_name] = str(arg_val)
        if isinstance(arg_val, set):
            args_for_saving[arg_name] = list(arg_val)
    with open(args_save_p, mode="w", encoding="utf-8") as args_save_fp:
        json.dump(args_for_saving, args_save_fp, ensure_ascii=False)
    logger.info(f"save command line arguments into path: {args_save_p}")


if __name__ == "__main__":
    args = parse_args()
    init_logger(args.logging_level, args.sub_out_dir)
    args_check_and_log(args)
    main(args)

