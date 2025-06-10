# archiving txt files into archive tars
import io, os, re
import sys, json
import math
import shutil
import tarfile
from tarfile import TarFile, TarInfo
from datetime import datetime
from typing import Dict, List
import argparse
from argparse import Namespace, ArgumentError
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL
from functools import partial
from concurrent import futures as futures
import multiprocessing
from multiprocessing.managers import AcquirerProxy, DictProxy
from multiprocessing.sharedctypes import Synchronized
from pathlib import PosixPath, Path


logger: logging.Logger = None
shared_data_num: Synchronized = None
recaption_dict: DictProxy = None
shared_lock: AcquirerProxy = None


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="archive txt files with new annotations into tar files")
    parser.add_argument("--base-data-folder", type=str, default="image-alttext-total-1.00M-at-2025-04-16-21:10:39", help="basic folder of Taisu2 dataset")
    parser.add_argument("--specific-data-folder", type=str, default="rename_and_rearchive", help="specific folder of Taisu2 dataset")
    parser.add_argument("--recaption-idx", type=int, default=None, help="recaption interger index of Taisu2 image-alttext pairs")
    parser.add_argument("--num-cnt-proc", type=int, default=64, help="process number for counting data")
    parser.add_argument("--num-archive-proc", type=int, default=64, help="process number for archiving txts into tars")
    parser.add_argument("--logging-level", type=int, default=NOTSET, help="logging level for logger")
    args = parser.parse_args()
    base_data_dir: PosixPath = Path(os.getenv("HOME", None)) / "datasets" / "Taisu2_datasets" / args.base_data_folder
    if not base_data_dir.exists():
        raise FileNotFoundError(f"basic Taisu2 dataset directory - {base_data_dir}, cannot be found!")
    args.base_data_dir = base_data_dir

    specific_data_dir: PosixPath = base_data_dir / args.specific_data_folder
    if not specific_data_dir.exists():
        raise FileNotFoundError(f"specific Taisu2 dataset directory - {specific_data_dir}, cannot be found!")
    args.specific_data_dir = specific_data_dir

    naive_tars_dir: PosixPath = specific_data_dir / "image-text-pairs"
    if not naive_tars_dir.exists():
        raise FileNotFoundError(f"naive tar files directory - {naive_tars_dir}, cannot be found!")
    args.naive_tars_dir = naive_tars_dir

    def get_tars_num() -> int:
        tars_iterator = naive_tars_dir.glob("*.tar")
        tars_num = 0
        while True:
            try:
                _ = next(tars_iterator)
                tars_num += 1
            except StopIteration as _:
                break
        return tars_num
    args.tars_num = get_tars_num()

    if (not args.recaption_idx) or (args.recaption_idx < 0):
        raise ArgumentError(f"recaption index should be an integer greater than 0, but get {args.recaption_idx}")
    output_dir: PosixPath = specific_data_dir / f"{args.recaption_idx}th_recaption"
    if not output_dir.exists():
        raise FileNotFoundError(f"output directory - {output_dir}, cannot be found!")
    args.output_dir = output_dir

    res_tars_dir: PosixPath = output_dir / "txt-tars"
    if res_tars_dir.exists():
        shutil.rmtree(res_tars_dir)
    args.res_tars_dir = res_tars_dir

    return args


def init_logger(output_dir: PosixPath = None, logging_level: int = NOTSET):
    global logger
    logger = logging.getLogger("txts -> tars archiving")
    formatter = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")
    stream_hndlr = logging.StreamHandler(); stream_hndlr.setFormatter(formatter); stream_hndlr.setLevel(logging_level)
    log_p: PosixPath = output_dir / "txts_archive.log"
    log_p.unlink(missing_ok=True)
    file_hndlr = logging.FileHandler(filename=log_p, mode="a", encoding="utf-8")
    file_hndlr.setFormatter(formatter); file_hndlr.setLevel(logging_level)
    logger.addHandler(stream_hndlr); logger.addHandler(file_hndlr)

    return None


def args_log(args: Namespace = None):
    global logger
    logger.info(f"basic Taisu2 dataset directory: {args.base_data_dir}")
    logger.info(f"specific Taisu2 dataset folder: {args.specific_data_folder}")
    logger.info(f"recaption integer index: {args.recaption_idx}")
    logger.info(f"naive tar files number: {args.tars_num}")
    logger.info(f"parallel process number for image-alttext pairs counting: {args.num_cnt_proc}")
    logger.info(f"parallel process number for txt -> tar archiving: {args.num_archive_proc}")

    return None


def recaption_json_read(output_dir: PosixPath = None, recaption_idx: int = None) -> Dict[str, str]:
    json_p: PosixPath = output_dir / f"{recaption_idx}th_recaption.json"
    if not json_p.exists():
        raise FileNotFoundError(f"recaption json file - {json_p.name}, cannot be found under diretory - {json_p.parent}!")
    read_st_time = datetime.now()
    with open(json_p, mode="r", encoding="ut-8") as json_fp:
        recaption_res = json.load(json_fp)
    read_ed_time = datetime.now()
    read_secs = (read_ed_time - read_st_time).total_seconds()
    logger.info(f"reading json file {json_p.name} from directory {json_p.parent} has taken {read_secs // 60} minutes, "
                f"and {read_secs % 60} seconds")

    return recaption_res


def tars_stem_iter(
                   naive_tars_dir: PosixPath = None, 
                   naive_tars_num: int = 0, 
                   num_proc: int = 0
                  ):
    naive_tars_iter = naive_tars_dir.glob("*..tar")
    tars_num_per_proc = naive_tars_num // num_proc
    tars_remainder_num = naive_tars_num % num_proc
    cur_tars_num = 0
    cur_tars_stem = []
    while True:
        try:
            cur_tar_p = next(naive_tars_iter)
            cur_tars_stem.append(cur_tar_p.stem)
            cur_tars_num += 1
            if cur_tars_num == tars_num_per_proc:
                if tars_remainder_num > 0:
                    cur_tar_p = next(naive_tars_iter)
                    cur_tars_stem.append(cur_tar_p.stem)
                    tars_remainder_num -= 1
                yield cur_tars_stem
                cur_tars_num = 0
                cur_tars_stem = []
        except StopIteration as _:
            break
    if cur_tars_stem:
        yield cur_tars_stem


def count_data_num_task_func(tars_stem: List[str], naive_tars_dir: PosixPath = None):
    global shared_data_num
    img_pat_str = r"\S+\.jpg"
    for tar_stem in tars_stem:
        tar_p = naive_tars_dir / f"{tar_stem}.tar"
        with tarfile.open(tar_p, mode="r", encoding="utf-8") as tar_fp:
            tar_imgnames = [memname for memname in tar_fp.getnames() if re.fullmatch(img_pat_str, memname)]
        with shared_data_num.get_lock():
            shared_data_num.value += len(tar_imgnames)
    return


def main():
    args = parse_args()
    init_logger(output_dir=args.output_dir, logging_level=args.logging_level)
    global logger
    st_time = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    logger.info(f"begin archiving txt files into tar files at {st_time}")

    args_log(args=args)
    recaption_res = recaption_json_read(output_dir=args.output_dir, recaption_idx=args.recaption_idx)
    data_num_from_json = len(recaption_res)
    global shared_data_num
    shared_data_num = multiprocessing.Value("i")
    count_data_num = partial(count_data_num_task_func, naive_tars_dir=args.naive_tars_dir)
    with futures.ProcessPoolExecutor(max_workers=args.num_cnt_proc) as cnt_proc_exec:
        _ = cnt_proc_exec.map(count_data_num, 
                              tars_stem_iter(
                                             naive_tars_dir=args.naive_tars_dir, 
                                             naive_tars_num=args.tars_num, 
                                             num_proc=args.num_cnt_proc
                                            )
                             )
    data_num_from_naive_tars = int(shared_data_num.value)
    if data_num_from_naive_tars != data_num_from_json:
        raise RuntimeError(f"image-alttext pairs number got from naive tars: {data_num_from_naive_tars}; "
                           f"from recaption json file: {data_num_from_json}, which are not equal")
    args.data_num = data_num_from_json

    with multiprocessing.Manager() as mp_manager:
        global recaption_dict, shared_lock
        recaption_dict = mp_manager.dict()
        shared_lock = mp_manager.Lock()
        mp_dict_st_time = datetime.now()
        recaption_dict.update(recaption_res)
        mp_dict_secs = (datetime.now() - mp_dict_st_time).total_seconds()
        logger.info(f"updating recaption results from json file into multi-process shared dictionary, "
                    f"takes {mp_dict_secs // 60} minutes, and {mp_dict_secs % 60} seconds in total")
        # TODO: Now here
        # with futures.ProcessPoolExecutor

    ed_time = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    logger.info(f"end archiving txt files into tar files at {ed_time}")


if __name__ == "__main__":
    main()
