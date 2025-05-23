import io, re, os
import gc
import sys
import time
import shutil
from datetime import datetime
import tarfile
from tarfile import TarInfo
from typing import Dict, Set, List
from pathlib import PosixPath, Path
import multiprocessing
from multiprocessing.managers import AcquirerProxy, DictProxy
from multiprocessing.synchronize import Barrier, Lock
from multiprocessing.sharedctypes import Synchronized
from concurrent import futures as futures
from functools import partial
import argparse
from argparse import Namespace
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL


logger: logging.Logger = None
pid_to_rank: DictProxy = None
proc_lock: AcquirerProxy = None
proc_barrier: Barrier = None

date_fmt_str = '%Y-%m-%d-%H:%M:%S'
log_fmt_str = "[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"

tartxt_info_lock: AcquirerProxy = None

native_tarnames: Set = None
native_txtnames: Set = None
native_tartxt_dict: DictProxy = None
shared_native_tarnum: Synchronized = None
shared_native_txtnum: Synchronized = None

recap_tarnames: Set = None
recap_txtnames: Set = None
recap_tartxt_dict: DictProxy = None
shared_recap_tarnum: Synchronized = None
shared_recap_txtnum: Synchronized = None


def parse_args():
    parser = argparse.ArgumentParser(description="command line arguments parser for Taisu2 image-alttext paris recaptions archiving")
    parser.add_argument("--taisu2-base-folder", type=str, default=None, help="Taisu2 image-alttext pairs basic folder")
    parser.add_argument("--taisu2-specific-folder", type=str, default=None, help="Taisu2 image-alttext pairs specific folder")
    parser.add_argument("--taisu2-recap-folder", type=str, default=None, help="folder stored recaptions tar files of Taisu2 image-alttext pairs")
    parser.add_argument("--tarread-workers", type=int, default=80, help="max parallel workers for getting tar and txt files names and total number")
    parser.add_argument("--max-workers", type=int, default=80, help="max parallel workers for archiving recaptions of Taisu2 image-alttext dataset")
    parser.add_argument("--logging-level", type=int, choices=(NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL), default=NOTSET, 
                        help="output logging level for logging.Logger")
    args = parser.parse_args()
    args.taisu2_base_dir = Path(os.getenv("HOME", "")) / "datasets" / args.taisu2_base_folder
    if not args.taisu2_base_dir.exists():
        raise FileNotFoundError(f"basic directory of Taisu2 image-alttext paris: {args.taisu2_base_dir}, does not exist!")
    args.taisu2_specific_folder = "/".join(args.taisu2_specific_folder).strip().strip("/").strip()
    args.taisu2_specific_dir = args.taisu2_base_dir / args.taisu2_specific_folder
    if not args.taisu2_specific_dir.exists():
        raise FileNotFoundError(f"specified directory of Taisu2 image-alttext pairs: {args.taisu2_specific_dir}, does not exist!")
    args.preprocessed_data_dir = args.taisu2_specific_dir / "image-text-pairs"  # (1)
    if not args.preprocessed_data_dir.exists():
        raise FileNotFoundError(f"preprocessed Taisu2 image-alttext pairs directory: {args.preprocessed_data_dir}, does not exist!")
    args.recap_txt_dir = args.taisu2_specific_dir / args.taisu2_recap_folder  # (2)
    if not args.recap_txt_dir.exists():
        raise FileNotFoundError(f"recaption output directory of Taisu2 image-alttext pairs: {args.recap_txt_dir}, does not exist!")
    args.archive_out_dir = args.taisu2_specific_dir / f"{args.taisu2_recap_folder}_imgtxt_pairs"  # (3)
    args.archive_out_data_dir = args.archive_out_dir / "image-text-pairs"  # (4)
    if args.archive_out_dir.exists():
        shutil.rmtree(args.archive_out_dir)
    args.archive_out_data_dir.mkdir(parents=True, exist_ok=False)

    return args


def init_logger(archive_out_dir: PosixPath, logging_level: int):
    global logger, log_fmt_str
    logger = logging.getLogger("Taisu2 image-alttext pairs recaptions archive")
    logger.setLevel(logging_level)
    formatter = logging.Formatter(log_fmt_str)
    stream_hndlr = logging.StreamHandler()
    stream_hndlr.setFormatter(formatter)
    stream_hndlr.setLevel(logging_level)
    file_hndlr = logging.FileHandler(archive_out_dir / "recaption_archive.log")
    file_hndlr.setFormatter(formatter)
    file_hndlr.setLevel(logging_level)
    logger.addHandler(stream_hndlr)
    logger.addHandler(file_hndlr)

    return


def args_check_and_log(args: Namespace = None):
    global logger
    logger.info(f"basic folder of Taisu2 image-alttext pairs (args.taisu2_base_folder): {args.taisu2_base_folder}")
    logger.info(f"specific folder of Taisu2 image-alttext pairs (args.taisu2_specific_folder): {args.taisu2_specific_folder}")
    logger.info(f"recaptions folder of Taisu2 image-alttext pairs (args.taisu2_recap_folder): {args.taisu2_recap_folder}")
    logger.info(f"recaptions archiving output directory (args.archive_out_dir): {args.archive_out_dir}")
    logger.info(f"maximum parallel workers for recaption archiving process pool (args.max_workers): {args.max_workers}")
    logger.info(f"maximum parallel workers for getting names and total numbers of tar and txt files (args.tarread_workers): {args.tarread_workers}")

    return


def single_tartxt_info(
                       tar_p: PosixPath, 
                       shared_lock: AcquirerProxy, 
                       tartxt_dict: DictProxy, 
                       shared_tarnum: Synchronized, 
                       shared_txtnum: Synchronized
                      ):
    tarname = tar_p.name
    with tarfile.open(tar_p, mode="r", encoding="utf-8") as tar_fp:
        txtnames = [mem_name for mem_name in tar_fp.getnames() if mem_name.endswith(".txt")]
    txtnum = len(txtnames)
    with shared_lock:
        tartxt_dict[tarname] = txtnames
    with shared_tarnum.get_lock():
        shared_tarnum.value += 1
    with shared_txtnum.get_lock():
        shared_txtnum.value += txtnum

    return


def get_tartxt_info_task_func(tar_p: PosixPath, data_type: str = "native"):
    global tartxt_info_lock
    if data_type == "native":
        global native_tartxt_dict, shared_native_tarnum, shared_native_txtnum
        single_tartxt_info(
                           tar_p=tar_p, shared_lock=tartxt_info_lock, 
                           tartxt_dict=native_tartxt_dict, 
                           shared_tarnum=shared_native_tarnum, 
                           shared_txtnum=shared_native_txtnum
                          )
    else:
        global recap_tartxt_dict, shared_recap_tarnum, shared_recap_txtnum
        single_tartxt_info(
                           tar_p=tar_p, shared_lock=tartxt_info_lock, 
                           tartxt_dict=recap_tartxt_dict, 
                           shared_tarnum=shared_recap_tarnum, 
                           shared_txtnum=shared_recap_txtnum
                          )
    return


def get_tartxt_info(tars_dir: PosixPath, max_workers: int, data_type: str = None, args: Namespace = None):
    tars_generator = Path(tars_dir).glob("*.tar")
    with multiprocessing.Manager() as info_manager:
        global tartxt_info_lock
        tartxt_info_lock = info_manager.Lock()
        if data_type == "native":
            global native_tartxt_dict
            native_tartxt_dict = info_manager.dict()
            global shared_native_tarnum, shared_native_txtnum
            shared_native_tarnum = multiprocessing.Value("i")
            shared_native_txtnum = multiprocessing.Value("i")
        elif data_type == "recap":
            global recap_tartxt_dict
            recap_tartxt_dict = info_manager.dict()
            global shared_recap_tarnum, shared_recap_txtnum
            shared_recap_tarnum = multiprocessing.Value("i")
            shared_recap_txtnum = multiprocessing.Value("i")
        else:
            raise ValueError(f"for tar and txt files information getting task, parameter `data_type` could only be `native`, or `recap`")
        partial_getinfo_func = partial(get_tartxt_info_task_func, data_type=data_type)
        with futures.ProcessPoolExecutor(max_workers) as info_exec:
            _ = info_exec.map(partial_getinfo_func, tars_generator, chunksize=1)

        if data_type == "native":
            global native_tarnames, native_txtnames
            native_tarnames = set(); native_txtnames = set()
            args.native_tarnum = shared_native_tarnum.value
            args.native_txtnum = shared_native_txtnum.value
            native_tarnames.update(native_tartxt_dict.keys())
            native_txtnames.update(native_tartxt_dict.values())
            del native_tartxt_dict; del shared_native_tarnum; del shared_native_txtnum
        else:  # recap
            global recap_tarnames, recap_txtnames
            recap_tarnames = set(); recap_txtnames = set()
            args.recap_tarnum = shared_recap_tarnum.value
            args.recap_txtnum = shared_recap_txtnum.value
            recap_tarnames.update(recap_tartxt_dict.keys())
            recap_txtnames.update(recap_tartxt_dict.values())
            del recap_tartxt_dict; del shared_recap_tarnum; del shared_recap_txtnum

    del info_manager
    gc.collect()
    return


def recaption_archive_init_func():
    global logger, pid_to_rank, proc_lock
    pid = str(os.getpid())
    with proc_lock:
        proc_rank = str(len(pid_to_rank))
        pid_to_rank[pid] = proc_rank
    global proc_barrier
    if not isinstance(proc_barrier, Barrier) or proc_barrier is None:
        with proc_lock:
            logger.error(f"process with rank index {proc_rank} hasn't got the `Barrier` object, hence exit its initialization abnormally")
        sys.exit(1)
    logger.info(f"process with pid {pid} got rank index {proc_rank}, and has finished initialization")


def tarnames_generator_func(max_workers: int = None):
    global native_tarnames
    tarnames_num_per_worker = len(native_tarnames) // max_workers
    remainder = len(native_tarnames) % max_workers
    tarnames_iter = iter(native_tarnames)
    tarnames_list = []
    while True:
        try:
            tarnames_list.append(next(tarnames_iter))
            if len(tarnames_list) == tarnames_num_per_worker:
                if remainder > 0:
                    tarnames_list.append(next(tarnames_iter))
                    remainder -= 1
                yield tarnames_list
        except StopIteration as _:
            break
    if tarnames_list:
        yield tarnames_list


def recaption_archive_task_func(tarnames: List[str], args: Dict = None):
    global logger, proc_lock, proc_barrier
    global date_fmt_str
    native_data_dir = args["preprocessed_data_dir"]
    recaption_dir = args["recap_txt_dir"]
    output_dir = args["archive_out_data_dir"]

    proc_rank = os.getenv("RANK", None)
    if proc_rank is None:
        with proc_lock:
            logger.error(f"process with pid {os.getpid()} cannot get its process rank, henc exit abnormally")
        sys.exit(1)

    proc_st_time = datetime.now()
    with proc_lock:
        logger.info(f"process with rank {proc_rank} begins to handle all tar files dispendided to it at {datetime.strftime(proc_st_time, date_fmt_str)}")
    tarnum = 0
    datanum = 0
    for tarname in tarnames:
        global recap_txtnames
        tar_stem = tarname.split(".")[0]
        filter_func = lambda x: True if re.match(tar_stem, x) is not None and re.match(tar_stem, x).span() == (0, 5) else False
        txtnames = filter(filter_func, recap_txtnames)

        native_tarp = native_data_dir / tarname
        native_tarfp = tarfile.open(native_tarp, mode="r", encoding="utf-8")

        recap_tarp = recaption_dir / tarname
        recap_tarfp = tarfile.open(recap_tarp, mode="r", encoding="utf-8")

        output_tarp = output_dir / tarname
        output_tarfp = tarfile.open(output_tarp, mode="w", encoding="utf-8")

        with proc_lock:
            logger.info(f"process with rank {proc_rank} begins to archive images and recaptions of tar file {tarname}")

        for txtname in txtnames:
            imgname = txtname.split(".")[0] + ".jpg"
            img_raw_bytes = native_tarfp.extractfile(imgname).read()
            img_bytes = io.BytesIO(img_raw_bytes)
            img_tarinfo = TarInfo(imgname); img_tarinfo.mtime = int(time.time()); img_tarinfo.size = len(img_raw_bytes)
            output_tarfp.addfile(img_tarinfo, img_bytes)

            recap_raw_bytes = recap_tarfp.extractfile(txtname).read()
            recap_bytes = io.BytesIO(recap_raw_bytes)
            recap_tarinfo = TarInfo(txtname); recap_tarinfo.mtiem = int(time.time()); recap_tarinfo.size = len(recap_raw_bytes)
            output_tarfp.addfile(img_tarinfo, img_bytes)
            output_tarfp.addfile(recap_tarinfo, recap_bytes)

            datanum += 1

        native_tarfp.close()
        recap_tarfp.close()
        output_tarfp.close()
        tarnum += 1

        with proc_lock:
            logger.info(f"process with rank {proc_rank} ends archiving images and recaptions of tar file {tarname}")

    proc_ed_time = datetime.now()
    with proc_lock:
        logger.info(f"process with rank {proc_rank} ends handling all tar files dispendided to it at {datetime.strftime(proc_ed_time, date_fmt_str)}, "
                    f"takes {(proc_ed_time - proc_st_time).total_seconds() / 60:.3f} minutes in total, has completed {tarnum} tar files and {datanum} "
                    f"image-alttext pairs in total")

    return


def main():
    args = parse_args()
    init_logger(args.archive_out_dir, args.logging_level)
    args_check_and_log(args)
    global logger, date_fmt_str

    st_time = datetime.now()
    logger.info(f"begin checking tar and txt files before and after recpation at {datetime.strftime(st_time, date_fmt_str)}")
    get_tartxt_info(tars_dir=args.preprocessed_data_dir, max_workers=args.tarread_workers, data_type="native", args=args)
    get_tartxt_info(tars_dir=args.recap_txt_dir, max_workers=args.tarread_workers, data_type="recap", args=args)

    if args.native_tarnum != args.recap_tarnum:
        logger.error(f"tar files number isn't equal between before and after recaption, before: {args.native_tarnum}; after: {args.recap_tarnum}")
        sys.exit(1)
    if args.native_txtnum != args.recap_txtnum:
        logger.error(f"image-alttext pairs number isn't equal between before and after recaption, before: {args.native_txtnum}; after: {args.recap_txtnum}")
        sys.exit(1)

    global native_tarnames, recap_tarnames
    if native_tarnames != recap_tarnames:
        logger.error(f"native tar files are not absolutely equal to ones after recaption")
        diff_tarnames_1 = native_tarnames.difference(recap_tarnames)
        diff_tarnames_2 = recap_tarnames.difference(native_tarnames)
        if diff_tarnames_1:
            logger.error(f"extra native tar files number: {len(diff_tarnames_1)}; extra native tar files: {diff_tarnames_1}")
        if diff_tarnames_2:
            logger.error(f"extra recaption tar files number: {len(diff_tarnames_2)}; extra recaption tar files: {diff_tarnames_2}")
        sys.exit(1)

    global native_txtnames, recap_txtnames
    if native_txtnames != recap_txtnames:
        logger.error(f"native image-alttext pairs are not absolutely equal to ones after recaption")
        diff_txtnames_1 = native_txtnames.difference(recap_txtnames)
        diff_txtnames_2 = recap_txtnames.difference(native_txtnames)
        if diff_txtnames_1:
            logger.error(f"extra native txt files number: {len(diff_txtnames_1)}; extra native txt files: {diff_txtnames_1}")
        if diff_txtnames_2:
            logger.error(f"extra recaption txt files number: {len(diff_txtnames_2)}; extra recaption txt files: {diff_txtnames_2}")
        sys.exit(1)
    ed_time = datetime.now()
    logger.info(f"end checking tar and txt files before and after recaption at {datetime.strftime(ed_time, date_fmt_str)}, "
                f"takes {(ed_time - st_time).total_seconds() / 60:3.f} minutes in total")

    st_time = datetime.now()
    logger.info(f"begin archiving recaptions into new tar files at {datetime.strftime(st_time, date_fmt_str)}")
    with multiprocessing.Manager() as archive_manager:
        global pid_to_rank, proc_lock
        global proc_barrier
        pid_to_rank = archive_manager.dict()
        proc_lock = archive_manager.Lock()
        proc_barrier = multiprocessing.Barrier(args.max_workers)
        partial_recaption_archive_func = partial(recaption_archive_task_func, args=vars(args))
        with futures.ProcessPoolExecutor(args.max_workers, initializer=recaption_archive_init_func, initargs=()) as archive_exec:
            _ = archive_exec.map(partial_recaption_archive_func, tarnames_generator_func(args.max_workers), chunksize=1)
    ed_time = datetime.now()
    logger.info(f"end archiving recaptions into new tar files at {datetime.strftime(ed_time, date_fmt_str)}, takes {(ed_time - st_time) // 3600:d} hours "
                f"and {(((ed_time - st_time) % 3600) / 60):.3f} minutes in total")


if __name__ == "__main__":
    main()
