import io, os
import gc, sys
import json
import time
import math
import shutil
import argparse
from argparse import Namespace
from typing import Dict, List, Tuple
import tarfile
from tarfile import TarInfo
from datetime import datetime
from concurrent import futures as futures
import multiprocessing
from multiprocessing.managers import DictProxy, BarrierProxy, AcquirerProxy
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock
from pathlib import Path, PosixPath
from functools import partial
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL


logger: logging.Logger = None
pid_to_rank: DictProxy = None
proc_lock: AcquirerProxy = None
proc_barrier: BarrierProxy = None
date_fmt = '%Y-%m-%d-%H:%M:%S'

native_data_num: Synchronized = None
data_num_lock: Lock = None
tarname_to_imgnames: Dict[str, List[str]] = None


def parse_args():
    parser = argparse.ArgumentParser(description="command line arguments parser for preprocessed image-alttext pairs renaming and rearchiving")
    parser.add_argument("--taisu2-base-folder", type=str, default=None, help="basic image-alltext folder for Taisu2")
    parser.add_argument("--taisu2-specific-folder", type=str, default=None, nargs="+", 
                        help="specific Taisu2 image-alttext folders under which image and text in tars will be renamed and rearchived")
    parser.add_argument("--taisu2-output-folder", type=str, default="rename_and_rearchive", help="output folder for Taisu2 renaming and rearchiving")
    parser.add_argument("--max-workers-for-data-num", type=int, default=90, help="maximum workers for get native image-alttext pairs number via a process pool")
    parser.add_argument("--data-num-per-tar", type=lambda x: int(eval(x)), default=10000, help="image-alttext data number each tar file holds")
    parser.add_argument("--max-workers", type=int, default=90, help="maximum workers for Taisu2 image-alttext data renaming and rearchiving")
    parser.add_argument("--logging-level", type=int, default=DEBUG, choices=(NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL), 
                        help="logging level for logging.Logger")
    args = parser.parse_args()
    args.taisu2_base_dir = Path(os.getenv("HOME", "")) / "datasets" / "Taisu2_datasets" / args.taisu2_base_folder
    if not args.taisu2_base_dir.exists():
        raise FileNotFoundError(f"basic directory of Taisu2 dataset - {args.taisu2_base_dir}, does not exist!")
    args.taisu2_specific_folder = "/".join(args.taisu2_specific_folder).strip("/")
    args.taisu2_specific_dir = args.taisu2_base_dir / args.taisu2_specific_folder
    if not args.taisu2_specific_dir.exists():
        raise FileNotFoundError(f"specific directory of Taisu2 dataset - {args.taisu2_specific_dir}, does not exist!")
    args.taisu2_output_dir = args.taisu2_base_dir / args.taisu2_output_folder
    if args.taisu2_output_dir.exists():
        shutil.rmtree(args.taisu2_output_dir)
    else:
        args.taisu2_output_dir.mkdir(parents=False, exist_ok=False)
    args.taisu2_out_data_folder = "image-text-pairs"
    args.taisu2_out_data_dir = args.taisu2_output_dir / args.taisu2_out_data_folder
    if args.taisu2_out_data_dir.exists():
        shutil.rmtree(args.taisu2_out_data_dir)
    args.taisu2_out_data_dir.mkdir(parents=False, exist_ok=False)

    return args


def init_logger(out_dir: PosixPath, logging_level: int = NOTSET):
    global logger
    logger = logging.getLogger("Taisu2 rename and rearchive"); logger.setLevel(logging_level)
    fmt_str = "[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"
    hndlr_fmt = logging.Formatter(fmt_str)
    logger_p = out_dir / "rename_and_rearchive.log"
    if logger_p.exists():
        logger_p.unlink(missing_ok=False)
    stream_hndlr = logging.StreamHandler()
    stream_hndlr.setFormatter(hndlr_fmt); stream_hndlr.setLevel(logging_level)
    logger.addHandler(stream_hndlr)
    file_hndlr = logging.FileHandler(logger_p, encoding="utf-8")
    file_hndlr.setFormatter(hndlr_fmt); file_hndlr.setLevel(logging_level)
    logger.addHandler(file_hndlr)

    return


def get_native_tar_num(native_data_dir: PosixPath = None, args: Namespace = None):
    global logger
    naitve_tars_generator = native_data_dir.glob("*.tar")
    native_tar_num = 0
    while True:
        try:
            _ = next(naitve_tars_generator)
            native_tar_num += 1
        except StopIteration as _:
            break
    if native_tar_num == 0:
        logger.error(f"tar files number under Taisu2 native directory - {native_data_dir} is 0!")
        sys.exit(1)
    args.native_tar_num = native_tar_num

    return


def data_num_init_func(shared_data_num: Synchronized, shared_lock: Lock):
    global native_data_num, data_num_lock
    native_data_num = shared_data_num
    data_num_lock = shared_lock


def tars_path_generator(native_data_dir: PosixPath, native_tar_num: int, max_workers: int):
    tars_generator = native_data_dir.glob("*.tar")
    tarnum_per_worker = native_tar_num // max_workers
    remain_tarnum = native_tar_num % max_workers
    tars_list: List[PosixPath] = []
    while True:
        try:
            if len(tars_list) == tarnum_per_worker:
                if remain_tarnum:
                    tars_list.append(next(tars_generator))
                    remain_tarnum -= 1
                yield tars_list
                tars_list = []
            elif len(tars_list) < tarnum_per_worker:
                tars_list.append(next(tars_generator))
            else:  # error situation (length greater than `tarnum_per_worker`)
                raise ValueError(f"the length of generated tar files list {len(tars_list)} is greater than predefined max value {tarnum_per_worker}")
        except StopIteration as _:
            break
    if tars_list:
        yield tars_list

    return


def data_num_task_func(native_tars_list: List[PosixPath], res_p: PosixPath = None):
    global native_data_num, data_num_lock
    tar_to_imgs_dict = dict()
    for native_tar_p in native_tars_list:
        with tarfile.open(native_tar_p, mode="r", encoding="utf-8") as native_tar_fp:
            imgnames = [memname for memname in native_tar_fp.getnames() if memname.endswith(".jpg")]
        with native_data_num.get_lock():
            native_data_num.value += len(imgnames)
        tar_to_imgs_dict[native_tar_p.name] = imgnames
    with data_num_lock:
        with open(res_p, mode="a", encoding="utf-8") as res_fp:
            for tarname, imgnames in tar_to_imgs_dict.items():
                json.dump({tarname: imgnames}, res_fp, ensure_ascii=False)
                res_fp.write("\n")

    return


def get_native_data_num(native_data_dir: PosixPath = None, args: Namespace = None):
    global logger
    native_tars_generator = tars_path_generator(native_data_dir, args.native_tar_num, args.max_workers_for_data_num)
    if args.max_workers_for_data_num > args.native_tar_num:
        logger.warning(f"native tar files number for renaming and rearchiving: {args.native_tar_num}; maximum workers for "
                       f"counting data number: {args.max_workers_for_data_num}, hence set workers number to {args.native_tar_num}")
        args.max_workers_for_data_num = args.native_tar_num
    shared_data_num = multiprocessing.Value("i")
    shared_lock = multiprocessing.Lock()
    tarname_to_imgnames_p: PosixPath = args.taisu2_output_dir / "tarname_to_imgnames.jsonl"
    args.tarname_to_imgnames_p = tarname_to_imgnames_p
    try:
        initargs = (shared_data_num, shared_lock)
        partial_task_func = partial(data_num_task_func, res_p=tarname_to_imgnames_p)
        with futures.ProcessPoolExecutor(args.max_workers_for_data_num, initializer=data_num_init_func, initargs=initargs) as data_num_exec:
            _ = data_num_exec.map(partial_task_func, native_tars_generator, chunksize=1)
    except Exception as err:
        logger.error(f"when using process pool to get native image-alttext pairs number, encounter an error ({type(err)}): {err}")
        sys.exit(1)
    args.native_data_num = shared_data_num.value
    gc.collect()

    return


def args_set_and_check(args: Namespace = None):
    global logger
    logger.info(f"Taisu2 renaming and rearchiving basic folder (args.taisu2_base_foder): {args.taisu2_base_folder}")
    logger.info(f"Taisu2 renaming and rearchiving specific folder (args.taisu2_specific_folder): {args.taisu2_specific_folder}")
    logger.info(f"Taisu2 renaming and rearchiving output folder (args.taisu2_output_folder): {args.taisu2_output_folder}")
    logger.info(f"Taisu2 renaming and rearchiving data output folder (args.taisu2_out_data_folder): {args.taisu2_out_data_folder}")
    logger.info(f"maximum number of workers for collecting native image-alttext pairs data (args.max_workers_for_data_num): {args.max_workers_for_data_num}")
    logger.info(f"image-alttext data number per renamed and rearchived tar file (args.data_num_per_tar): {args.data_num_per_tar}")
    logger.info(f"maximum workers number for renaming and rearchiving task (args.max_workers): {args.max_workers}")
    get_native_tar_num(args.taisu2_specific_dir, args=args)
    logger.info(f"the native tar files number (args.native_tar_num): {args.native_tar_num}")
    get_native_data_num(args.taisu2_specific_dir, args=args)
    logger.info(f"the native image-alttext data number (args.native_data_num): {args.native_data_num}")
    logger.info(f"the tar file name -> image names mapping dictionary is stored at (args.tarname_to_imgnames_p): {args.tarname_to_imgnames_p}")

    logger.info(f"loading tar files name -> image names mapping from local file `tarname_to_imgnames.jsonl`")
    global tarname_to_imgnames
    if tarname_to_imgnames is None or (not isinstance(tarname_to_imgnames, dict)) or (len(tarname_to_imgnames) != 0):
        tarname_to_imgnames = dict()
    with open(args.tarname_to_imgnames_p, mode="r", encoding="utf-8") as tar_to_imgs_fp:
        for tar_to_imgs in tar_to_imgs_fp:
            tarname_to_imgnames.update(json.loads(tar_to_imgs))
    tmp_tar_num = len(tarname_to_imgnames)
    if tmp_tar_num != args.native_tar_num:
        raise ValueError(f"real native tar files number: {args.native_tar_num}, but get {tmp_tar_num} tar files from local file "
                         f"`tarname_to_imgnames.jsonl`")
    tmp_data_num = sum(len(imgnames) for imgnames in tarname_to_imgnames.values())
    if tmp_data_num != args.native_data_num:
        raise ValueError(f"real native image-alttext pairs number: {args.native_data_num}, but get {tmp_data_num} data from local file "
                         f"`tarname_to_imgnames.jsonl`")
    del tmp_tar_num, tmp_data_num
    logger.info(f"having loaded tar files name -> image names mapping from jsonl file into global dictionary `tarname_to_imgnames`")

    return


def proc_init_func():
    global logger, pid_to_rank, proc_lock, proc_barrier
    pid = os.getpid()
    if not isinstance(pid_to_rank, DictProxy):
        raise TypeError(f"in process with pid {pid}, the variable `pid_to_rank` is not a `DictProxy` instance")
    with proc_lock:
        rank = len(pid_to_rank)
        pid_to_rank[str(pid)] = str(rank)
    os.environ["RANK"] = str(rank)
    if not isinstance(proc_lock, AcquirerProxy):
        raise TypeError(f"in process with pid {pid}, the variable `proc_lock` is not a `AcquirerProxy` instance")
    if not isinstance(proc_barrier, BarrierProxy):
        raise TypeError(f"in process with pid {pid}, the variable `proc_barrier` is not a `BarrierProxy` instance")
    with proc_lock:
        logger.info(f"process with pid {pid} has finished initialization, and got process rank `{rank}`")


def rename_and_rearchive_generator(native_data_dir: PosixPath, native_datanum: int, datanum_per_tar: int, num_workers: int):
    global tarname_to_imgnames

    new_tars_num = math.ceil(native_datanum / datanum_per_tar)
    tarnum_per_worker = native_datanum // (num_workers * datanum_per_tar)
    tarnum_all_workers = [tarnum_per_worker] * num_workers
    remain_datanum = native_datanum - num_workers * tarnum_per_worker * datanum_per_tar
    remain_tarnum = math.ceil(remain_datanum / datanum_per_tar)
    datanum_last_tar = remain_datanum % datanum_per_tar
    has_last_tar = [False] * num_workers
    for remain_tar_idx in range(remain_tarnum):
        tarnum_all_workers[remain_tar_idx % num_workers] += 1
        if remain_tar_idx == remain_tarnum - 1:
            has_last_tar[remain_tar_idx % num_workers] = True
    last_tar_worker_idx = has_last_tar.index(True)
    datanum_all_workers = [0] * num_workers
    for worker_idx in range(num_workers):
        if not has_last_tar[worker_idx]:
            datanum_all_workers[worker_idx] = datanum_per_tar * tarnum_all_workers[worker_idx]
        else:
            datanum_all_workers[worker_idx] = datanum_per_tar * (tarnum_all_workers[worker_idx] - 1)
            datanum_all_workers[worker_idx] += datanum_last_tar
    tarnum_last_tar_worker = tarnum_all_workers.pop(last_tar_worker_idx)
    datanum_last_tar_worker = datanum_all_workers.pop(last_tar_worker_idx)
    tarnum_all_workers.append(tarnum_last_tar_worker)
    datanum_all_workers.append(datanum_last_tar_worker)
    if sum(tarnum_all_workers) != new_tars_num:
        worker_to_tar_num = {str(idx): str(tar_num) for idx, tar_num in enumerate(tarnum_all_workers)}
        raise ValueError(f"after splitting renamed and rearchived tar files for all workers, the total number of renamed and rearchived tar files should be "
                         f"{new_tars_num}, but get {sum(tarnum_all_workers)}, details: {worker_to_tar_num}")
    if sum(datanum_all_workers) != native_datanum:
        raise ValueError(f"after splitting renamed and rearchived tar files for all workers, the total number of renamed and rearchived data number "
                         f"should be equal to the native one `{native_datanum}`, but get {sum(datanum_all_workers)}")
    for worker_idx, (tar_per_worker, data_per_worker) in enumerate(zip(tarnum_all_workers, datanum_all_workers)):
        if math.ceil(data_per_worker / datanum_per_tar) != tar_per_worker:
            raise ValueError(f"for the {worker_idx}th worker, it handles {data_per_worker} image-alttext pairs, but {tar_per_worker} tar files "
                             f"(should be {math.ceil(data_per_worker / datanum_per_tar)})!")
    tars_generator = native_data_dir.glob("*.tar")
    prev_workers_acc_imgnum = 0
    cur_worker_imgnum = 0
    worker_idx = 0
    tar_p = next(tars_generator)
    imgnames = tarname_to_imgnames[tar_p.name]
    while True:
        try:
            if cur_worker_imgnum == 0:
                st_newtar_name = f"{int(prev_workers_acc_imgnum // datanum_per_tar):05d}.tar"
                st_native_imgname = imgnames[0]
            if cur_worker_imgnum + len(imgnames) < datanum_all_workers[worker_idx]:
                cur_worker_imgnum += len(imgnames)
                tar_p = next(tars_generator)
                imgnames = tarname_to_imgnames[tar_p.name]
            else:
                tar_p = next(tars_generator)
                next_imgnames = tarname_to_imgnames[tar_p.name]
                if cur_worker_imgnum + len(imgnames) == datanum_all_workers[worker_idx]:
                    yield st_native_imgname, st_newtar_name, datanum_all_workers[worker_idx], tarnum_all_workers[worker_idx]
                    cur_worker_imgnum = 0
                    st_native_imgname = next_imgnames[0]
                else:
                    yield st_native_imgname, st_newtar_name, datanum_all_workers[worker_idx], tarnum_all_workers[worker_idx]
                    cur_worker_imgnum = len(imgnames) - (datanum_all_workers[worker_idx] - cur_worker_imgnum)
                    st_native_imgname = imgnames[(-1) * cur_worker_imgnum]
                prev_workers_acc_imgnum += datanum_all_workers[worker_idx]
                st_newtar_name = f"{int(prev_workers_acc_imgnum // datanum_per_tar):05d}.tar"
                imgnames = next_imgnames; del next_imgnames
                worker_idx += 1
        except StopIteration as _:
            break
    if cur_worker_imgnum:
        yield st_native_imgname, st_newtar_name, datanum_all_workers[worker_idx], tarnum_all_workers[worker_idx]

    return


def rename_and_rearchive_task_func(iter_params: Tuple[str, str, int, int], args: Dict = None):
    global logger, proc_lock, proc_barrier

    if len(iter_params) != 4:
        with proc_lock:
            logger.error(f"for each sub-process, parameters number got from the iterable parameter of process pool's map method should be 4")
        sys.exit(1)

    st_native_imgname = iter_params[0]
    st_newtar_name = iter_params[1]
    imgnum = iter_params[2]
    newtar_num = iter_params[3]

    proc_rank = os.getenv("RANK", None)
    if proc_rank is None:
        with proc_lock:
            logger.error(f"process with pid {os.getpid()} cannot get its process rank index, hence exit procedure abnormally")
        sys.exit(1)

    native_data_dir: PosixPath = args["taisu2_specific_dir"]
    native_tars_generator = native_data_dir.glob("*.tar")
    datanum_per_tar: int = args["data_num_per_tar"]
    output_data_dir: PosixPath = args["taisu2_out_data_dir"]

    cur_newtar_name = st_newtar_name
    cur_newtar_imgnum = 0
    added_imgnum = 0

    st_native_tarname = st_native_imgname[: 5] + ".tar"
    while True:
        tmp_tar_p = next(native_tars_generator)
        if tmp_tar_p.name == st_native_tarname:
            cur_native_tar_p = tmp_tar_p
            break
    cur_native_tar_fp = tarfile.open(cur_native_tar_p, mode="r", encoding="utf-8")
    imgnames_list = [mem_name for mem_name in cur_native_tar_fp.getnames() if mem_name.endswith(".jpg")]
    imgnames_list = imgnames_list[imgnames_list.index(st_native_imgname): ]
    cur_native_imgnames = (imgname for imgname in imgnames_list)
    with proc_lock:
        logger.info(f"process with rank {proc_rank} rearchive native tar file {cur_native_tar_p.name} into new tar file {cur_newtar_name}")

    while newtar_num > 0:
        with proc_lock:
            logger.info(f"process with rank {proc_rank} begins renaming and rearchiving new tar file: {cur_newtar_name}")
        cur_newtar_p = output_data_dir / cur_newtar_name
        if cur_newtar_p.exists():
            with proc_lock:
                logger.error(f"before process with rank {proc_rank} rearchiving the new tar file {cur_newtar_name}, "
                             f"this file has already existed which is wrong")
            sys.exit(1)
        with tarfile.open(output_data_dir / cur_newtar_name, mode="w", encoding="utf-8") as newtar_fp:
            name_prefix = cur_newtar_name.split(".")[0]
            while cur_newtar_imgnum < datanum_per_tar:
                try:
                    native_imgname = next(cur_native_imgnames)
                    # extract and add current image file
                    native_img_bytes = cur_native_tar_fp.extractfile(native_imgname).read()
                    cur_newtar_imgname = f"{name_prefix}{int(cur_newtar_imgnum):04d}.jpg"
                    cur_newtar_imgobj = TarInfo(cur_newtar_imgname)
                    cur_newtar_imgobj.size = len(native_img_bytes)
                    cur_newtar_imgobj.mtime = int(time.time())
                    newtar_fp.addfile(cur_newtar_imgobj, io.BytesIO(native_img_bytes))
                    added_imgnum += 1
                    cur_newtar_imgnum += 1

                    # extract and add corresponding txt file
                    native_txtname = native_imgname.split(".")[0] + ".txt"
                    native_txt_bytes = cur_native_tar_fp.extractfile(native_txtname).read()
                    cur_newtar_txtname = cur_newtar_imgname.split(".")[0] + ".txt"
                    cur_newtar_txtobj = TarInfo(cur_newtar_txtname)
                    cur_newtar_txtobj.size = len(native_txt_bytes)
                    cur_newtar_txtobj.mtime = int(time.time())
                    newtar_fp.addfile(cur_newtar_txtobj, io.BytesIO(native_txt_bytes))

                except StopIteration as _:
                    cur_native_tar_fp.close(); cur_native_tar_fp = None
                    try:
                        cur_native_tar_p = next(native_tars_generator)
                        with proc_lock:
                            logger.info(f"process with rank {proc_rank} rearchive native tar file {cur_native_tar_p.name} into new tar file {cur_newtar_name}")
                    except StopIteration as _:
                        cur_native_tar_p = None
                        if cur_native_tar_fp is not None:
                            cur_native_tar_fp.close()
                        cur_native_tar_fp = None
                        cur_native_imgnames = None
                        break
                    gc.collect()
                    cur_native_tar_fp = tarfile.open(cur_native_tar_p, mode="r", encoding="utf-8")
                    cur_native_imgnames = (mem_name for mem_name in cur_native_tar_fp.getnames() if mem_name.endswith(".jpg"))
        with proc_lock:
            logger.info(f"process with rank {proc_rank} ends renaming and rearchiving this new tar file: {cur_newtar_name}, "
                        f"and rearchive {cur_newtar_imgnum} image-alttext pairs in total")
        newtar_num -= 1
        cur_newtar_name = f"{int(int(cur_newtar_name.split('.')[0]) + 1):05d}.tar"
        cur_newtar_imgnum = 0
    if added_imgnum != imgnum:
        with proc_lock:
            logger.error(f"process with rank {proc_rank} should extract then add {imgnum} image-alttext pairs data, "
                         f"but it handled {added_imgnum}, which is wrong")
    else:
        with proc_lock:
            logger.info(f"process with rank {proc_rank} has finished renaming and rearchiving all tars and images dispendid to it, {added_imgnum} in total")
    if cur_native_tar_fp is not None:
        cur_native_tar_fp.close()
    proc_barrier.wait()
    return


def args_save(args: Namespace, save_p: PosixPath):
    args_dict = vars(args)
    for arg_name, arg_val in args_dict.items():
        if isinstance(arg_val, PosixPath):
            args_dict[arg_name] = str(arg_val)
    with open(save_p, mode="w", encoding="utf-8") as args_save_fp:
        json.dump(args_dict, args_save_fp, ensure_ascii=False)


def main():
    args = parse_args()
    init_logger(args.taisu2_output_dir, logging_level=args.logging_level)
    args_set_and_check(args=args)
    global logger, date_fmt
    st_time = datetime.now()
    # start renaming and rearchiving image-alttext data of Taisu2
    logger.info(f"start renaming and rearchiving of pre-processed Taisu2 image-alttext pairs at {datetime.strftime(st_time, date_fmt)}")
    if args.max_workers > math.floor(args.native_data_num / args.data_num_per_tar):
        tmp_tars_num = math.floor(args.native_data_num / args.data_num_per_tar)
        logger.warning(f"new tar files number: {tmp_tars_num} or {tmp_tars_num + 1}, but maximum workers: {args.max_workers}, "
                       f"hence set maximum workers to {tmp_tars_num}")
        args.max_workers = tmp_tars_num
    with multiprocessing.Manager() as proc_manager:
        global pid_to_rank, proc_lock, proc_barrier
        pid_to_rank = proc_manager.dict()
        proc_lock = proc_manager.Lock()
        proc_barrier = proc_manager.Barrier(args.max_workers)

        partial_task_func = partial(rename_and_rearchive_task_func, args=vars(args))
        generator_func = rename_and_rearchive_generator(
                                                        native_data_dir=args.taisu2_specific_dir, 
                                                        native_datanum=args.native_data_num, 
                                                        datanum_per_tar=args.data_num_per_tar, 
                                                        num_workers=args.max_workers
                                                       )
        with futures.ProcessPoolExecutor(args.max_workers, initializer=proc_init_func) as mp_exec:
            _ = mp_exec.map(partial_task_func, generator_func, chunksize=1)
        gc.collect()

    ed_time = datetime.now()
    whole_secs = (ed_time - st_time).total_seconds()
    logger.info(f"end renaming and rearchiving of pre-processed Taisu2 image-alttext pairs at {datetime.strftime(ed_time, date_fmt)}, and has spent "
                f"{whole_secs / 60:.3f} minutes in total")

    # command line arguments saving
    args_save_p = args.taisu2_output_dir / "arguments.json"
    args_save(args, args_save_p)

    return


if __name__ == "__main__":
    main()
