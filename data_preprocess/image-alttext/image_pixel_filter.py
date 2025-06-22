# filter iamges which have extreme large/tiny width or height, 
# based on the dynamic image resolution strategy used in VLM training
import os, io, sys
import gc
import time
import json
import tarfile
from tarfile import TarInfo
import shutil
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL
from datetime import datetime
import multiprocessing
from multiprocessing.managers import DictProxy, AcquirerProxy
from multiprocessing.synchronize import Barrier
from multiprocessing.sharedctypes import Synchronized
from concurrent import futures as futures
from functools import partial
from typing import List, Dict
import argparse
from argparse import Namespace
from pathlib import Path, PosixPath
from PIL import Image


logger: logging.Logger = None
pid_to_rank: DictProxy = None
proc_lock: AcquirerProxy = None
proc_barrier: Barrier = None

mp_preserve: Synchronized = None
mp_filter: Synchronized = None
mp_filter_min: Synchronized = None
mp_filter_max: Synchronized = None

date_fmt = '%Y-%m-%d-%H:%M:%S'


def parse_args():
    parser = argparse.ArgumentParser(description="filter images with extreme large or small width/height")
    parser.add_argument("--raw-data-folder", type=str, default=None, help="the raw image-alttext data folder")
    parser.add_argument("--specific-data-folder", type=str, default=None, nargs="+", help="sub-folder/subset of raw image-alttext data")
    parser.add_argument("--filter-res-folder", type=str, default="images_pixel_filter", help="the pixel filtering results folder")
    parser.add_argument("--visual-input-size", type=int, default=448, help="vision encoder's input size of VLM")
    parser.add_argument("--min-img-split", type=int, default=1, help="raw image split minimum number of VLM dynamic image resolution")
    parser.add_argument("--max-img-split", type=int, default=12, help="raw image split maximum number of VLM dynamic image resolution")
    parser.add_argument("--max-workers", type=int, default=80, help="maximum parallel workers number for process pool")
    parser.add_argument("--logging-level", type=int, default=DEBUG, choices=(NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL), help="logging output level for logger")
    args = parser.parse_args()

    # directories & paths checking
    if args.raw_data_folder is None:
        raise argparse.ArgumentError(f"raw data folder should get a passed in parameter, but get None")
    args.raw_data_dir = Path(os.getenv("HOME", "")) / "datasets" / "Taisu2_datasets" / f"{args.raw_data_folder}"
    if not args.raw_data_dir.exists():
        raise FileNotFoundError(f"the raw image-alttext data directory - {args.raw_data_dir}, does not exist!")
    args.specific_data_folder = "/".join(args.specific_data_folder)
    args.specific_data_dir = args.raw_data_dir / args.specific_data_folder
    if not args.specific_data_dir.exists():
        raise FileNotFoundError(f"the specific image-alttext data directory - {args.specific_data_dir}, does not exist!")
    args.filter_res_folder = (args.filter_res_folder 
                              + f"_{args.visual_input_size}x{args.visual_input_size}" 
                              + f"_split_{args.min_img_split}_{args.max_img_split}")
    args.filter_res_dir = args.raw_data_dir / args.filter_res_folder
    args.filter_data_dir = args.filter_res_dir / "image-text-pairs"
    if args.filter_data_dir.exists():
        shutil.rmtree(args.filter_data_dir)
        shutil.rmtree(args.filter_res_dir)
    else:
        args.filter_data_dir.mkdir(parents=True, exist_ok=False)

    # args.min_pixels = args.min_img_split * (args.visual_input_size ** 2)  # not rigorous
    # args.max_pixels = args.max_img_split * (args.visual_input_size ** 2)  # not rigorous
    args.min_pixels = args.min_img_split * args.visual_input_size
    args.max_pixels = args.max_img_split * args.visual_input_size
    return args


def init_logger(filter_res_dir: PosixPath, logging_level: int = NOTSET):
    global logger
    logger = logging.getLogger("image-altext data pixel filtering")
    logger.setLevel(logging_level)
    formatter = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")

    stream_hndlr = logging.StreamHandler()
    stream_hndlr.setLevel(logging_level); stream_hndlr.setFormatter(formatter)
    logp = filter_res_dir / "images_pixel_filter.log"
    file_hndlr = logging.FileHandler(logp)
    file_hndlr.setLevel(logging_level); file_hndlr.setFormatter(formatter)
    logger.addHandler(stream_hndlr); logger.addHandler(file_hndlr)
    return


def get_total_tars_num(tar_files_dir: PosixPath):
    tar_generator = tar_files_dir.glob("*.tar")
    tars_num = 0
    while True:
        try:
            _ = next(tar_generator)
            tars_num += 1
        except StopIteration as _:
            break
    return tars_num


def args_set_and_log(args: Namespace):
    global logger
    logger.info(f"raw image-alltext folder argument (args.raw_data_folder): {args.raw_data_folder}")
    logger.info(f"specific processed image-alttext folder argument (args.specific_data_folder): {args.specific_data_folder}")
    logger.info(f"image pixel filtering result folder argument (args.filter_res_folder): {args.filter_res_folder}")
    logger.info(f"visual encoder of VLM input size argument (args.visual_input_size): {args.visual_input_size}x{args.visual_input_size}")
    logger.info(f"minimum image split of dynamic resolution argument (args.min_img_split): {args.min_img_split}")
    logger.info(f"maximum image split of dynamic resolution argument (args.max_img_split): {args.max_img_split}")
    logger.info(f"maximum workers for pixel filtering process pool argument (args.max_workers): {args.max_workers}")
    args.total_tars_num = get_total_tars_num(args.specific_data_dir)
    logger.info(f"total tar files number under input tar directory: {args.total_tars_num}")
    return


def worker_init_func(barrier: Barrier = None):
    pid = str(os.getpid())
    global pid_to_rank, proc_lock, logger
    with proc_lock:
        rank = str(len(pid_to_rank))
        pid_to_rank[pid] = rank
    os.environ["RANK"] = rank
    with proc_lock:
        logger.info(f"worker (pid {pid}) has finished initialized and got rank idx `{rank}`")
    global proc_barrier
    proc_barrier = barrier

    global mp_preserve, mp_filter, mp_filter_min, mp_filter_max
    mp_values_none = any([mp_preserve is None, mp_filter is None, mp_filter_min is None, mp_filter_max is None])
    mp_values_synchronized = all([
                                  mp_preserve.__class__ is Synchronized, mp_filter.__class__ is Synchronized, 
                                  mp_filter_min.__class__ is Synchronized, mp_filter_max.__class__ is Synchronized
                                 ])
    mp_values_zero = all([mp_preserve.value == 0, mp_filter.value == 0, mp_filter_min.value == 0, mp_filter_max.value == 0])
    if mp_values_none:
        with proc_lock:
            logger.error(f"one/some multi-process shared value is None for sub-process with rank {rank}, hence exit abnormally")
        sys.exit(1)
    if not mp_values_synchronized:
        with proc_lock:
            logger.error(f"one/some multiprocess shared value isn't instance of `Synchronized` for sub-process with rank {rank}, hence exit abnormally")
        sys.exit(1)
    if not mp_values_zero:
        with proc_lock:
            logger.error(f"one/some multiprocess share value isn't initialized to zero for sub-process with rank {rank}, hence exit abnormally")
        sys.exit(1)

    return


def tar_generator_func(tar_files_dir: PosixPath, total_tars_num: int, num_workers: int):
    tar_generator = tar_files_dir.glob("*.tar")
    tars_num_per_worker = total_tars_num // num_workers
    remained_tars_num = total_tars_num % num_workers
    tars_list_per_worker = []
    while True:
        try:
            if len(tars_list_per_worker) == tars_num_per_worker:
                if remained_tars_num != 0:
                    tars_list_per_worker.append(next(tar_generator))
                    remained_tars_num -= 1
                yield tars_list_per_worker
                tars_list_per_worker = []
            else:
                tars_list_per_worker.append(next(tar_generator))
        except StopIteration:
            break
    if tars_list_per_worker:
        yield tars_list_per_worker
    return


def whether_preserve(imgs_bytes: List[bytes], min_pixels: int = None, max_pixels: int = None):
    for imgbytes in imgs_bytes:
        img = Image.open(io.BytesIO(imgbytes))
        # img_pixels = img.height * img.width
        imgh = img.height
        imgw = img.width
        # if img_pixels < min_pixels or img_pixels > max_pixels:
        #     if img_pixels < min_pixels:
        #         yield (False, "min")
        #     else:
        #         yield (False, "max")
        # else:
        #     yield True
        if imgh < min_pixels or imgw < min_pixels:
            yield (False, "min")
        elif imgh > max_pixels or imgw > max_pixels:
            yield (False, "max")
        else:
            yield True
    return


def pixel_filter_task_func(tars_list: List[PosixPath], args: Dict):
    global logger, proc_lock, proc_barrier, date_fmt
    global mp_preserve, mp_filter, mp_filter_min, mp_filter_max
    proc_rank = os.getenv("RANK", None)

    st_time = datetime.now()
    with proc_lock:
        logger.info(f"process with rank {proc_rank} has got {len(tars_list)} tar files and starts filtering based on image pixles at "
                    f"{datetime.strftime(st_time, date_fmt)}")
    filter_total = 0
    filter_min_total = 0
    filter_max_total = 0
    preserve_total = 0
    filter_res_p = args["filter_res_dir"] / "tarname_to_filters.jsonl"

    for tar_p in tars_list:
        output_p = args["filter_data_dir"] / tar_p.name
        filter_pertar = 0
        filter_min_pertar = 0
        filter_max_pertar = 0
        preserve_pertar = 0
        filter_imgnames_pertar = set()
        with proc_lock:
            logger.info(f"process with rank {proc_rank} begins filtering tar file {tar_p.name}")
        try:
            with tarfile.open(tar_p, mode="r", encoding="utf-8") as tar_fp:
                imgnames = [mem_name for mem_name in tar_fp.getnames() if mem_name.endswith(".jpg")]
                imgs_bytes = [tar_fp.extractfile(imgname).read() for imgname in imgnames]
                preserve_generator = whether_preserve(imgs_bytes, min_pixels=args["min_pixels"], max_pixels=args["max_pixels"])
                txtnames = [mem_name for mem_name in tar_fp.getnames() if mem_name.endswith(".txt")]
                txts_bytes = [tar_fp.extractfile(txtname).read() for txtname in txtnames]
                with tarfile.open(output_p, mode="w", encoding="utf-8") as outtar_fp:
                    for idx, (imgbytes, preserve) in enumerate(zip(imgs_bytes, preserve_generator)):
                        if isinstance(preserve, bool) and preserve:
                            imgname = imgnames[idx]
                            img_tarinfo = TarInfo(imgname)
                            img_tarinfo.size = len(imgbytes); img_tarinfo.mtime = int(time.time())
                            outtar_fp.addfile(img_tarinfo, io.BytesIO(imgbytes))

                            txtbytes = txts_bytes[idx]
                            txtname = txtnames[idx]
                            if txtname.split(".", maxsplit=1)[0] != imgname.split(".", maxsplit=1)[0]:
                                with proc_lock:
                                    logger.error(f"image name: {imgname}, while correspond txt name: {txtname}")
                                sys.exit(1)
                            txt_tarinfo = TarInfo(txtname)
                            txt_tarinfo.size = len(txtbytes); txt_tarinfo.mtime = int(time.time())
                            outtar_fp.addfile(txt_tarinfo, io.BytesIO(txtbytes))

                            preserve_pertar += 1
                        else:
                            filter_pertar += 1
                            filter_imgnames_pertar.add(imgnames[idx])
                            if preserve[1] == "min":
                                filter_min_pertar += 1
                            else:
                                filter_max_pertar += 1
        except Exception as filter_err:
            with proc_lock:
                logger.error(f"when filtering images in tar file {tar_p.name} based on dynamic image resolution strategy, "
                             f"encounter an error ({type(filter_err)}): {filter_err}")
            sys.exit(1)
        with proc_lock:
            logger.info(f"process with rank {proc_rank} has ended filtering tar file {tar_p.name}, for tar file {tar_p.name}, "
                        f"{filter_pertar} images are filtered ({filter_min_pertar} smaller than low bound of image pixel, and "
                        f"{filter_max_pertar} greater than up bound of image pixel); {preserve_pertar} images are preserved")
            with open(filter_res_p, mode="a", encoding="utf-8") as filter_res_fp:
                json.dump({tar_p.name: list(filter_imgnames_pertar)}, filter_res_fp, ensure_ascii=False)
                filter_res_fp.write("\n")
        preserve_total += preserve_pertar
        filter_total += filter_pertar
        filter_min_total += filter_min_pertar
        filter_max_total += filter_max_pertar

    ed_time = datetime.now()
    interval_secs = (ed_time - st_time).total_seconds()
    with proc_lock:
        logger.info(f"process with rank {proc_rank} has finished all {len(tars_list)} tar files filtering at {datetime.strftime(ed_time, date_fmt)}, "
                    f"takes {int(interval_secs // 3600):d} hours and {((interval_secs % 3600) / 60):.3f} minutes in total. "
                    f"{filter_total} images are filtered ({filter_min_total} smaller than low bound, and {filter_max_total} greater than up bound), "
                    f"{preserve_total} images are preserved")

    with mp_preserve.get_lock():
        mp_preserve.value += preserve_total
    with mp_filter.get_lock():
        mp_filter.value += filter_total
    with mp_filter_min.get_lock():
        mp_filter_min.value += filter_min_total
    with mp_filter_max.get_lock():
        mp_filter_max.value += filter_max_total

    proc_barrier.wait()
    return


def args_save(args: Namespace, save_dir: PosixPath):
    global logger
    if not save_dir.exists():
        logger.error(f"arguments saving directory: {save_dir}, does not exist!")
        sys.exit(1)
    args_for_saving = {}
    for argname, argval in vars(args).items():
        if not isinstance(argval, PosixPath):
            args_for_saving.update({argname: argval})
        else:
            args_for_saving.update({argname: str(argval)})
    save_p: PosixPath = save_dir / "arguments.json"
    logger.info(f"saving arguments of filtering based on image pixel into {save_p}")
    with open(save_p, mode="w", encoding="utf-8") as save_fp:
        json.dump(args_for_saving, save_fp, ensure_ascii=False)
    logger.info(f"ends saving argument of filtering based on image pixel")

    return


def main():
    args = parse_args()
    init_logger(args.filter_res_dir, args.logging_level)
    args_set_and_log(args)
    global logger, date_fmt

    barrier = multiprocessing.Barrier(args.max_workers)
    partial_pixel_filter_func = partial(pixel_filter_task_func, args=vars(args))
    if args.max_workers > args.total_tars_num:
        logger.warning(f"tar files total number: {args.total_tars_num}, maximum worker: {args.max_workers}, "
                       f"hence set maximum workers to {args.total_tars_num}")
        args.max_workers = args.total_tars_num
    tar_list_generator = tar_generator_func(args.specific_data_dir, args.total_tars_num, args.max_workers)

    global mp_preserve, mp_filter, mp_filter_min, mp_filter_max
    mp_preserve = multiprocessing.Value("i", lock=True)
    mp_filter = multiprocessing.Value("i", lock=True)
    mp_filter_min = multiprocessing.Value("i", lock=True)
    mp_filter_max = multiprocessing.Value("i", lock=True)

    st_time = datetime.now()
    logger.info(f"begin images filtering based on dynamic image resolution at {datetime.strftime(st_time, date_fmt)}")
    with multiprocessing.Manager() as proc_manager:
        global pid_to_rank, proc_lock
        pid_to_rank = proc_manager.dict()
        proc_lock = proc_manager.Lock()
        with futures.ProcessPoolExecutor(args.max_workers, initializer=worker_init_func, initargs=(barrier, )) as filter_exec:
            _ = filter_exec.map(partial_pixel_filter_func, tar_list_generator, chunksize=1)
        gc.collect()
    ed_time = datetime.now()
    interval_secs = (ed_time - st_time).total_seconds()
    logger.info(f"end images filtering base on dynamic image resolution strategy at {datetime.strftime(ed_time, date_fmt)}, taking "
                f"{int((interval_secs // 3600)):d} hours and {((interval_secs % 3600) / 60):.3f} minutes in total\n\n")

    preserve_num = mp_preserve.value
    filter_total_num = mp_filter.value
    filter_min_num = mp_filter_min.value
    filter_max_num = mp_filter_max.value
    del mp_preserve, mp_filter, mp_filter_min, mp_filter_max
    gc.collect()
    if filter_min_num + filter_max_num != filter_total_num:
        logger.error(f"total images which have been filtered: {filter_total_num}, while filtered images number lower than minimum pixel bound: {filter_min_num}; "
                     f"filtered images number greater than maximum pixle bound: {filter_max_num} (summed up to{filter_min_num + filter_max_num})")
        sys.exit(1)
    logger.info(f"statistics:")
    logger.info(f"native image-alttext pairs number: {preserve_num + filter_total_num}")
    logger.info(f"preserved image-alttext pairs number: {preserve_num}")
    logger.info(f"number of filtered image-alttext pairs of which image resolution is lower than the minimum pixel bound: {filter_min_num} "
                f"(hold total filter images num {(filter_min_num / filter_total_num * 100):.3f} % percent)")
    logger.info(f"number of filtered image-alttext pairs of which image resolution is greater than the maximum pixel bound: {filter_max_num} "
                f"(hold total filter images num {(filter_max_num / filter_total_num * 100):.3f} % percent)\n\n")

    args_save(args=args, save_dir=args.filter_res_dir)
    return


if __name__ == "__main__":
    main()
