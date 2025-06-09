# archiving txt files into archive tars
import io, os
import sys, json
import shutil
import tarfile
from tarfile import TarFile, TarInfo
from datetime import datetime
import argparse
from argparse import Namespace, ArgumentError
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL
from concurrent import futures as futures
import multiprocessing
from pathlib import PosixPath, Path


logger: logging.Logger = None


def parse_args():
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
    logger.info(f"parallel process number for image-alttext pairs counting: {args.num_cnt_proc}")
    logger.info(f"parallel process number for txt -> tar archiving: {args.num_archive_proc}")

    return None


def recaption_json_read(output_dir: PosixPath = None, recaption_idx: int = None):
    json_p: PosixPath = output_dir / f"{recaption_idx}th_recaption.json"
    if not json_p.exists():
        raise FileNotFoundError(f"recaption json file - {json_p.name}, cannot be found under diretory - {json_p.parent}!")
    read_st_time = datetime.now()
    with open(json_p, mode="r", encoding="ut-8") as json_fp:
        recaption_res = json.load(json_fp)
    read_ed_time = datetime.now()
    read_secs = (read_ed_time - read_st_time).total_seconds()
    logger.info(f"reading json file {json_p.name} from directory {json_p.parent} has taken {read_secs // 60} minutes, and {read_secs % 60} seconds")
    return recaption_res


def main():
    args = parse_args()
    init_logger(output_dir=args.output_dir, logging_level=args.logging_level)
    global logger
    st_time = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    logger.info(f"begin archiving txt files into tar files at {st_time}")

    args_log(args=args)
    recaption_res = recaption_json_read(output_dir=args.output_dir, recaption_idx=args.recaption_idx)
    with multiprocessing.Manager() as mp_manager:
        recaption_res_list = mp_manager.list()
        # TODO: Now here
        pass

    ed_time = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    logger.info(f"end archiving txt files into tar files at {ed_time}")


if __name__ == "__main__":
    main()
