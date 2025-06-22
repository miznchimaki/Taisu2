import io, os
import gc
import sys
import json
import time
import math
import warnings
import shutil
import random
import tarfile
from tarfile import TarInfo
from datetime import datetime
import logging
from logging import Logger

from pathlib import PurePath, Path, PosixPath
from typing import Tuple, Dict, List, Optional, Union

import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.managers import AcquirerProxy, DictProxy, ListProxy
from multiprocessing.synchronize import Barrier
from concurrent import futures as futures
from functools import partial
import numpy as np
import pywt
from scipy.fftpack import dct

from imagededup.handlers.search.retrieval import HashEval
from imagededup.utils.general_utils import (
    get_files_to_remove,
    save_json,
    parallelise,
    generate_files,
    generate_relative_names
)
from imagededup.utils.image_utils import load_image, preprocess_image, check_image_array_hash
from imagededup.utils.logger import return_logger


logger = return_logger(__name__)
worker_lock: AcquirerProxy = None
worker_barrier: Barrier = None
worker_logger: Logger = None
date_fmt = '%Y-%m-%d-%H:%M:%S'


"""
TODO:
Wavelet hash: Zero the LL coeff, reconstruct image, then get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

"""


class Hashing:
    """
    Find duplicates using hashing algorithms and/or generate hashes given a single image or a directory of images.

    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encoding generation:
    To generate hashes using specific hashing method. The generated hashes can be used at a later time for
    deduplication. Using the method 'encode_image' from the specific hashing method object, the hash for a
    single image can be obtained while the 'encode_images' method can be used to get hashes for all images in a
    directory.

    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplicates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        self.target_size = (8, 8)  # resizing to dims
        self.verbose = verbose

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.

        Args:
            hash1: hash string
            hash2: hash string

        Returns:
            hamming_distance: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(
            64
        )  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    @staticmethod
    def _array_to_hash(hash_mat: np.ndarray) -> str:
        """
        Convert a matrix of binary numerals to 64 character hash.

        Args:
            hash_mat: A numpy array consisting of 0/1 values.

        Returns:
            An hexadecimal hash string.
        """
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(
        self, image_file=None, image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate hash for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            hash: A 16 character hexadecimal string hash for the image.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        myhash = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        myhash = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True
                )

            elif isinstance(image_array, np.ndarray):
                check_image_array_hash(image_array)  # Do sanity checks on array
                image_pp = preprocess_image(
                    image=image_array, target_size=self.target_size, grayscale=True
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir=None, recursive: bool = False, num_enc_workers: int = cpu_count()):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        files = generate_files(image_dir, recursive)

        logger.info(f'Start: Calculating hashes...')

        hashes = parallelise(function=self.encode_image, data=files, verbose=self.verbose, num_workers=num_enc_workers)
        hash_initial_dict = dict(zip(generate_relative_names(image_dir, files), hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        logger.info(f'End: Calculating hashes!')
        return hash_dict

    @staticmethod
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

    @staticmethod
    def tar_generator_func(tar_files_dir: PosixPath, total_tars_num: int, num_enc_workers: int):
        tar_generator = tar_files_dir.glob("*.tar")
        tars_num_per_worker = total_tars_num // num_enc_workers
        remained_tars_num = total_tars_num % num_enc_workers
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

    @staticmethod
    def worker_init_func(pid_to_rank: DictProxy, lock: AcquirerProxy, barrier: Barrier, logger: Logger):
        pid = str(os.getpid())
        global worker_lock, worker_logger
        worker_lock = lock; worker_logger = logger
        with worker_lock:
            worker_rank = str(len(pid_to_rank))
            pid_to_rank[pid] = worker_rank
        os.environ["RANK"] = worker_rank
        with worker_lock:
            worker_logger.info(f"worker (pid {pid}) has finished initialized and got rank idx `{worker_rank}`")
        global worker_barrier
        worker_barrier = barrier

    @staticmethod
    def dedup_logger_init(raw_data_dir: str, task_name: str):
        global logger; logger = None
        logger = logging.getLogger(f"Taisu2 raw data {task_name.split('_')[0]} {task_name.split('_', maxsplit=1)[1]}")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")
        stream_hndlr = logging.StreamHandler(); stream_hndlr.setLevel(logging.DEBUG); stream_hndlr.setFormatter(formatter)
        logger.addHandler(stream_hndlr)
    
        log_p = Path(raw_data_dir) / f"{task_name}" / f"{task_name}.log"
        log_p.parent.mkdir(parents=True, exist_ok=True); log_p.touch(exist_ok=False)
        file_hndlr = logging.FileHandler(log_p); file_hndlr.setLevel(logging.DEBUG); file_hndlr.setFormatter(formatter)
        logger.addHandler(file_hndlr)
    
        return

    @staticmethod
    def no_exist_path(tars_path: List[PosixPath]):
        no_exist = set()
        for tarp in tars_path:
            if not tarp.exists():
                no_exist.add(tarp)
        return no_exist

    def images_hash_task_func(self, tars_path_list: List[PosixPath], output_dir: PosixPath = None):
        global worker_lock, worker_barrier, worker_logger
        worker_rank = os.getenv("RANK", None)
        if worker_rank is None:
            with worker_lock:
                worker_logger.error(f"process with pid {os.getpid()} for images hash encoding cannot get environmental variable named `RANK`")
            sys.exit(1)
        with worker_lock:
            worker_logger.info(f"process with rank {worker_rank} get {len(tars_path_list)} tar files to execute images hash encoding")
        for tar_p in tars_path_list:
            if not tar_p.exists():
                with worker_lock:
                    worker_logger.error(f"process with rank {worker_rank} gets a tar file path - {tar_p}, which does not exist!")
                sys.exit(1)
            tar_name = tar_p.name
            with worker_lock:
                worker_logger.info(f"process with rank {worker_rank} hash encodes images in tar file {tar_name}")
            result_tar_p = output_dir / tar_name
            try:
                with tarfile.open(tar_p, "r", encoding="utf-8") as tar_fp:
                    img_names = [mem_name for mem_name in tar_fp.getnames() if mem_name.endswith(".jpg")]
                    img_bytes = [io.BytesIO(tar_fp.extractfile(img_name).read()) for img_name in img_names]
                    img_hashes = [self._hash_func(load_image(per_img_bytes, target_size=self.target_size, grayscale=True)) for per_img_bytes in img_bytes]
                    img_names_to_hashes = dict(zip(img_names, img_hashes))
            except Exception as hash_err:
                with worker_lock:
                    worker_logger.error(f"when process with rank {worker_rank} hash encodes images in tar file {tar_name}, "
                                        f"encounter an error ({type(hash_err)}): {hash_err}")
                sys.exit(1)
            with worker_lock:
                worker_logger.info(f"process with rank {worker_rank} has finished encoding images into hashes, "
                                   f"then begins writing hash results into path {result_tar_p}")
            try:
                with tarfile.open(result_tar_p, "w", encoding="utf-8") as result_tar_fp:
                    for img_name, img_hash in img_names_to_hashes.items():
                        tarinfo_name = img_name.split(".")[0] + ".txt"
                        img_hash_tarinfo = TarInfo(tarinfo_name)
                        hash_bytes = img_hash.encode(encoding="utf-8")
                        hash_bytes_obj = io.BytesIO(hash_bytes)
                        img_hash_tarinfo.size = len(hash_bytes)
                        img_hash_tarinfo.mtime = int(time.time())
                        result_tar_fp.addfile(img_hash_tarinfo, hash_bytes_obj)
            except Exception as write_err:
                with worker_lock:
                    worker_logger.error(f"when process with rank {worker_rank} saving hash results into path {result_tar_p}, "
                                        f"encounter an error ({type(write_err)}): {write_err}")
                sys.exit(1)
            with worker_lock:
                worker_logger.info(f"process with rank {worker_rank} hash finished writing hash results into path {result_tar_p}")

        with worker_lock:
            worker_logger.info(f"process with rank {worker_rank} has finished all {len(tars_path_list)} tar files hash encoding and result saving")
        worker_barrier.wait()
        return

    def encode_images_in_tars(self, data_dir: str = None, num_enc_workers: int = os.cpu_count()):
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"the native dataset directory - {data_dir} does not exist!")
        self.dedup_logger_init(data_dir, "images_hash")
        global logger
        tars_dir = Path(data_dir) / "image-text-pairs"
        if not tars_dir.exists():
            raise FileNotFoundError(f"the image tar files directory - {tars_dir} does not exist!")
        logger.info(f"using Python3 library `imagededup` to execute image hashing encoding, encoding method: {self.__class__.__name__}")
        logger.info(f"images native directory: {data_dir}")
        logger.info(f"tar files native folder: image-text-pairs")
        hash_res_dir = Path(data_dir) / "images_hash" / "image-text-pairs"
        hash_res_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"images hash encoding output directory: {hash_res_dir}")

        total_tars_num = self.get_total_tars_num(tars_dir)
        logger.info(f"tar files number for hash encoding: {total_tars_num}")
        if num_enc_workers <= total_tars_num:
            logger.info(f"parallel hash encoding workers: {num_enc_workers}")
        else:
            num_enc_workers = total_tars_num
            logger.warning(f"tar files number for hashing: {total_tars_num}, but hashing workers: {num_enc_workers}, "
                           f"which is greater than tar files number, hence set hashing workers number to {total_tars_num}")

        logger.info(f"Start images hash encoding in tar files via multi-process pool at {datetime.strftime(datetime.now(), date_fmt)}")
        with multiprocessing.Manager() as hash_manager:
            hash_pid_to_rank = hash_manager.dict()
            hash_lock = hash_manager.Lock()
            hash_barrier = hash_manager.Barrier(num_enc_workers)
            partial_images_hash_func = partial(self.images_hash_task_func, output_dir=hash_res_dir)
            with futures.ProcessPoolExecutor(num_enc_workers, initializer=self.worker_init_func, initargs=(hash_pid_to_rank, hash_lock, hash_barrier, logger)) as hash_exec:
                _ = hash_exec.map(partial_images_hash_func, self.tar_generator_func(tars_dir, total_tars_num, num_enc_workers), chunksize=1)
        logger.info(f"End images hash encoding in tar files via multi-process pool at {datetime.strftime(datetime.now(), date_fmt)}")

        # images hash arguments saving
        save_args = {"raw_data_dir": data_dir, "hash_method": self.__class__.__name__, "total_tars_num": total_tars_num, 
                     "images_hash_workers": num_enc_workers, "hash_result_dir": str(hash_res_dir)}
        with open(hash_res_dir.parent / "arguments.json", mode="w", encoding="utf-8") as args_fp:
            json.dump(save_args, args_fp, ensure_ascii=False)
        logger.info(f"save image hashing arguments into path: {hash_res_dir.parent / 'arguments.json'}")

        return

    def hashtars_read_task_func(self, hashtars_p: List[PosixPath], name_to_hash: DictProxy = None):
        """
        Take in a list contained path of image hash tar files, read the images hash txt files from all 
        image hash tar file path, then save image names and corresponding hash encodings into the shared hash dictoinary

        Args:
            hashtars_p: List[PosixPath], list of hash tar files path for a worker in process pool to handle.
            name_to_hash: multiprocessing.manager.DictProxy, process-shared dictionary for saving image hash results.
        """
        global worker_lock, worker_barrier, worker_logger
        if os.getenv("RANK", None) is None:
            with worker_lock:
                worker_logger.error(f"process with pid {os.getpid()} for hash tar files dedup cannot get the environmental variable named `RANK`")
            sys.exit(1)
        worker_rank = os.getenv("RANK", None)

        no_exist = self.no_exist_path(hashtars_p)
        if no_exist:
            with worker_lock:
                worker_logger.error(f"process with rank {worker_rank} gets some hash tar files which don't exist: {no_exist}")
            sys.exit(1)
        del no_exist

        with worker_lock:
            worker_logger.info(f"process with rank {worker_rank} has got {len(hashtars_p)} tar files to read images hash from them")
        imgnames_to_hash = {}
        for tar_p in hashtars_p:
            with worker_lock:
                worker_logger.info(f"process with rank {worker_rank} reads image hashes from tar file {tar_p.name}")
            try:
                with tarfile.open(tar_p, mode="r", encoding="utf-8") as hashtar_fp:
                    hash_mem_names = hashtar_fp.getnames()
                    imgnames = (mem_name.split(".")[0] + ".jpg" for mem_name in hash_mem_names)
                    hash_strs = (hashtar_fp.extractfile(mem_name).read().decode(encoding="utf-8") for mem_name in hash_mem_names)
                    hashtar_res = zip(imgnames, hash_strs)
                    imgnames_to_hash.update(hashtar_res)
            except Exception as tarread_err:
                with worker_lock:
                    worker_logger.error(f"when process with rank {worker_rank} reading image hashes from tar file {tar_p.name}, "
                                        f"encounter an error ({type(tarread_err)}): {tarread_err}")
                sys.exit(1)
            with worker_lock:
                worker_logger.info(f"process with rank {worker_rank} has finished reading image hashed from tar file {tar_p.name}")
        with worker_lock:
            name_to_hash.update(imgnames_to_hash)
            worker_logger.info(f"process with rank {worker_rank} has written its read image hashes into multi-process shared dictionary")
            worker_logger.info(f"process with rank {worker_rank} has finished reading image hashes from all received hash tar files")

        worker_barrier.wait()
        return

    def tarimages_dedup_task_func(self, tars_list: List[PosixPath], dupimgs_shared_list: ListProxy, output_dir: PosixPath):
        global worker_lock, worker_barrier, worker_logger
        worker_rank = os.getenv("RANK", None)
        if worker_rank is None:
            with worker_lock:
                worker_logger.error(f"")
            sys.exit(1)
        with worker_lock:
            worker_logger.info(f"process with rank {worker_rank} has got {len(tars_list)} image tar files to execute deduplication")

        no_exist = self.no_exist_path(tars_list)
        if no_exist:
            with worker_lock:
                worker_logger.error(f"process with rank {worker_rank} has got tar files path that don't exist: {no_exist}")
            sys.exit(1)

        with worker_lock:
            worker_logger.info(f"process with rank {worker_rank} begins executing image deduplication")
        dedup_total_num = 0
        remained_total_num = 0
        for tar_p in tars_list:
            dedup_num_pertar = 0
            remained_num_pertar = 0
            tar_name = tar_p.name
            out_tar_p = output_dir / tar_name
            if out_tar_p.exists():
                out_tar_p.unlink(missing_ok=False)
            try:
                with tarfile.open(out_tar_p, mode="w", encoding="utf-8") as out_tarfp:
                    with tarfile.open(tar_p, mode="r", encoding="utf-8") as origin_tarfp:
                        origin_imgnames = [mem_name for mem_name in origin_tarfp.getnames() if mem_name.endswith(".jpg")]
                        origin_txtnames = [mem_name for mem_name in origin_tarfp.getnames() if mem_name.endswith(".txt")]
                        for imgname in origin_imgnames:
                            if imgname in dupimgs_shared_list:
                                dedup_num_pertar += 1
                            else:
                                remained_num_pertar += 1
                                txtname = imgname.split(".", maxsplit=1)[0] + ".txt"
                                if txtname not in origin_txtnames:
                                    with worker_lock:
                                        worker_logger.error(f"when doing deduplication, image {imgname} in tar file {tar_name} "
                                                            f"does not have its corresponding txt file {txtname}")
                                    sys.exit(1)

                                img_tarinfo = TarInfo(imgname)
                                img_bytes = origin_tarfp.extractfile(imgname).read()
                                img_tarinfo.size = len(img_bytes); img_tarinfo.mtime = int(time.time())
                                img_bytes_obj = io.BytesIO(img_bytes)
                                out_tarfp.addfile(img_tarinfo, img_bytes_obj)

                                txt_tarinfo = TarInfo(txtname)
                                txt_bytes = origin_tarfp.extractfile(txtname).read()
                                txt_tarinfo.size = len(txt_bytes); txt_tarinfo.mtime = int(time.time())
                                txt_bytes_obj = io.BytesIO(txt_bytes)
                                out_tarfp.addfile(txt_tarinfo, txt_bytes_obj)
            except Exception as dedup_err:
                with worker_lock:
                    worker_logger.error(f"when process with rank {worker_rank} dedup tar file {tar_name}, "
                                        f"encounter an error ({type(dedup_err)}): {dedup_err}")
                sys.exit(1)
            dedup_total_num += dedup_num_pertar
            remained_total_num += remained_num_pertar
            with worker_lock:
                worker_logger.info(f"process with rank {worker_rank} has finished deduping tar file {tar_name}, "
                                   f"{dedup_num_pertar} images are deduped and {remained_num_pertar} images are remained")
        with worker_lock:
            worker_logger.info(f"process with rank {worker_rank} has ended image deduplication, {dedup_total_num} images are deduped and "
                               f"{remained_total_num} images are remained")

        worker_barrier.wait()
        return

    def dedup_images_in_tars(
                             self, 
                             hashtars_dir_str: str, 
                             max_distance_threshold: int = 5, 
                             scores: bool = False, 
                             search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree', 
                             num_tar_workers: int = 8, 
                             num_dist_workers: int = 4, 
                             num_dedup_workers: int = 16
                            ):
        """
        Find duplicated images of al tar files under a specified directory. All images with hamming distance less than or
        equal to the max_distance_threshold are regarded as duplicates.

        Args:
            hash_tars_dir: pathlib.PosixPath, tar files directory, in which each tar file contains hash encodings of some images.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are valid. 
                                    (must be an integer between 0 and 64). Default is 5
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            search_method: Optional, Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            num_tar_workers: Optional, an integer indicating number of workers to read hash tar files in parallel.
            num_dist_workers: Optional, an integer indicating number of workers to find duplications in tars in parallel.
        """
        hashtars_dir = Path(hashtars_dir_str)
        if not hashtars_dir.parent.parent.exists():
            raise FileNotFoundError(f"the raw image-alttext pairs directory - {hashtars_dir.parent.parent} doesn't exist!")
        if hashtars_dir.name != "image-text-pairs":
            raise ValueError(f"outermost path of `hashtars_dir_str` should be `image-text-pairs`, but got {hashtars_dir.name}")
        if hashtars_dir.parent.name != "images_hash":
            raise ValueError(f"the second to last path of `hashtars_dir_str` should be `images_hash`, but got {hashtars_dir.parent.name}")
        if not hashtars_dir.exists():
            raise FileNotFoundError(f"tar files directory string - {hashtars_dir} does not exist")
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)

        self.dedup_logger_init(hashtars_dir.parent.parent, "images_dedup")
        global logger
        logger.info(f"parameter `max_distance_threshold` (int) for images dedup: {max_distance_threshold}")
        logger.info(f"parameter `scores` (bool) for image finding duplications: {scores}")
        logger.info(f"parameter `search_method` (str) for image finding duplications: {search_method}")
        logger.info(f"number of workers for hash tar files reading: {num_tar_workers}")
        logger.info(f"number of workers for image finding duplications: {num_dist_workers}")
        logger.info(f"number of workers for image deduplication: {num_dedup_workers}")

        dupsfind_res_p = hashtars_dir.parent.parent / "images_dedup" / "duplications.jsonl"
        logger.info(f"output path for hash tar files duplications find: {dupsfind_res_p}")

        dedup_res_dir = hashtars_dir.parent.parent / "images_dedup" / "image-text-pairs"
        if dedup_res_dir.exists():
            shutil.rmtree(dedup_res_dir)
        else:
            dedup_res_dir.mkdir(parents=False, exist_ok=False)
        logger.info(f"output directory for images deduplication: {dedup_res_dir}")

        total_num_hashtars = self.get_total_tars_num(hashtars_dir)
        logger.info(f"number of images hash tar files reading in total: {total_num_hashtars}")

        with multiprocessing.Manager() as dedup_manager:
            dedup_pid_to_rank = dedup_manager.dict()
            dedup_lock = dedup_manager.Lock()
            hashtars_read_barrier = dedup_manager.Barrier(num_tar_workers)
            name_to_hash = dedup_manager.dict()

            # (1) read hash tars contents
            logger.info(f"start hash results of tar files reading via multi-process pool at {datetime.strftime(datetime.now(), date_fmt)}")
            hashtars_read_initargs = (dedup_pid_to_rank, dedup_lock, hashtars_read_barrier, logger)
            hashtars_res_func = partial(self.hashtars_read_task_func, name_to_hash=name_to_hash)
            hashtars_generator = self.tar_generator_func(hashtars_dir, total_num_hashtars, num_tar_workers)
            with futures.ProcessPoolExecutor(num_tar_workers, initializer=self.worker_init_func, initargs=hashtars_read_initargs) as hash_tars_read_exec:
                _ = hash_tars_read_exec.map(hashtars_res_func, hashtars_generator, chunksize=1)
            native_imgnum = len(name_to_hash)
            logger.info(f"having read {native_imgnum} images hash from image hash tars directory: {hashtars_dir}")
            logger.info(f"end hash results of tar files reading via multi-process pool at {datetime.strftime(datetime.now(), date_fmt)}\n\n")

            # (2) image duplications finding based on hash
            logger.info(f"start image duplications finding based on hash read from tars at {datetime.strftime(datetime.now(), date_fmt)}")
            gc.collect()
            try:
                dupsfind_obj = HashEval(
                                        test=name_to_hash, queries=name_to_hash, distance_function=self.hamming_distance, 
                                        verbose=self.verbose, threshold=max_distance_threshold, search_method=search_method, 
                                        num_dist_workers=num_dist_workers
                                       )
                dupsfind_res = dupsfind_obj.retrieve_results(scores=scores)
            except Exception as dupsfind_err:
                logger.error(f"when executing duplications finding task, encounter an error ({type(dupsfind_err)}): {dupsfind_err}")
                sys.exit(1)
            name_to_hash.clear(); gc.collect()
            logger.info(f"end image duplications finding based on hash read from tars at {datetime.strftime(datetime.now(), date_fmt)}\n\n")

            # (3) image duplications result saving
            logger.info(f"start saving image duplications finding results (into {dupsfind_res_p}) at {datetime.strftime(datetime.now(), date_fmt)}")
            with open(dupsfind_res_p, mode="w", encoding="utf-8") as dupsfind_fp:
                for imgname, imgdups in dupsfind_res.items():
                    json.dump({imgname: imgdups}, dupsfind_fp, ensure_ascii=False)
                    dupsfind_fp.write("\n")
            dupimgs_set = set()
            for imgdups in dupsfind_res.values():
                if not scores:
                    dupimgs_set.update(imgdups)
                else:
                    imgdups_noscore = (imgdup[0] for imgdup in imgdups)
                    dupimgs_set.update(imgdups_noscore)
            dupimgs_list = dedup_manager.list()
            dupimgs_list.extend(dupimgs_set)
            del dupimgs_set; gc.collect()
            logger.info(f"end saving image duplications finding results (into {dupsfind_res_p}) at {datetime.strftime(datetime.now(), date_fmt)}\n\n")

            # (4) image deduplications
            dedup_pid_to_rank.clear()
            dedup_barrier = dedup_manager.Barrier(num_dedup_workers)
            logger.info(f"start images deduplication via multi-processing pool at {datetime.strftime(datetime.now(), date_fmt)}")
            logger.info(f"native image number: {native_imgnum}, and {len(dupimgs_list)} images are duplicated and will be discarded")
            dedup_initargs = (dedup_pid_to_rank, dedup_lock, dedup_barrier, logger)

            partial_tarimgs_dedup_func = partial(self.tarimages_dedup_task_func, dupimgs_shared_list=dupimgs_list, output_dir=dedup_res_dir)
            imgtars_dir = hashtars_dir.parent.parent / "image-text-pairs"
            total_num_imgtars = self.get_total_tars_num(imgtars_dir)
            imgtars_generator = self.tar_generator_func(imgtars_dir, total_num_imgtars, num_dedup_workers)
            with futures.ProcessPoolExecutor(num_dedup_workers, initializer=self.worker_init_func, initargs=dedup_initargs) as dedup_exec:
                _ = dedup_exec.map(partial_tarimgs_dedup_func, imgtars_generator, chunksize=1)
            logger.info(f"end images deduplication via multi-processing pool at {datetime.strftime(datetime.now(), date_fmt)}\n\n")

        # (5) arguments saving
        save_args = {
                     "hashtars_dir": hashtars_dir_str, "max_distance_threshold": max_distance_threshold, "scores": scores, 
                     "search_method": search_method, "hashtar_reading_workers": num_tar_workers, "dupsfind_workers": num_dist_workers, 
                     "dedup_workers": num_dedup_workers, "dupsfind_result_path": str(dupsfind_res_p), "dedup_result_dir": str(dedup_res_dir)
                    }
        args_save_p = dedup_res_dir.parent / "arguments.json"
        with open(args_save_p, mode="w", encoding="utf-8") as args_save_fp:
            json.dump(save_args, args_save_fp, ensure_ascii=False)
        logger.info(f"having saved duplications finding and deduplication arguments into file {args_save_p}\n\n")

        return

    @staticmethod
    def group_dupsfind_worker_init_func(pid_to_rank: DictProxy, proc_lock: AcquirerProxy, proc_barrier: Barrier, proc_logger: Logger, 
                                        outer_iter_rank: int, inner_iter_rank: int):
        global worker_lock, worker_barrier, worker_logger
        worker_lock = proc_lock
        worker_barrier = proc_barrier
        worker_logger = proc_logger
        proc_id = str(os.getpid())
        with worker_lock:
            proc_rank = str(len(pid_to_rank))
            pid_to_rank[proc_id] = proc_rank
        os.environ["OUTER_ITER_RANK"] = str(outer_iter_rank)
        os.environ["INNER_ITER_RANK"] = str(inner_iter_rank)
        os.environ["RANK"] = proc_rank
        with worker_lock:
            logger.info(f"process with pid {proc_id} initialization gets outer loop iter index {outer_iter_rank}, "
                        f"inner loop iter index {inner_iter_rank}, group index {proc_rank}")

    @staticmethod
    def group_name_to_hash_generator(
                                     name_to_hash: DictProxy = None, 
                                     total_imgs: int = None, 
                                     num_imgs_per_inner_iter: int = None, 
                                     inner_loop_idx: int = None, 
                                     num_imgs_per_grp: int = None, 
                                     last_inner_loop: bool = False, 
                                     num_grp_last_inner_iter: int = None, 
                                     num_imgs_last_inner_iter: int = None
                                    ):
        inner_loop_img_st_idx = num_imgs_per_inner_iter * (inner_loop_idx - 1)
        inner_loop_img_ed_idx = num_imgs_per_inner_iter * inner_loop_idx

        if inner_loop_img_st_idx >= total_imgs:
            return
        if not last_inner_loop:
            group_name_to_hash = dict()
            for idx, (imgname, imghash) in enumerate(name_to_hash.items()):
                if idx < inner_loop_img_st_idx:
                    continue
                elif idx == inner_loop_img_ed_idx:
                    break
                elif (idx != inner_loop_img_st_idx) and ((idx - inner_loop_img_st_idx) % num_imgs_per_grp == 0):
                    if len(group_name_to_hash) != num_imgs_per_grp:
                        raise ValueError(f"for the {inner_loop_idx}th inner loop, and the {(idx - inner_loop_img_st_idx) % num_imgs_per_grp - 1}th group of "
                                         f"the inner loop, the length of `name_to_hash` dictionary is wrong, get {len(group_name_to_hash)}, "
                                         f"but {num_imgs_per_grp} needed")
                    yield group_name_to_hash
                    group_name_to_hash = {imgname: imghash}
                elif (idx == inner_loop_img_st_idx) or ((idx - inner_loop_img_st_idx) % num_imgs_per_grp != 0):
                    group_name_to_hash.update({imgname: imghash})
            if group_name_to_hash:
                yield group_name_to_hash
            return
        else:  # last inner loop with non-zero remainder
            num_imgs_per_grp_last_inner_iter = num_imgs_last_inner_iter // num_grp_last_inner_iter
            remained_imgs_last_inner_iter = num_imgs_last_inner_iter % num_grp_last_inner_iter
            group_name_to_hash = dict()
            idx = inner_loop_img_st_idx
            name_to_hash_items = name_to_hash.items()
            while True:
                if idx < inner_loop_img_ed_idx and idx < total_imgs:
                    if len(group_name_to_hash) < num_imgs_per_grp_last_inner_iter:
                        # imgname, imghash = name_to_hash_items.pop(0)
                        imgname, imghash = name_to_hash_items[idx][0], name_to_hash_items[idx][1]
                        group_name_to_hash.update({imgname: imghash})
                        idx += 1
                        if len(group_name_to_hash) == num_imgs_per_grp_last_inner_iter:
                            if remained_imgs_last_inner_iter:
                                imgname, imghash = name_to_hash_items[idx][0], name_to_hash_items[idx][1]
                                group_name_to_hash.update({imgname: imghash})
                                idx += 1
                                remained_imgs_last_inner_iter -= 1
                            yield group_name_to_hash
                            group_name_to_hash = dict()
                else:
                    break
            if group_name_to_hash:
                yield group_name_to_hash
            return

    def group_dupsfind_task_func(
                                 self, 
                                 group_name_to_hash: Dict, 
                                 res_save_p: PosixPath = None, 
                                 dup_names: ListProxy = None, 
                                 max_distance_threshold: int = 5, 
                                 scores: bool = False, 
                                 search_method: str = 'brut_force_cython' if not sys.platform == 'win32' else 'bktree', 
                                 group_dist_workers: int = 4
                                ):
        global worker_logger, worker_lock, worker_barrier
        # necessary environmental viariables checking
        outer_iter_rank = os.getenv("OUTER_ITER_RANK", None)
        if outer_iter_rank is None:
            with worker_lock:
                worker_logger.error(f"group with pid {os.getpid()} cannot get is outer loop index, i.e. environmental varialbe `OUTER_ITER_RANK`")
            sys.exit(1)
        inner_iter_rank = os.getenv("INNER_ITER_RANK", None)
        if inner_iter_rank is None:
            with worker_lock:
                worker_logger.error(f"group with pid {os.getpid()} cannot get its inner loop index, i.e. environmental variable `INNER_ITER_RANK`")
            sys.exit(1)
        group_rank = os.getenv("RANK", None)
        if group_rank is None:
            with worker_lock:
                worker_logger.error(f"group with pid {os.getpid()} cannot get its group/process index , i.e. environmental variable `RANK`")
            sys.exit(1)

        cur_res_p = res_save_p.parent / f"iter_{outer_iter_rank}th_{res_save_p.name}"

        st_time = datetime.now()
        with worker_lock:
            worker_logger.info(f"group with outer loop index {outer_iter_rank}, inner loop index {inner_iter_rank} and rank {group_rank} "
                               f"starts its duplications finding task among {len(group_name_to_hash)} images with hashes")
        try:
            dupsfind_obj = HashEval(
                                    test=group_name_to_hash, queries=group_name_to_hash, distance_function=self.hamming_distance, verbose=self.verbose, 
                                    threshold=max_distance_threshold, search_method=search_method, num_dist_workers=group_dist_workers
                                   )
            dupsfind_res: Dict[str, List[Union[str, Tuple[str, int]]]] = dupsfind_obj.retrieve_results(scores=scores)
            dups_set = set()
            for dups in dupsfind_res.values():
                if not scores:
                    dups_set.update(dups)
                else:
                    dups_set.update(dup_ele[0] for dup_ele in dups)
        except Exception as dupsfind_err:
            with worker_lock:
                worker_logger.error(f"when group with outer loop index {outer_iter_rank}, inner loop index {inner_iter_rank} and rank {group_rank} "
                                    f"executing its duplications finding task, encounter an error ({type(dupsfind_err)}): {dupsfind_err}")
            sys.exit(1)

        # save duplications finding results into dupsfind result path
        with worker_lock:
            # save duplications finding results into dupsfind result path
            try:
                with open(cur_res_p, mode="a", encoding="utf-8") as cur_res_fp:
                    for dupsfind_k, dupsfind_v in dupsfind_res.items():
                        json.dump({dupsfind_k: dupsfind_v}, cur_res_fp, ensure_ascii=False)
                        cur_res_fp.write("\n")
            except Exception as save_err:
                worker_logger.error(f"group with outer loop index {outer_iter_rank}, inner loop index {inner_iter_rank} and rank {group_rank} "
                                    f"encounters a saving error ({type(save_err)}): {save_err}")

            # add duplicated image names into duplicated names shared list
            for dupname in dups_set:
                if dupname not in dup_names:
                    dup_names.append(dupname)
            ed_time = datetime.now()
            worker_logger.info(f"group with outer loop index {outer_iter_rank}, inner loop index {inner_iter_rank} and rank {group_rank} finds "
                               f"{len(dups_set)} duplications, taking {((ed_time - st_time).total_seconds() / 60):.3f} minutes")
        worker_barrier.wait()
        return

    def group_dedup_in_tars(
                            self, 
                            hashtars_dir_str: str, 
                            outer_loop_iter: int = 3, 
                            shuffle_per_outer_iter: int = 5, 
                            num_grp_inner_iter: int = 12, 
                            num_imgs_per_grp: int = 200000, 
                            max_distance_threshold: int = 5, 
                            scores: bool = False, 
                            search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree', 
                            num_tarread_workers: int = 8, 
                            group_dist_workers: int = 4, 
                            num_dedup_workers: int = 16
                           ):
        """
        Find duplicated images of al tar files under a specified directory via group-based duplications finding. All images with hamming distance 
        less than or equal to the max_distance_threshold are regarded as duplicates.

        Args:
            hashtars_dir_str: str, tar files directory, in which each tar file contains hash encodings of some images.
            outer_loop_iter: int, when using group-based duplications finding, number of iterations for outer loop.
            dups_num_threshold: int, when using group-based duplications finding, duplications number for ending condition.
            num_grp_inner_iter: int, when using group-based duplications finding, number of groups (process pool) for each inner iteration.
            num_imgs_per_grp: int, when using group-based duplications finding, images number for each group (process pool).
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are valid. 
                                    (must be an integer between 0 and 64). Default is 5
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            search_method: Optional, Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            num_tarread_workers: Optional, an integer indicating number of workers to read hash tar files in parallel.
            group_dist_workers: Optional, an integer indicating number of workers to find duplications in each group (process pool).
            num_dedup_workers: Optional, an interger indicating number of workers to execute deduplicating of images in tars.
        """
        hashtars_dir = Path(hashtars_dir_str)
        if not hashtars_dir.parent.parent.exists():
            raise FileNotFoundError(f"the raw image-alttext pairs directory - {hashtars_dir.parent.parent} doesn't exist!")
        if hashtars_dir.name != "image-text-pairs":
            raise ValueError(f"outermost path of `hashtars_dir_str` should be `image-text-pairs`, but got {hashtars_dir.name}")
        if hashtars_dir.parent.name != "images_hash":
            raise ValueError(f"the second to last path of `hashtars_dir_str` should be `images_hash`, but got {hashtars_dir.parent.name}")
        if not hashtars_dir.exists():
            raise FileNotFoundError(f"tar files directory string - {hashtars_dir} does not exist")
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)

        self.dedup_logger_init(hashtars_dir.parent.parent, "images_group_dedup")
        global logger
        logger.info(f"maximum number of outer loop for group-based duplications finding: {outer_loop_iter}")
        logger.info(f"group (process pool) number in each iteration of inner loop for group-based duplications finding: {num_grp_inner_iter}")
        logger.info(f"image number for each gropu (process pool) for group-based duplications finding: {num_imgs_per_grp}")
        logger.info(f"parameter `max_distance_threshold` (int) for duplications finding: {max_distance_threshold}")
        logger.info(f"parameter `scores` (bool) for duplications finding: {scores}")
        logger.info(f"parameter `search_method` (str) for duplications finding: {search_method}")
        logger.info(f"number of workers for hash tar files reading: {num_tarread_workers}")
        logger.info(f"number of workers for group-based duplications finding: {group_dist_workers}")
        logger.info(f"number of workers for image deduplication: {num_dedup_workers}")

        dupsfind_res_p = hashtars_dir.parent.parent / "images_group_dedup" / "duplications.jsonl"
        logger.info(f"output path for hash tar files group-based duplications find: {dupsfind_res_p}")

        dedup_res_dir = hashtars_dir.parent.parent / "images_group_dedup" / "image-text-pairs"
        if dedup_res_dir.exists():
            shutil.rmtree(dedup_res_dir)
        else:
            dedup_res_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"output directory for group-based deduplication: {dedup_res_dir}")

        total_num_hashtars = self.get_total_tars_num(hashtars_dir)
        logger.info(f"number of images hash tar files reading in total: {total_num_hashtars}")
        if num_tarread_workers > total_num_hashtars:
            logger.warning(f"hash tars read worker number ({num_tarread_workers}) is greater than hash tar files number ({total_num_hashtars}), "
                           f"hence set hash tars read worker number to {total_num_hashtars}")
            num_tarread_workers = total_num_hashtars

        # group-based duplications finding, finding results saving, and deduplications
        with multiprocessing.Manager() as dedup_manager:
            dedup_pid_to_rank = dedup_manager.dict()
            dedup_lock = dedup_manager.Lock()
            hashtars_read_barrier = dedup_manager.Barrier(num_tarread_workers)
            name_to_hash = dedup_manager.dict()  # process-shared dictionary for hash tar files reading
            dup_names = dedup_manager.list()  # process-shared list saving duplicated image names
            pre_dups_num = 0
            cur_dups_num = 0

            # (1) hash tar files reading
            hashread_st_time = datetime.now()
            logger.info(f"start hash results of tar files reading via process pool at {datetime.strftime(hashread_st_time, date_fmt)}")
            hashtars_read_initargs = (dedup_pid_to_rank, dedup_lock, hashtars_read_barrier, logger)
            hashtars_read_func = partial(self.hashtars_read_task_func, name_to_hash=name_to_hash)
            hashtars_generator = self.tar_generator_func(hashtars_dir, total_num_hashtars, num_tarread_workers)
            with futures.ProcessPoolExecutor(num_tarread_workers, initializer=self.worker_init_func, initargs=hashtars_read_initargs) as hash_tars_read_exec:
                _ = hash_tars_read_exec.map(hashtars_read_func, hashtars_generator, chunksize=1)
            logger.info(f"having read {len(name_to_hash)} images hash from image hash tars directory: {hashtars_dir}")
            hashread_ed_time = datetime.now()
            logger.info(f"end hash results of tar files reading via process pool at {datetime.strftime(hashread_ed_time, date_fmt)}, "
                        f"take {((hashread_ed_time - hashread_st_time).total_seconds() / 60):.3f} minutes totally\n\n")
            dedup_pid_to_rank.clear(); del hashtars_read_barrier
            gc.collect()

            # (2-1) execute duplications finding via group-based deduplication;
            # (2-2) and save duplications finding results into secified file path
            dupsfind_st_time = datetime.now()
            logger.info(f"start group-based duplications finding via process pool at {datetime.strftime(dupsfind_st_time, date_fmt)}")
            native_total_imgs = len(name_to_hash)
            for outer_idx in range(outer_loop_iter):
                outer_iter_st_time = datetime.now()
                logger.info(f"start the {outer_idx + 1}th outer loop group-based duplications finding at {datetime.strftime(outer_iter_st_time, date_fmt)}")

                if outer_idx  % shuffle_per_outer_iter == 0 and outer_idx != 0:
                    logger.info(f"for the {outer_idx + 1}th outer loop of group-based duplications finding, randomly shuffling")
                    name_to_hash_items = list(name_to_hash.items())
                    random.seed(int(time.time()))
                    random.shuffle(name_to_hash_items)
                    name_to_hash.clear()
                    name_to_hash = dedup_manager.dict(name_to_hash_items)
                gc.collect()

                inner_idx = 0
                group_dupsfind_func = partial(
                                              self.group_dupsfind_task_func, 
                                              res_save_p=dupsfind_res_p, 
                                              dup_names=dup_names, 
                                              max_distance_threshold=max_distance_threshold, 
                                              scores=scores, 
                                              search_method=search_method, 
                                              group_dist_workers=group_dist_workers
                                             )
                total_imgs = len(name_to_hash)
                logger.info(f"images number for group-based duplications finding (total_imgs): {total_imgs}")
                num_imgs_per_inner_iter = num_imgs_per_grp * num_grp_inner_iter
                if num_imgs_per_inner_iter < total_imgs:
                    logger.info(f"pre-defined images number for each inner iteration of group-based duplications finding "
                                f"(num_imgs_per_grp * num_grp_inner_iter): {num_imgs_per_inner_iter}")
                else:
                    org_num_imgs_per_inner_iter = num_imgs_per_inner_iter
                    num_grp_inner_iter = 10
                    num_imgs_per_grp = math.floor(total_imgs / (num_grp_inner_iter * 3))
                    num_imgs_per_inner_iter = num_imgs_per_grp * num_grp_inner_iter
                    logger.warning(f"in the {outer_idx}th outer loop, naitve image number inside per inner iteration: "
                                   f"{org_num_imgs_per_inner_iter}, which is greater than total image number: {total_imgs}, "
                                   f"Hence set images per group to {num_imgs_per_grp}; groups per inner "
                                   f"iteration to {num_grp_inner_iter}."
                                  )
                max_inner_iters = math.ceil(total_imgs / num_imgs_per_inner_iter)
                logger.info(f"maximum inner iterations of group-based duplications finding (max_inner_iters): {max_inner_iters}")
                remainder = total_imgs % num_imgs_per_inner_iter
                if remainder != 0:
                    if remainder <= num_grp_inner_iter:
                        num_grp_last_inner_iter = 1
                    else:
                        num_grp_last_inner_iter = num_grp_inner_iter
                    num_imgs_last_inner_iter = remainder

                while True:
                    inner_idx += 1
                    inner_st_time = datetime.now()
                    if inner_idx != max_inner_iters or (inner_idx == max_inner_iters and remainder == 0):
                        inner_iter_grp = num_grp_inner_iter
                        last_inner_loop = False
                    elif inner_idx == max_inner_iters and remainder != 0:
                        inner_iter_grp = num_grp_last_inner_iter
                        last_inner_loop = True
                    logger.info(f"outer loop index: {outer_idx + 1}; inner loop index: {inner_idx}, starting of {inner_iter_grp} groups via a process pool for "
                                f"group-based duplications finding at {datetime.strftime(inner_st_time, date_fmt)}")
                    dedup_pid_to_rank.clear()
                    dupsfind_barrier = dedup_manager.Barrier(inner_iter_grp)
                    procpool_initargs = (dedup_pid_to_rank, dedup_lock, dupsfind_barrier, logger, outer_idx + 1, inner_idx)
                    name_to_hash_generator = self.group_name_to_hash_generator(
                                                                               name_to_hash=name_to_hash, 
                                                                               total_imgs=total_imgs, 
                                                                               num_imgs_per_inner_iter=num_imgs_per_inner_iter, 
                                                                               inner_loop_idx=inner_idx, 
                                                                               num_imgs_per_grp=num_imgs_per_grp, 
                                                                               last_inner_loop=last_inner_loop, 
                                                                               num_grp_last_inner_iter=num_grp_last_inner_iter,
                                                                               num_imgs_last_inner_iter=num_imgs_last_inner_iter
                                                                              )
                    with futures.ProcessPoolExecutor(
                                                     inner_iter_grp, 
                                                     initializer=self.group_dupsfind_worker_init_func, 
                                                     initargs=procpool_initargs
                                                    ) as group_dupsfind_exec:
                        _ = group_dupsfind_exec.map(group_dupsfind_func, name_to_hash_generator, chunksize=1)
                    inner_ed_time = datetime.now()
                    logger.info(f"outer loop index: {outer_idx + 1}; inner loop index: {inner_idx}, ending of {inner_iter_grp} groups via a process pool "
                                f"for group-based duplications finding at {datetime.strftime(inner_ed_time, date_fmt)}, this inner iteration takes "
                                f"{(((inner_ed_time - inner_st_time).total_seconds()) / 60):.3f} minutes in total")
                    if inner_idx == max_inner_iters:
                        break

                outer_iter_ed_time = datetime.now()
                spend_seconds = (outer_iter_ed_time - outer_iter_st_time).total_seconds()
                logger.info(f"end the {outer_idx + 1}th outer loop group-based duplications finding at {datetime.strftime(outer_iter_ed_time, date_fmt)},"
                            f"takes {spend_seconds // 3600} hours and {((spend_seconds % 3600) / 60):.3f} minutes in total")

                # post-process of a outer loop at the end time
                cur_dups_num = len(dup_names)
                logger.info(f"the {outer_idx + 1}th outer loop of group-based duplications finding get {cur_dups_num - pre_dups_num} duplicated images.")
                try:
                    for dupname in dup_names[pre_dups_num: cur_dups_num]:
                        _ = name_to_hash.pop(dupname)
                except Exception as err:
                    logger.error(f"when delete corresponding item according to the duplicated image name, encounter an error ({type(err)}): {err}")
                    sys.exit(1)
                pre_dups_num = cur_dups_num
                gc.collect()

            dupsfind_ed_time = datetime.now()
            dupsfind_elapsed_sec = (dupsfind_ed_time - dupsfind_st_time).total_seconds()
            logger.info(f"end group-based duplications finding via process pool at {datetime.strftime(dupsfind_ed_time, date_fmt)}, "
                        f"takes {dupsfind_elapsed_sec // 3600} hours and {((dupsfind_elapsed_sec % 3600) / 60):.3f} minutes in total\n\n")
            dedup_pid_to_rank.clear(); del dupsfind_barrier
            gc.collect()

            # (3) deduplication by re-archiving to new tar files
            logger.info(f"start images deduplication via process pool at {datetime.strftime(datetime.now(), date_fmt)}")
            partial_tarimgs_dedup_func = partial(self.tarimages_dedup_task_func, dupimgs_shared_list=dup_names, output_dir=dedup_res_dir)
            if len(name_to_hash) + len(dup_names) != native_total_imgs:
                raise ValueError(f"remained images number {len(name_to_hash)} plus duplicated images {len(dup_names)} "
                                 f"number should be equal to native images number {native_total_imgs}, "
                                 f"but get {len(name_to_hash) + len(dup_names)}")
            logger.info(f"{len(name_to_hash)} images are remained, and {len(dup_names)} images will be deduplicated")
            imgtars_dir = hashtars_dir.parent.parent / "image-text-pairs"
            total_num_imgtars = self.get_total_tars_num(imgtars_dir)
            if num_dedup_workers > total_num_imgtars:
                logger.warning(f"number worker for deduplication ({num_dedup_workers}) is greater than total number "
                               f"of image-text tars ({total_num_imgtars}), hence set num_dedup_workers to {total_num_imgtars}")
                num_dedup_workers = total_num_imgtars
            imgtars_generator = self.tar_generator_func(imgtars_dir, total_num_imgtars, num_dedup_workers)
            dedup_barrier = dedup_manager.Barrier(num_dedup_workers)
            dedup_initargs = (dedup_pid_to_rank, dedup_lock, dedup_barrier, logger)
            with futures.ProcessPoolExecutor(num_dedup_workers, initializer=self.worker_init_func, initargs=dedup_initargs) as dedup_exec:
                _ = dedup_exec.map(partial_tarimgs_dedup_func, imgtars_generator, chunksize=1)
            logger.info(f"end images deduplication via process pool at {datetime.strftime(datetime.now(), date_fmt)}\n\n")
            dedup_pid_to_rank.clear(); del dedup_barrier
            gc.collect()

        # (4) arguments saving
        save_args = {
                     "hashtars_dir": hashtars_dir_str, 
                     "outer_loop_iter": outer_loop_iter, 
                     "shuffle_per_outer_iter": shuffle_per_outer_iter, 
                     "num_grp_inner_iter": num_grp_inner_iter, 
                     "num_imgs_per_grp": num_imgs_per_grp, 
                     "max_distance_threshold": max_distance_threshold, 
                     "scores": scores, 
                     "search_method": search_method, 
                     "hashtar_reading_workers": num_tarread_workers, 
                     "dupsfind_workers": group_dist_workers, 
                     "dedup_workers": num_dedup_workers, 
                     "dupsfind_result_path": str(dupsfind_res_p), 
                     "dedup_result_dir": str(dedup_res_dir)
                    }
        args_save_p = dedup_res_dir.parent / "arguments.json"
        with open(args_save_p, mode="w", encoding="utf-8") as args_save_fp:
            json.dump(save_args, args_save_fp, ensure_ascii=False)
        logger.info(f"having saved group-based duplications finding and deduplication arguments into file {args_save_p}\n\n")

        return

    def _hash_algo(self, image_array: np.ndarray):
        pass

    def _hash_func(self, image_array: np.ndarray):
        hash_mat = self._hash_algo(image_array)
        return self._array_to_hash(hash_mat)

    # search part

    @staticmethod
    def _check_hamming_distance_bounds(thresh: int) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be int between 0 and 64)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If invalid value is provided.
        """
        if not isinstance(thresh, int):
            raise TypeError('Threshold must be an int between 0 and 64')
        elif thresh < 0 or thresh > 64:
            raise ValueError('Threshold must be an int between 0 and 64')
        else:
            return None

    def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, str],
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes).
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether hamming distance scores are to be returned along with retrieved
            duplicates.
            outfile: Optional, name of the file to save the results. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        logger.info('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            distance_function=self.hamming_distance,
            verbose=self.verbose,
            threshold=max_distance_threshold,
            search_method=search_method,
            num_dist_workers=num_dist_workers
        )

        logger.info('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            save_json(self.results, outfile)
        return self.results

    def find_duplicates(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as
        duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each
        query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional,  used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            duplicates dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, scores=True, outfile='results.json')
        ```
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
                recursive=recursive,
                num_enc_workers=num_enc_workers,
                num_dist_workers=num_dist_workers
            )
        elif encoding_map:
            if recursive:
                warnings.warn('recursive parameter is irrelevant when using encodings.', SyntaxWarning)
            
            warnings.warn('Parameter num_enc_workers has no effect since encodings are already provided', RuntimeWarning)
            
            result = self._find_duplicates_dict(
                encoding_map=encoding_map,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
                num_dist_workers=num_dist_workers
            )
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

    def _find_duplicates_dir(
        self,
        image_dir: PurePath,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected below the given hamming distance
        threshold. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Name of the file the results should be written to.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        encoding_map = self.encode_images(image_dir, recursive=recursive, num_enc_workers=num_enc_workers)
        results = self._find_duplicates_dict(
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=scores,
            outfile=outfile,
            search_method=search_method,
            num_dist_workers=num_dist_workers
        )
        return results

    def find_duplicates_to_remove(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        outfile: Optional[str] = None,
        recursive: Optional[bool] = False,
        num_enc_workers: int = cpu_count(),
        num_dist_workers: int = cpu_count()
    ) -> List:
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.
            num_dist_workers: Optional, number of cpu cores to use for multiprocessing distance computation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            duplicates: List of image file names that are found to be duplicate of me other file in the directory.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        max_distance_threshold=15)

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, outfile='results.json')
        ```
        """
        result = self.find_duplicates(
            image_dir=image_dir,
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=False,
            recursive=recursive,
            num_enc_workers=num_enc_workers,
            num_dist_workers=num_dist_workers
        )
        files_to_remove = get_files_to_remove(result)
        if outfile:
            save_json(files_to_remove, outfile)
        return files_to_remove


class PHash(Hashing):
    """
    Inherits from Hashing base class and implements perceptual hashing (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html).

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Perceptual hash for images
    from imagededup.methods import PHash
    phasher = PHash()
    perceptual_hash = phasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    perceptual_hash = phasher.encode_image(image_array = <numpy image array>)
    OR
    perceptual_hashes = phasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import PHash
    phasher = PHash()
    duplicates = phasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = phasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import PHash
    phasher = PHash()
    files_to_remove = phasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = phasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True, target_size: Tuple[int, int]  = (32, 32), 
                 coefficient_extract: Tuple[int, int] = (8, 8)) -> None:
        """
        Initialize perceptual hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.__coefficient_extract = coefficient_extract
        self.target_size = target_size

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the perceptual hash of the image.
        """
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[
            : self.__coefficient_extract[0], : self.__coefficient_extract[1]
        ]

        # median of coefficients excluding the DC term (0th term)
        # mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat


class AHash(Hashing):
    """
    Inherits from Hashing base class and implements average hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Average hash for images
    from imagededup.methods import AHash
    ahasher = AHash()
    average_hash = ahasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    average_hash = ahasher.encode_image(image_array = <numpy image array>)
    OR
    average_hashes = ahasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import AHash
    ahasher = AHash()
    duplicates = ahasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = ahasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import AHash
    ahasher = AHash()
    files_to_remove = ahasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = ahasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True, target_size: Tuple[int, int] = (8, 8)) -> None:
        """
        Initialize average hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.target_size = target_size

    def _hash_algo(self, image_array: np.ndarray):
        """
        Get average hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the average hash of the image.
        """
        avg_val = np.mean(image_array)
        hash_mat = image_array >= avg_val
        return hash_mat


class DHash(Hashing):
    """
    Inherits from Hashing base class and implements difference hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Difference hash for images
    from imagededup.methods import DHash
    dhasher = DHash()
    difference_hash = dhasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    difference_hash = dhasher.encode_image(image_array = <numpy image array>)
    OR
    difference_hashes = dhasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import DHash
    dhasher = DHash()
    duplicates = dhasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = dhasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import DHash
    dhasher = DHash()
    files_to_remove = dhasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = dhasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True, target_size: Tuple[int, int] = (9, 8)) -> None:
        """
        Initialize difference hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.target_size = target_size

    def _hash_algo(self, image_array):
        """
        Get difference hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the difference hash of the image.
        """
        # Calculates difference between consecutive columns and return mask
        hash_mat = image_array[:, 1:] > image_array[:, :-1]
        return hash_mat


class WHash(Hashing):
    """
    Inherits from Hashing base class and implements wavelet hashing. (Implementation reference:
    https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Wavelet hash for images
    from imagededup.methods import WHash
    whasher = WHash()
    wavelet_hash = whasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    wavelet_hash = whasher.encode_image(image_array = <numpy image array>)
    OR
    wavelet_hashes = whasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import WHash
    whasher = WHash()
    duplicates = whasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = whasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import WHash
    whasher = WHash()
    files_to_remove = whasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = whasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True, target_size: Tuple[int, int] = (256, 256)) -> None:
        """
        Initialize wavelet hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.target_size = target_size
        self.__wavelet_func = 'haar'

    def _hash_algo(self, image_array):
        """
        Get wavelet hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the wavelet hash of the image.
        """
        # decomposition level set to 5 to get 8 by 8 hash matrix
        image_array = image_array / 255
        coeffs = pywt.wavedec2(data=image_array, wavelet=self.__wavelet_func, level=5)
        LL_coeff = coeffs[0]

        # median of LL coefficients
        median_coef_val = np.median(np.ndarray.flatten(LL_coeff))

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = LL_coeff >= median_coef_val
        return hash_mat
