import io, os, re
import json
import time
import gc, sys
from collections import deque
from math import floor, ceil
import pandas as pd
import soundfile as sf
import tarfile
from tarfile import TarInfo
from datetime import datetime
from typing import List, Tuple
import argparse
import multiprocessing
from multiprocessing.managers import AcuquirerProxy, DictProxy
from multiprocessing.synchronize import Barrier, Lock
from multiprocessing.sharedctypes import Synchronized
from concurrent import futures as futures
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL
from functools import partial


logger: logging.Logger = None
pid_to_rank: DictProxy = None
proc_lock: AcuquirerProxy = None
proc_barrier: Barrier = None

date_fmt = '%Y-%m-%d-%H:%M:%S'
log_fmt = '[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]'

total_audio_num = 0


def parse_args():
    parser = argparse.ArgumentParser(description='archive multi audio-caption dataset into a whole webdataset-format dataset')
    parser.add_argument('--audio-dir', nargs='+', default=[
        '/home/lizongshu/datasets/AudioSetCaps/AudioSet',
        '/home/lizongshu/datasets/AudioSetCaps/YouTube-8M',
        '/home/lizongshu/datasets/Auto-ACD/train',
        '/home/lizongshu/datasets/Clotho/development',
        '/home/lizongshu/datasets/Sound-VECaps/Sound-VECaps_audio',
        '/home/lizongshu/datasets/WavCaps/Zip_files/AudioSet_SL',
        '/home/lizongshu/datasets/WavCaps/Zip_files/BBC_Sound_Effects',
        '/home/lizongshu/datasets/WavCaps/Zip_files/FreeSound',
        '/home/lizongshu/datasets/WavCaps/Zip_files/SoundBible',
    ], type=str, help='raw audio files directories')
    parser.add_argument('--anno-path', nargs='+', default=[
        '/home/lizongshu/datasets/AudioSetCaps/AudioSetCaps_caption.csv',
        '/home/lizongshu/datasets/AudioSetCaps/YouTube-8M_AudioSetCaps_caption.csv',
        '/home/lizongshu/datasets/Auto-ACD/train.csv',
        '/home/lizongshu/datasets/Clotho/clotho_captions_development.csv',
        '/home/lizongshu/datasets/Sound-VECaps/Sound-VECaps_audio.csv',
        '/home/lizongshu/datasets/WavCaps/json_files/AudioSet_SL/as_final.json',
        '/home/lizongshu/datasets/WavCaps/json_files/BBC_Sound_Effects/bbc_final.json',
        '/home/lizongshu/datasets/WavCaps/json_files/FreeSound/fsd_final.json',
        '/home/lizongshu/datasets/WavCaps/json_files/SoundBible/sb_final.json',
    ], type=str, help='paths of annotation files holding audio captions')
    parser.add_argument('--output-dir', default=None, type=str, help='final resulted wds-format dataset directory')
    parser.add_argument('--num-workers', default=8, type=int, help='number of parallel workers to archive audio-caption pairs')
    parser.add_argument('--data-num-per-tar', default=10000, type=int, help='number of audio-caption pairs in each archived tar file')
    parser.add_argument('--log-level', type=int, default=DEBUG, choices=(NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL),
        help='logging level for Logger instance'
    )
    args = parser.parse_args()
    if not (args.audio_dir and args.anno_path):
        raise argparse.ArgumentError(
            'both `--audio-dir` and `--anno-path` should not be empty, but get --audio-dir: '
            f'{args.audio_dir}; --anno-path: {args.anno_path}.'
        )
    if len(args.audio_dir) != len(args.anno_path):
        raise argparse.ArgumentError(
            f'the audio directories and annotation paths should be one-to-one, but get {len(args.audio_dir)} audio directories and '
            f'{len(args.anno_path)} annotation file paths'
        )
    for audio_d in args.audio_dir:
        if not os.path.exists(audio_d):
            raise FileNotFoundError(f'Invalid audio directory: {audio_d}')
    for anno_p in args.anno_path:
        if not os.path.exists(anno_p):
            raise FileNotFoundError(f'Invalid anno path: {anno_p}')
    return args


def init_logger(log_level: int):
    global logger, log_fmt
    logger = logging.getLogger('multi audio-caption datasets archiving using process pool')
    logger.setLevel(log_level)
    fmtter = logging.Formatter(log_fmt)
    strm_hndlr = logging.StreamHandler()
    strm_hndlr.setFormatter(fmtter)
    strm_hndlr.setLevel(log_level)
    logger.addHandler(strm_hndlr)


def proc_pool_init_func():
    global logger, pid_to_rank, proc_lock
    pid = str(os.getpid())
    with proc_lock:
        proc_rank = str(len(pid_to_rank))
        pid_to_rank[pid] = proc_rank
    os.environ['RANK'] = proc_rank

    global proc_barrier
    if not isinstance(proc_barrier, Barrier) or proc_barrier is None:
        with proc_lock:
            logger.error(
                f'process with rank index {proc_rank} hasn\'t got the `Barrier` object, '
                'hence exit its initialization abnormally'
            )
        sys.exit(1)
    logger.info(f'process with pid {pid} get global rank index {proc_rank}, and has finished initialization')


def archive_func(
    # (audio/caption file idx, data item start idx)
    worker_task_indices: deque[Tuple[int, int]],
    audio_dir_list: List[str] = None,
    anno_path_list: List[str] = None,
    output_dir: str = None,
    data_num_per_tar: int = None,
    tar_num_list: List[int] = None,
):
    global logger, proc_lock, proc_barrier
    # single process initialization
    proc_rank = int(os.getenv('RANK', 0))
    data_idx = 0
    worker_task_indices = deque(worker_task_indices)
    tar_idx = sum(tar_num_list[: proc_rank])
    remain_raw_audio_names = None
    prev_data_dir_idx = data_dir_idx = None

    while worker_task_indices and (not remain_raw_audio_names):
        prev_data_dir_idx = data_dir_idx
        tar_p = os.path.join(output_dir, f'{tar_idx:05d}')
        if not remain_raw_audio_names:
            data_dir_idx, data_item_st_idx = worker_task_indices.popleft()
            audio_dir = audio_dir_list[data_dir_idx]
            anno_path = anno_path_list[data_dir_idx]
            remain_raw_audio_names = deque(
                os.listdir(audio_dir)[data_item_st_idx: data_item_st_idx + data_num_per_tar]
            )

        # process raw annotations
        if prev_data_dir_idx != data_dir_idx:
            if anno_path.endswith('.csv'):
                annos = pd.read_csv(anno_path)
            elif anno_path.endswith('.json'):
                with open(anno_path, encoding='utf-8', mode='r') as anno_fp:
                    annos = json.load(anno_fp)['data']
            else:
                raise TypeError(
                    f'On process with rank {proc_rank}, get a invalid annotation file: {anno_path}, '
                    'current only support annotation file with format `csv` or `json`'
                )
            if isinstance(annos, pd.DataFrame):
                csv_anno = True
                json_anno = False
                anno_names = annos[annos.columns[0]]
                anno_captions = annos[annos.columns[1]]
            elif isinstance(annos, list):
                csv_anno = False
                json_anno = True
                anno_names = [ann['id'].split('.')[0] for ann in annos]
                anno_captions = [ann['caption'] for ann in annos]

        with tarfile.open(tar_p, mode='w', encoding='utf-8') as tar_fp:
            while data_idx == data_num_per_tar or (not remain_raw_audio_names):
                raw_audio_name = remain_raw_audio_names.popleft()
                raw_audio_stem = raw_audio_name.split('.')[0]
                raw_audio_suffix = raw_audio_name.split('.')[1]
                raw_audio_p = os.path.join(audio_dir, raw_audio_name)

                # search the paired caption for current audio
                if csv_anno:
                    matched_series = anno_names.str.contains(raw_audio_stem, case=True, regex=False, na=False)
                    if matched_series.any():
                        matched_ids = anno_names.index[matched_series].values
                        matched_id = matched_ids[0]
                        audio_caption: str = anno_captions[matched_id]
                    else:
                        with proc_lock:
                            logger.warning(
                                f'On process  with rank {proc_rank}, get a raw audio with path - '
                                f'{raw_audio_p}, which doesn\'t have '
                                f'matched annotation in the csv file - {anno_path}'
                            )
                        continue
                if json_anno:
                    try:
                        matched_id = anno_names.index(raw_audio_stem)
                        audio_caption: str = anno_captions[matched_id]
                    except ValueError as _:
                        with proc_lock:
                            logger.warning(
                                f'On process with rank {proc_rank}, get a raw_audio with path - '
                                f'{raw_audio_p}, which doesn\'t have '
                                f'matched annotation in the json file - {anno_path}'
                            )
                        continue

                data_stem = f'{data_idx:05d}'
                new_audio_name = data_stem + '.' + raw_audio_suffix
                audio_tarinfo = TarInfo(new_audio_name)
                with open(raw_audio_p, mode='rb') as audio_fp:
                    audio_bytes = audio_fp.read()
                audio_tarinfo.size = len(audio_bytes)
                raw_audio_stat = os.stat(raw_audio_p)
                raw_audio_mtime = raw_audio_stat.st_mtime
                raw_audio_mode = raw_audio_stat.st_mode
                audio_tarinfo.mtime = raw_audio_mtime
                audio_tarinfo.mode = raw_audio_mode

                new_caption_name = data_stem + '.txt'
                caption_tarinfo = TarInfo(new_caption_name)
                caption_bytes = audio_caption.encode(encoding='utf-8')
                caption_tarinfo.size = len(caption_bytes)
                raw_anno_stat = os.stat(anno_path)
                raw_anno_mtime = raw_anno_stat.st_mtime
                raw_anno_mode = raw_anno_stat.st_mode
                caption_tarinfo.mtime = raw_anno_mtime
                caption_tarinfo.mode = raw_anno_mode

                tar_fp.addfile(audio_tarinfo, io.BytesIO(audio_bytes))
                tar_fp.addfile(caption_tarinfo, io.BytesIO(caption_bytes))
                data_idx += 1
        with proc_lock:
            logger.info(
                f'process with rank {proc_rank} has archived a tar file whose name is {tar_idx:05d} successfully; '
                f'{data_idx + 1} audio-caption pairs are archived into this tar.'
            )
        if data_idx == data_num_per_tar:
            tar_idx += 1
            data_idx = 0


def main():
    args = parse_args()
    init_logger(args.log_level)
    # element for per worker: (audio/caption file path idx, data item start idx)
    archive_indices = []
    global total_audio_num
    for ds_idx, audio_d in enumerate(args.audio_dir):
        audio_num = len(os.listdir(audio_d))
        total_audio_num += audio_num
        for audio_st_idx in range(0, audio_num, args.data_num_per_tar):
            archive_indices.append((ds_idx, audio_st_idx))
    num_tasks_per_worker = floor(len(archive_indices) // args.num_workers)
    worker_tasks_list: List[List[Tuple[int, int]]] = []
    for worker_idx in range(args.num_worker):
        if worker_idx != args.num_worker - 1:
            worker_tasks_list.append(
                archive_indices[worker_idx * num_tasks_per_worker: ]
            )
        else:
            worker_tasks_list.append(
                archive_indices[worker_idx * num_tasks_per_worker: (worker_idx + 1) * num_tasks_per_worker]
            )
    tar_num_list = [len(task_list) for task_list in worker_tasks_list]

    st_time = datetime.now()
    logger.info(f'begin archiving audio-caption dataset at {datetime.strftime(st_time, date_fmt)}')
    with multiprocessing.Manager() as archive_manager:
        global pid_to_rank, proc_lock, proc_barrier
        pid_to_rank = archive_manager.dict()
        proc_lock = archive_manager.Lock()
        proc_barrier = archive_manager.Barrier(args.num_workers)
        archive_task_func = partial(
            archive_func,
            audio_dir_list=args.audio_dir,
            anno_path_list=args.anno_path,
            output_dir=args.output_dir,
            data_num_per_tar=args.data_num_per_tar,
            tar_num_list=tar_num_list,
        )
        with futures.ProcessPoolExecutor(max_workers=args.num_workers, initializer=proc_pool_init_func) as archive_proc_pool:
            _ = archive_proc_pool.map(
                archive_task_func,
                worker_tasks_list,
                chunksize=args.num_workers
            )
        gc.collect()

    ed_time = datetime.now()
    elapsed_secs = (ed_time - st_time).total_seconds()
    elapsed_hours = int(elapsed_secs // 3600)
    elapsed_mins = (elapsed_secs % 3600) / 60
    logger.info(
        f'end archiving audio-caption datasets at {datetime.strftime(ed_time, date_fmt)}, takes '
        f'{elapsed_hours:d} hours and {elapsed_mins:.3f} minutes'
    )


if __name__ == '__main__':
    main()
