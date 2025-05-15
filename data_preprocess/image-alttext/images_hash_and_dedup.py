import os
import argparse
from pathlib import Path, PosixPath
from imagededup import AHash, WHash, PHash, DHash
from imagededup import Hashing


def parse_args():
    parser = argparse.ArgumentParser(description="command line argument parser for images hash encoding and deduplicatoin")
    parser.add_argument("--raw-data-folder", type=str, default=None, help="raw image-alttext pairs data folder for hash encoding and deduplication")
    parser.add_argument("--hash-method", type=str, default="PHash", choices=["PHash", "DHash", "WHash", "AHash"], 
                        help="hash method string for images hashing and deduplication")
    parser.add_argument("--num-hash-workers", type=int, default=16, help="max workers for images hash encoding")
    parser.add_argument("--verbose", type=lambda x: bool(eval(x)), default=False, help="whether verbose output of Hashing class")
    parser.add_argument("--group-dupsfind", type=lambda x: bool(eval(x)), default=True, help="whether used the group-based duplications finding")
    parser.add_argument("--outer-loop-iter", type=int, default=3, help="when using group-based duplications finding, number of iterations for outer loop")
    parser.add_argument("--shuffle-interval", type=int, default=5, help="the random shuffling iteration interval for group-based duplications finding")
    parser.add_argument("--num-grp-inner-iter", type=int, default=15, help="when using group-based duplications finding, number of groups (process pool) for each innter iteration")
    parser.add_argument("--num-imgs-per-grp", type=lambda x: int(eval(x)), default=200000, help="when using group-based duplications finding, image files number for each group (process pool)")
    parser.add_argument("--max-distance-threshold", type=int, default=5, help="hamming distance threshold for duplications judgement")
    parser.add_argument("--scores", type=lambda x: bool(eval(x)), default=False, help="whether returns hamming distance scores together with duplicated names")
    parser.add_argument("--search-method", type=str, default="brute_force_cython", choices=("bktree", "brute_force", "brute_force_cython"), 
                        help="duplications search method")
    parser.add_argument("--num-tarread-workers", type=int, default=16, help="max workers for image hash reading from hash tar files")
    parser.add_argument("--num-dist-workers", type=int, default=16, help="max workers for distance caculation of duplications finding")
    parser.add_argument("--group-dist-workers", type=int, default=4, help="max workers for each group's distance caculation of group-based duplications finding")
    parser.add_argument("--num-dedup-workers", type=int, default=16, help="max workers for image-alttext data in tar files deduplication")
    args = parser.parse_args()
    args.raw_data_dir = os.getenv("HOME", None) + f"/datasets/Taisu2_datasets/{args.raw_data_folder}"
    return args


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


def main():
    args = parse_args()
    hash_obj: Hashing = eval(args.hash_method)(verbose=args.verbose)
    # images hash encoding and saving hash results into corresponding tar files
    hashtars_dir = Path(args.raw_data_dir) / "images_hash" / "image-text-pairs"
    if not hashtars_dir.exists():
        hash_obj.encode_images_in_tars(data_dir=args.raw_data_dir, num_enc_workers=args.num_hash_workers)

    raw_image_text_dir: PosixPath = Path(args.raw_data_dir) / "image-text-pairs"
    raw_tars_num = get_total_tars_num(raw_image_text_dir)
    if not hashtars_dir.exists():
        raise FileNotFoundError(f"image hash tar files directory - {hashtars_dir}, does not exist!")
    hash_tars_num = get_total_tars_num(hashtars_dir)
    if raw_tars_num != hash_tars_num:
        raise ValueError(f"tar files number of raw image-alttext data ({raw_tars_num}) is not equal to that of image hash data({hash_tars_num})!")

    # hash reading, find duplications, and deduplications
    if not args.group_dupsfind:
        hash_obj.dedup_images_in_tars(
                                      str(hashtars_dir), max_distance_threshold=args.max_distance_threshold, scores=args.scores, 
                                      search_method=args.search_method, num_tar_workers=args.num_tarread_workers, 
                                      num_dist_workers=args.num_dist_workers, num_dedup_workers=args.num_dedup_workers
                                     )
    else:  # group-based duplications finding
        hash_obj.group_dedup_in_tars(
                                     str(hashtars_dir), outer_loop_iter=args.outer_loop_iter, shuffle_per_outer_iter=args.shuffle_interval, 
                                     num_grp_inner_iter=args.num_grp_inner_iter, num_imgs_per_grp=args.num_imgs_per_grp, 
                                     max_distance_threshold=args.max_distance_threshold, scores=args.scores, search_method=args.search_method, 
                                     num_tarread_workers=args.num_tarread_workers, group_dist_workers=args.group_dist_workers, 
                                     num_dedup_workers=args.num_dedup_workers
                                    )


if __name__ == "__main__":
    main()
