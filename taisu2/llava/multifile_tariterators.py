from typing import Set, Tuple, Dict, Any, Optional, Iterable, Iterator, Callable
from webdataset.tariterators import base_plus_ext, valid_sample
from webdataset.tariterators import url_opener, tar_file_expander
from webdataset.handlers import reraise_exception
from webdataset.filters import pipelinefilter


__all__ = ["tarfile_to_samples"]


def group_by_keys(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[str, str]] = base_plus_ext,
    lcase: bool = True,
    trace: bool = False, 
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Group tarfile contents by keys and yield samples.
       Support single suffix with multi files (e.g. multi images with .jpg suffix).
       This may be useful for further interleaved image-text data.

    Args:
        data: Iterator over tarfile contents.
        keys: Function that takes a file name and returns a key and a suffix.
        lcase: Whether to lowercase the suffix.
        trace: Whether to trace and print each dictionary sample.
        suffixes: List of suffixes to keep.
        handler: Exception handler.

    Raises:
        ValueError: If there are duplicate file names in the tar file.

    Yields:
        Iterator over samples.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            if filesample == {}:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = None
                continue
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if prefix is not None:
                if len(prefix.split("_", maxsplit=1)) == 1:
                    multi_flag = False
                else:
                    multi_flag = True
                    prefix, sub_prefix = prefix.split("_", maxsplit=1)
            if trace:
                print(
                    prefix,
                    suffix,
                    current_sample.keys() if isinstance(current_sample, dict) else None,
                )
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                if not multi_flag:
                    raise ValueError(
                        f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
                    )
                else:
                    current_sample[suffix].update({str(sub_prefix): value})
                    continue
            if suffixes is None or suffix in suffixes:
                if not multi_flag:
                    current_sample[suffix] = value
                else:
                    current_sample[suffix] = {str(sub_prefix): value}
            local_path = filesample.get("__local_path__")
            if local_path is not None:
                current_sample["__local_path__"] = local_path
        except Exception as exn:
            exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
            if handler(exn):
                continue
            else:
                break
    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples(
    src: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterable[Dict[str, Any]]:
    """Generate samples from a stream of tar files.

    Args:
        src: Stream of tar files.
        handler: Exception handler.
        select_files: Function that selects files to be included.
        rename_files: Function to rename files.

    Returns:
        Stream of samples.
    """
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = group_by_keys(files, handler=handler)
    return samples


tarfile_to_samples = pipelinefilter(tarfile_samples)
