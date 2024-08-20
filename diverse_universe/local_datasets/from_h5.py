import h5py
import os
import sys
import torch
import numpy as np
import random
from stuned.utility.utils import (
    get_project_root_path,
    # append_dict,
    # pretty_json
)
from stuned.local_datasets.utils import (
    get_generic_train_eval_dataloaders
)
from stuned.utility.utils import (
    log_or_print,
    error_or_print,
    get_with_assert,
    get_hash
)


# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(
    0,
    get_project_root_path()
)
from diverse_universe.local_datasets.utils import (
    extract_subset_indices
)
# from utility.utils import (
#     get_with_assert,
#     get_hash,
#     log_or_print,
#     error_or_print
# )
sys.path.pop(0)


SEP = '_'
MAX_CHUNK_SIZE = 200000
FROM_H5 = "from_h5"


class HDF5Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        hdf5_file_path,
        keys_order,
        indices_to_keep,
        index_key,
        unique_hash,
        max_chunk,
        total_samples,
        reverse_indices,
        return_index,
        logger
    ):

        self.logger = logger

        if unique_hash is not None:
            filename_hash = extract_hdf5_hash(os.path.basename(hdf5_file_path))

            assert filename_hash == unique_hash, \
                f"Hash mismatch: {filename_hash} vs {unique_hash}"

        self.keys_order = keys_order
        self.hdf5_file_path = hdf5_file_path
        self.index_key = index_key
        self.max_chunk = max_chunk
        self.total_samples = total_samples
        self.reverse_indices = reverse_indices
        self.reversed = False
        self.return_index = return_index

        if indices_to_keep is None:
            assert self.reverse_indices is False, \
                "Can't reverse indices without indices_to_keep"
            self.indices_to_keep = None
        else:
            self.indices_to_keep = set(indices_to_keep.numpy().tolist())

        self._init_from_chunk(0, max_chunk)

    def _init_from_chunk(self, start_index, end_index):

        self.access_counter = 0

        with h5py.File(self.hdf5_file_path, 'r') as file:

            dataset_index = file[self.index_key]

            if self.total_samples is None:
                self.total_samples = len(dataset_index)
                log_or_print(
                    f"Total samples: {self.total_samples}",
                    self.logger
                )

            if self.reverse_indices and not self.reversed:
                self.indices_to_keep \
                    = set(range(self.total_samples)) - self.indices_to_keep
                self.reversed = True

            if end_index is None:
                end_index = self.total_samples

            if (
                    self.max_chunk is not None
                and
                    self.max_chunk >= self.total_samples
            ):
                self.max_chunk = None

            if (
                self.max_chunk is not None
                and self.total_samples % self.max_chunk != 0
            ):
                error_or_print(
                    "Total samples % max_chunk != 0, "
                    "it can lead to undefined behaviour",
                    self.logger
                )

            end_index = min(end_index, self.total_samples)

            log_or_print(
                f"Init for chunk: {start_index} {end_index}",
                self.logger
            )

            # TODO(Alex | 13.01.2024) init through keys,
            # remove hardcoded 'embed' and 'label'
            self.embed = None
            self.label = None
            self.embed = torch.Tensor(file['embed'][start_index:end_index])
            self.label = torch.tensor(
                file['label'][start_index:end_index],
                dtype=torch.int64
            )
            indices = (
                dataset_index[start_index:end_index]
                    .squeeze()
                    .astype(np.int32)
                    .tolist()
            )

            if self.indices_to_keep is not None:

                self.active_indices = []
                for i in range(len(indices)):
                    if indices[i] in self.indices_to_keep:
                        self.active_indices.append(i)
            else:
                self.active_indices = list(range(len(indices)))
            log_or_print(
                f"Num active indices: {len(self.active_indices)}",
                self.logger
            )

        if end_index == self.total_samples:
            self.prev_end_index = 0
        else:
            self.prev_end_index = end_index

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, index):

        dataset_len = len(self)
        if self.access_counter == dataset_len:
            while True:
                log_or_print("Accessed all data, re-init", self.logger)
                self._init_from_chunk(
                    self.prev_end_index,
                    self.prev_end_index + self.max_chunk
                )
                dataset_len = len(self)
                # for small chunk sizes num_active indices might be 0
                if dataset_len > 0:
                    break

        # TODO(Alex | 17.01.2024): can we avoid such situations?
        if index >= dataset_len:
            error_or_print(
                f"Index {index} out of range {dataset_len}, "
                f"substituting it with random",
                self.logger
            )
            index = random.randint(0, dataset_len - 1)

        active_index = self.active_indices[index]

        if self.max_chunk is not None:
            self.access_counter += 1

        res = [self.embed[active_index], self.label[active_index].squeeze()]

        if self.return_index:
            res += [active_index]

        return res


def make_hdf5_dataset(
    hdf5_file_path,
    keys_order,
    indices_to_keep,
    index_key,
    unique_hash,
    max_chunk,
    total_samples,
    reverse_indices,
    return_index,
    logger
):
    return HDF5Dataset(
        hdf5_file_path,
        keys_order,
        indices_to_keep,
        index_key,
        unique_hash,
        max_chunk,
        total_samples,
        reverse_indices,
        return_index,
        logger
    )


def get_h5_dataloaders(
    train_batch_size,
    eval_batch_size,
    from_h5_config,
    num_workers,
    logger=None
):

    def make_datasets_dict(
        splits,
        paths_dict,
        keys_order,
        indices_to_keep,
        index_key,
        dict_for_hash,
        max_chunk,
        total_samples,
        reverse_indices,
        return_index,
        logger
    ):
        return {
            split: make_hdf5_dataset(
                get_with_assert(paths_dict, split),
                keys_order,
                extract_subset_indices(indices_to_keep, split),
                index_key,
                (
                    None
                    if dict_for_hash is None
                    else get_hash(get_with_assert(dict_for_hash, split))
                ),
                max_chunk=max_chunk,
                total_samples=total_samples.get(split),
                reverse_indices=reverse_indices.get(split, False),
                return_index=return_index,
                logger=logger
            ) for split in splits
        }

    keys_order = get_with_assert(from_h5_config, "keys_order")
    paths_dict = get_with_assert(from_h5_config, "path")
    indices_to_keep = from_h5_config.get("indices_to_keep", {})
    index_key = from_h5_config.get("index_key", "index")
    dict_for_hash = from_h5_config.get("dict_for_hash")
    max_chunk = from_h5_config.get("max_chunk", MAX_CHUNK_SIZE)
    total_samples = from_h5_config.get("total_samples", {})
    reverse_indices = from_h5_config.get("reverse_indices", {})
    return_index = from_h5_config.get("return_index", False)

    if max_chunk is not None:
        assert num_workers == 0, \
            "max_chunk is not compatible with num_workers > 0"

    if train_batch_size > 0:
        train_datasets_dict = make_datasets_dict(
            ["train"],
            paths_dict,
            keys_order,
            indices_to_keep,
            index_key,
            dict_for_hash,
            max_chunk=max_chunk,
            total_samples=total_samples,
            reverse_indices=reverse_indices,
            return_index=return_index,
            logger=logger
        )
    else:
        train_datasets_dict = None
    if eval_batch_size > 0:
        eval_splits = get_with_assert(from_h5_config, "eval_splits")
        for eval_split in eval_splits:
            assert eval_split in paths_dict
        eval_datasets_dict = make_datasets_dict(
            eval_splits,
            paths_dict,
            keys_order,
            indices_to_keep,
            index_key,
            dict_for_hash,
            max_chunk=max_chunk,
            total_samples=total_samples,
            reverse_indices=reverse_indices,
            return_index=return_index,
            logger=logger
        )
    else:
        eval_datasets_dict = None
    train_dataloaders, eval_dataloaders = get_generic_train_eval_dataloaders(
        train_datasets_dict,
        eval_datasets_dict,
        train_batch_size,
        eval_batch_size,
        shuffle_train=True,
        shuffle_eval=False,
        num_workers=num_workers
    )

    if train_dataloaders is not None:
        train_dataloader = train_dataloaders["train"]
    else:
        train_dataloader = None

    return train_dataloader, eval_dataloaders


def make_hdf5_name(unique_hash, suffix):
    return f"{unique_hash}{SEP}{suffix}.hdf5"


def extract_hdf5_hash(hdf5_name):
    return hdf5_name.split(".")[0].split(SEP)[0]
