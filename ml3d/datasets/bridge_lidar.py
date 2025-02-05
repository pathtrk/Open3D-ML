from pathlib import Path
from os.path import join, exists
import logging

import numpy as np
from .base_dataset import BaseDataset, BaseDatasetSplit
from .utils import DataProcessing as DP
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class BridgeLiDAR(BaseDataset):
    def __init__(self,
                 dataset_path,
                 name="BridgeLiDAR",
                 cache_dir='./logs/cache',
                 use_cache=False,
                 ignored_label_inds=[0],
                 val_files=['b_10'],
                 test_result_folder='./test',
                 **kwargs):
        # read file lists.
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         ignored_label_inds=ignored_label_inds,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.all_files = [str(p) for p in Path(
            self.cfg.dataset_path).iterdir() if p.suffix == '.npy']

        self.val_files = []
        self.test_files = []

        for _, file_path in enumerate(self.all_files):
            for val_file in cfg.val_files:
                if val_file in file_path:
                    self.val_files.append(file_path)
                    self.all_files.remove(file_path)
                    break

            for test_file in cfg.test_files:
                if test_file in file_path:
                    self.test_files.append(file_path)
                    self.all_files.remove(file_path)
                    break

        self.val_files = np.sort(self.val_files)
        self.test_files = np.sort(self.test_files)
        self.train_files = np.sort(self.all_files)

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'unlabeled',
            1: 'ground',
            2: 'deck',
            3: 'pier',
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return BridgeLiDARSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))
        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)

        with open(store_path, 'wb') as f:
            np.save(store_path, pred.astype(np.int32))

        log.info("Saved {} in {}.".format(name, store_path))


class BridgeLiDARSplit(BaseDatasetSplit):
    """This class is used to create a split for Semantic3D dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        with open(pc_path, 'rb') as f:
            pc = np.load(f)

        points = pc[:, 0:3].astype(np.float32)
        labels = pc[:, 3].astype(np.int32)

        data = {
            'point': points,
            'label': labels + 1,
            'feat': None
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(BridgeLiDAR)
