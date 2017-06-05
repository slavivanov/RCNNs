import os
import pickle
import shutil

from misc import load_array, save_array
from torch.utils.data.dataset import Dataset as torchDataset


class Dataset(torchDataset):
    def __init__(self, cache_dir):
        super().__init__()
        self.set_cache_dir(cache_dir)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, n):
        raise NotImplementedError

    def load_cache(self, filename):
        """ Load data from cache."""
        fullpath = os.path.join(self.cache_dir, filename)
        if os.path.exists(fullpath + '.bcolz'):
            return load_array(fullpath + '.bcolz')
        elif os.path.exists(fullpath + '.pkl'):
            with open(fullpath + '.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError

    def save_cache(self, filename, to_save, use_pickle=False):
        """ Save data to cache."""
        fullpath = os.path.join(self.cache_dir, filename)
        if use_pickle or type(to_save) == dict:
            with open(fullpath + '.pkl', 'wb') as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            save_array(fullpath + '.bcolz', to_save)

    def set_cache_dir(self, dirs):
        """Set the directory where to save cached data."""
        # create the directories if needed
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        self.cache_dir = dirs

    def clear_cache(self, filename=None):
        """ Delete a specific file in cache dir or all of
            them if unspecified. """
        if filename:
            shutil.rmtree(os.path.join(self.cache_dir, filename))
        elif self.cache_dir:
            shutil.rmtree(self.cache_dir)
            # recreate the folder
            self.set_cache_dir(self.cache_dir)