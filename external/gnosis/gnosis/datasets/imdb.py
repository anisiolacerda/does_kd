"""Adapted from https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py"""

from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torchtext.data.datasets_utils import _create_dataset_directory
import io
import os

URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

NUM_LINES = {
    'train': 25000,
    'test': 25000,
}

_PATH = 'aclImdb_v1.tar.gz'

DATASET_NAME = "IMDB"


def get_abs_path_list(root):
    files_list = os.listdir(root)
    return [os.path.join(root, f) for f in files_list]


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def IMDB(root, split, download=False):
    def generate_imdb_data(key, extracted_files):
        for fname in extracted_files:
            if 'urls' in fname:
                continue
            elif key in fname and ('pos' in fname or 'neg' in fname):
                with io.open(fname, encoding="utf8") as f:
                    label = 'pos' if 'pos' in fname else 'neg'
                    yield label, f.read()

    if download:
        dataset_tar = download_from_url(URL, root=root,
                                        hash_value=MD5, hash_type='md5')
        extracted_files = extract_archive(dataset_tar)
    else:
        extracted_files = get_abs_path_list(os.path.join(root, "aclImdb/{}/{}".format(split, "pos"))) + \
                     get_abs_path_list(os.path.join(root, "aclImdb/{}/{}".format(split, "neg")))
    iterator = generate_imdb_data(split, extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], iterator)
