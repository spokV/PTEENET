from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
from typing import Any, Callable, Optional, Tuple
import ipdb
#import color_convert

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import torchvision.transforms.functional as tf


class CIFAR10_noisy(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ]

    validation_list = [
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            validate: bool = False,
            add_noise: bool = False,
            noise_levels: [] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10_noisy, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.validate = validate
        self.noise_levels = noise_levels
        self.add_noise = add_noise

        if download:
            self.download()
        
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if self.train:
            downloaded_list = self.train_list
        elif self.validate:
            downloaded_list = self.validation_list
        else: 
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.levels = []

        # now load the picked numpy arrays
        for level in range(len(self.noise_levels)):
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    #ipdb.set_trace(context=6)
                    self.levels.extend(np.full(shape=len(entry['data']), fill_value=self.noise_levels[level], dtype=np.int))
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _add_noise_and_normalize(self, image, req_snr):
        mean_i = (0.4914, 0.4822, 0.4465)
        std_i = (0.2023, 0.1994, 0.2010)
        #ipdb.set_trace(context=6)

        #for t, m, s in zip(image, mean_i, std_i):
        #    t.mul_(s).add_(m)
        
        """
        %matplotlib inline

        # Generate the noise as you did
        im = image[0].detach().cpu().numpy()
        #ipdb.set_trace(context=6)
        im = np.transpose(im,(1,2,0))
        plt.imshow(im) 
        plt.show()
        #ipdb.set_trace(context=6)
        """

        ch, row, col = image.shape
        mean_n = 0
        var_n = 0.1
        sigma = var_n**0.5
        noise = torch.normal(mean_n,sigma,(ch, row, col))
        snr = 10.0 ** (req_snr / 10.0)
        #ipdb.set_trace(context=6)
        # work out the current SNR
        current_snr = torch.mean(image) / torch.std(noise)

        # scale the noise by the snr ratios (smaller noise <=> larger snr)
        noise = noise * (current_snr / snr)
        img_noise = image + noise
        
        """
        im = img_noise[0].detach().cpu().numpy()
        im = np.transpose(im,(1,2,0))
        plt.imshow(im) 
        plt.show()
        ipdb.set_trace(context=6)
        """
        here()
        #utils.save_image(args, img_noise, 'image1')
        #img_noise = tf.normalize(img_noise, mean_i, std_i)
        return img_noise

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, level = self.data[index], self.targets[index], self.levels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.add_noise:
            img = self._add_noise_and_normalize(img, level)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target#, level


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
    
    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
