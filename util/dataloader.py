import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class RadioImageList:
    """
    Construct a dataset based on RadImageNet.
    For more details, please refer to https://github.com/BMEII-AI/RadImageNet.
    """
    def __init__(self, image_lists, flag, inference, gray=True):
        self.imgs = image_lists
        self.class_list = [
            'abd_normal_ct', 'Airspace_opacity', 'bladder_pathology_ct', 'bowel_abnormality_ct',
            'Bronchiectasis', 'interstitial_lung_disease', 'liver_lesion_ct', 'lung_normal',
            'Nodule', 'osseous_neoplasm_ct', 'ovarian_pathology_ct', 'pancreatic_lesion_ct',
            'prostate_lesion_ct', 'renal_lesion_ct', 'splenic_lesion_ct', 'uterine_pathology_ct'
        ]
        self.label_list = list(np.arange(len(self.class_list)))
        self.current_labels = torch.tensor(self.get_label(self.imgs))
        self.global_weights = self.get_global_classw()
        self.gray = gray

        # Define transforms for training or evaluation.
        if flag == "train" and not inference:
            if gray:
                self.transforms = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1))
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1))
                ])
        else:
            self.transforms = None

        self.resize = transforms.Resize((32, 32))
        self.toGray = transforms.Grayscale()

    def get_global_classw(self):
        """
        Compute global class weights based on label frequency.
        """
        y_count = torch.bincount(self.current_labels)
        cate_w = y_count.sum() / (y_count * len(self.class_list))
        return cate_w

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.current_labels[index]
        img = read_image(img_path)

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            if self.gray:
                img = self.toGray(img)

        fingerprints = self.resize(img)
        onehot_target = self.encode_onehot(target)
        # Normalize the images to [0, 1]
        return img / 255., fingerprints / 255., onehot_target, index, img_path

    def normalize_img(self, img):
        """
        Normalize the image with per-channel mean and std.
        """
        mean = img.mean([1, 2], keepdim=True)
        std = img.std([1, 2], keepdim=True) + 1e-8
        return (img - mean) / std

    def encode_onehot(self, label):
        """
        Convert a label index to a one-hot encoded tensor.
        """
        one_hot = torch.zeros(len(self.class_list))
        one_hot[label] = 1
        return one_hot

    def __len__(self):
        return len(self.imgs)

    def get_label(self, fids):
        """
        Retrieve labels from file paths.
        Assumes the category is the directory name after 'retrieval/'.
        """
        labels = []
        for fid in fids:
            cate_id = fid.split('retrieval/')[-1].split('/')[0]
            labels.append(self.label_list[self.class_list.index(cate_id)])
        return labels


def seed_worker(worker_id):
    """
    Seed the NumPy random generator for DataLoader workers.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def get_radio_data(config, root_path, inference_flat=False, numworkers=1, gray=True, shuffle=True):
    """
    Create DataLoaders for training, validation, and testing datasets.

    Args:
        config (dict): Configuration parameters.
        root_path (str): Root directory for the image dataset.
        inference_flat (bool): Whether to process data for inference only.
        numworkers (int): Number of workers for the DataLoader.
        gray (bool): Convert images to grayscale if True.
        shuffle (bool): Whether to shuffle the dataset for training.

    Returns:
        tuple: DataLoaders for train, valid, test along with their respective lengths and class weights.
    """
    dsets = {}
    dset_loaders = {}
    class_weights = {}
    image_lists = []

    # Collect all image file paths.
    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            image_lists.append(os.path.join(root, name))

    # Split dataset into train, validation, and test sets.
    Train_fid, Valid_test_fid = train_test_split(image_lists, test_size=0.3, random_state=0)
    Valid_fid, Test_fid = train_test_split(Valid_test_fid, test_size=1 / 3, random_state=0)

    fid_lists = [Train_fid, Valid_fid, Test_fid]
    flags = ["train", "valid", "test"]

    for i in range(len(flags)):
        dsets[flags[i]] = RadioImageList(fid_lists[i], flag=flags[i], inference=inference_flat, gray=gray)
        class_weights[flags[i]] = dsets[flags[i]].global_weights
        shuffle_flag = True if flags[i] == 'train' else False
        print(f"{flags[i]}: {len(dsets[flags[i]])}")
        dset_loaders[flags[i]] = DataLoader(
            dsets[flags[i]],
            batch_size=config["batch_size"],
            shuffle=shuffle_flag,
            num_workers=numworkers,
            worker_init_fn=seed_worker
        )

    return (
        dset_loaders["train"], dset_loaders["valid"], dset_loaders["test"],
        len(dsets["train"]), len(dsets["valid"]), len(dsets["test"]),
        class_weights
    )
