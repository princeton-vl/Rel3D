from os import path
import math
import random
import json
import io
from collections import defaultdict

import h5py
import pdb
import tqdm
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from runstats import Statistics

# always produce image of size RESIZE_DIM * RESIZE_DIM for training
RESIZE_DIM = 224
# the size of the image we store,
# this is done so that we can do shift data augmentation
EXTRA_RESIZE_DIM = 256

TRANS_VEC_STR = "transform_vector"
TRANS_VEC_KEY = ["raw_absolute", "aligned_absolute", "aligned_relative"]
TRANS_VEC_DIM = {
    "raw_absolute": 24,
    "aligned_absolute": 18,
    "aligned_relative": 12
}


class SpatialDataset(Dataset):
    def __init__(self, split, predicate_dim, object_dim, data_path, load_img,
                 data_aug_shift, data_aug_color, crop, norm_data, resize_mask,
                 trans_vec):
        """
        :param split: which split of the json to use
        :param predicate_dim: number of predicates
        :param object_dim: number of object
        :param data_path: path to the json file
        :param load_img: whether to load image or not
        :param data_aug_shift: whether to do shift data augmentation
        :param data_aug_color: whether to do color data augmentation
        :param crop: whether to crop the image to only fit the relations
        :param norm_data: whether to normalize the data before feeding
        :param resize_mask: whether or not to resize the mask before feeding
        :param trans_vec: if empty list, no transform vector is loaded.
            if not none, it is list of string containing the
            name of the transformation vectors to add
        """
        super().__init__()
        data = json.load(open(data_path))
        self.split = split
        self.relations = data[split]

        self.predicates = data['predicates']
        self.objects = data['objects']

        assert len(self.predicates) == predicate_dim
        assert len(self.objects) == object_dim

        # printing the dataset statistics
        print('%d relations in %s' % (len(self.relations), split))
        num_pos = 0
        num_neg = 0
        num_rel_pos = defaultdict(int)
        num_rel_neg = defaultdict(int)
        for sample in self.relations:
            if sample['label']:
                num_pos += 1
                num_rel_pos[sample['predicate']] += 1
            else:
                num_neg += 1
                num_rel_neg[sample['predicate']] += 1
        all_rel = list(set(num_rel_neg) | set(num_rel_pos))
        assert set(all_rel) == set(self.predicates)

        print(f"Percentage of positive labels  in {split}:"
              f" {round(num_pos / (num_neg + num_pos), 3)}")
        print(f"Percentage of negative labels  in {split}:"
              f" {round(num_neg / (num_neg + num_pos), 3)}")
        print({
            x: (num_rel_pos[x], num_rel_neg[x],
                round(num_rel_pos[x] / (num_rel_pos[x] + num_rel_neg[x]), 3))
            for x in all_rel
        })

        # reweighing the weights so that the average weight is one
        # this is useful as the loss would remain in the same range
        # as before if these weights are used
        self.pred_weight = {
            x: 1 / (num_rel_pos[x] + num_rel_neg[x])
            for x in all_rel
        }
        sum_pred_weights = sum(list(self.pred_weight.values()))
        for x in self.pred_weight:
            self.pred_weight[x] = (
                self.pred_weight[x] / sum_pred_weights) * len(self.pred_weight)
        assert (sum(self.pred_weight.values()) - len(self.predicates)) < 1e-3

        self.load_img = load_img
        self.data_aug_shift = data_aug_shift
        self.data_aug_color = data_aug_color
        self.crop = crop
        self.norm_data = norm_data
        self.resize_mask = resize_mask

        assert split in ["train", "test", "valid"]
        if (self.data_aug_shift or self.data_aug_color) and split != "train":
            print("WARNING: Doing data augmentation for not train set")
        if ((not (self.data_aug_shift or self.data_aug_color))
                and split == "train"):
            print("WARNING: Not doing data augmentation for train set")
        if self.load_img:
            print(f"Loading image for {split} set")
        else:
            print(f"Not loading image for {split} set")
        if self.data_aug_shift:
            print(f"Doing shift data augmentation for {split} set")
        else:
            print(f"Not doing shift data augmentation for {split} set")
        if self.data_aug_color:
            print(f"Doing color data augmentation for {split} set")
        else:
            print(f"Not doing color data augmentation for {split} set")

        if self.crop:
            print(f"Doing image cropping for {split} set")
        else:
            print(f"Not doing image cropping for {split} set")
        if self.norm_data:
            print(f"Normalizing data for {split} set")
        else:
            print(f"Not normalizing rgb for {split} set")

        self.trans_vec = trans_vec
        if self.trans_vec:
            for x in self.trans_vec:
                assert x in TRANS_VEC_KEY

            h5_stats_path = f"{data_path[0:-5]}_stats_trans.h5"

            if split == "train" and not path.exists(h5_stats_path):
                print(f"Saving the trans stat at {h5_stats_path}")
                self.save_trans_stats(h5_stats_path)

            with h5py.File(h5_stats_path, 'r') as h5_stats:
                self.trans_stats = {}
                for key in TRANS_VEC_KEY:
                    self.trans_stats[key] = {
                        'mean': h5_stats[f'{key}_mean'][()],
                        'std': h5_stats[f'{key}_std'][()]
                    }

        assert data_path[-5:] == ".json"
        if self.load_img:
            h5_path = f"{data_path[0:-5]}_{split}_{self.crop}.h5"
            # a separate file so that validation and test data loaders
            # can also read it
            h5_stats_path = f"{data_path[0:-5]}_stats_{self.crop}.h5"
            # WARNING: first run train dataloader once before running
            # the test or validation data loader
            if ((not path.exists(h5_path))
                    or (split == "train" and not path.exists(h5_stats_path))):
                print(f"Saving the h5 at {h5_path}")
                if split == "train":
                    self.save_h5(h5_path, h5_stats_path)
                else:
                    self.save_h5(h5_path)

            print("Reading all the data into memory")
            with h5py.File(h5_path, 'r') as h5:
                self.rgb_dataset = h5['rgb'][()]
                self.depth_dataset = h5['depth'][()]
                self.bbox_dataset = h5['bbox'][()]
                self.rgb_path_to_id = json.loads(h5['rgb_path_to_id'][()])
                self.depth_path_to_id = json.loads(h5['depth_path_to_id'][()])
                assert len(self.depth_path_to_id) == len(self.rgb_path_to_id)

            with h5py.File(h5_stats_path, 'r') as h5_stats:
                self.img_mean = h5_stats['img_mean'][()]
                self.depth_mean = h5_stats['depth_mean'][()]
                self.bbox_mean = h5_stats['bbox_mean'][()]
                self.img_std = h5_stats['img_std'][()]
                self.depth_std = h5_stats['depth_std'][()]
                self.bbox_std = h5_stats['bbox_std'][()]

    def save_trans_stats(self, h5_stats_path):
        stats = {
            key: {'mean': 0.0, 'std': 0.0}
            for key in TRANS_VEC_KEY
        }
        for key in TRANS_VEC_KEY:
            all_data = np.zeros((len(self.relations), TRANS_VEC_DIM[key]))
            for i, rel in enumerate(self.relations):
                all_data[i] = np.array(rel[TRANS_VEC_STR][key])

            stats[key]['mean'] = np.mean(all_data, 0)
            stats[key]['std'] = np.std(all_data, 0)

        with h5py.File(h5_stats_path, 'w') as hf:
            for key in stats:
                for key2 in stats[key]:
                    print(f'{key}_{key2}')
                    hf.create_dataset(f'{key}_{key2}',
                                      data=stats[key][key2])

    def save_h5(self, h5_path, h5_stats_path=None):
        """
        Preprocess the images and save them in h5 format. Based on whether crop
        is True or False, we first find the area of the image we want to store.
        The assumption is the we would resize this relevant are to size
        224*224.  To support shift data augmentation, we we instead save an
        image of size 256*256, such that the central 224*224 area corresponds
        to the part of the image we actually desire.
        :param h5_path: where to save
        :param h5_stats_path: save stats if not none
        :return:
        """
        if h5_stats_path is not None:
            r_sta = Statistics()
            g_sta = Statistics()
            b_sta = Statistics()
            depth_sta = Statistics()
            sub_sta = Statistics()
            obj_sta = Statistics()

        with h5py.File(h5_path, 'w') as hf:
            # storing rbg images images in compressed format
            # for saving disk space
            data_type = h5py.special_dtype(vlen=np.dtype('uint8'))
            rgb_dataset = hf.create_dataset('rgb', (len(self.relations), ),
                                            dtype=data_type)
            depth_dataset = hf.create_dataset('depth', (len(self.relations), ),
                                              dtype=data_type)
            bbox_dataset = hf.create_dataset('bbox', (len(self.relations), ),
                                             dtype=data_type)
            rgb_path_to_id = {}
            depth_path_to_id = {}

            for i, sample in tqdm.tqdm(enumerate(self.relations)):
                img = Image.open(sample['rgb'])
                depth = Image.open(sample['depth'])

                rgb_path_to_id[sample['rgb']] = i
                depth_path_to_id[sample['depth']] = i

                width = sample['width']
                height = sample['height']
                subj_bbox = self.convert_bbox(sample['subject']['bbox'])
                obj_bbox = self.convert_bbox(sample['object']['bbox'])

                bbox_mask = self.get_bbox_mask(height, width, subj_bbox,
                                               obj_bbox)
                padded_bbox_mask = Image.fromarray(np.concatenate(
                    [bbox_mask, np.zeros((height, width, 1), dtype=np.uint8)],
                    2))

                if self.crop:
                    union_bbox = self.get_union_bbox(subj_bbox, obj_bbox)
                else:
                    union_bbox = (0, height, 0, width)

                union_top, union_bottom, union_left, union_right = union_bbox
                union_width = union_right - union_left
                union_height = union_bottom - union_top

                crop_width = int(
                    (EXTRA_RESIZE_DIM / RESIZE_DIM) * union_width)
                crop_height = int(
                    (EXTRA_RESIZE_DIM / RESIZE_DIM) * union_height)
                left_extra_width = int(
                    (crop_width - union_width) / 2)
                right_extra_width = int(
                    (crop_width - union_width) - left_extra_width)
                top_extra_height = int(
                    (crop_height - union_height) / 2)
                bottom_extra_height = int(
                    (crop_height - union_height) - top_extra_height)

                pad = (left_extra_width, top_extra_height,
                       right_extra_width, bottom_extra_height)
                img_extra = TF.pad(img, padding=pad)
                depth_extra = TF.pad(depth, padding=pad)
                padded_bbox_mask_extra = TF.pad(padded_bbox_mask, padding=pad)

                img_crop = TF.crop(img_extra, union_top, union_left,
                                   crop_height, crop_width)
                depth_crop = TF.crop(depth_extra, union_top, union_left,
                                     crop_height, crop_width)
                padded_bbox_mask_crop = TF.crop(
                    padded_bbox_mask_extra, union_top, union_left, crop_height,
                    crop_width)

                img_crop = TF.resize(img_crop,
                                     (EXTRA_RESIZE_DIM, EXTRA_RESIZE_DIM))
                depth_crop = TF.resize(depth_crop,
                                       (EXTRA_RESIZE_DIM, EXTRA_RESIZE_DIM))
                padded_bbox_mask_crop = TF.resize(
                    padded_bbox_mask_crop, (EXTRA_RESIZE_DIM,
                                            EXTRA_RESIZE_DIM))

                rgb_dataset[i] = self.compress_png(img_crop, is_unit8=True)
                depth_crop = self.convert_img_uint8(depth_crop)
                depth_dataset[i] = self.compress_png(depth_crop, is_unit8=True)
                bbox_dataset[i] = self.compress_png(padded_bbox_mask_crop,
                                                    is_unit8=True)

                if h5_stats_path is not None:
                    # calculating statistics of the center region
                    _rgb = np.array(
                        TF.to_tensor(TF.center_crop(img_crop, RESIZE_DIM)))
                    _depth = np.array(
                        TF.to_tensor(TF.center_crop(depth_crop, RESIZE_DIM)))
                    _bbox = np.array(
                        TF.to_tensor(TF.center_crop(padded_bbox_mask_crop,
                                                    RESIZE_DIM)))

                    assert _rgb.shape[0] == 3
                    assert _bbox.shape[0] == 3
                    assert _depth.shape[0] == 1
                    self.push_list(r_sta, _rgb[0].ravel())
                    self.push_list(g_sta, _rgb[1].ravel())
                    self.push_list(b_sta, _rgb[2].ravel())
                    self.push_list(depth_sta, _depth[0].ravel())
                    self.push_list(sub_sta, _bbox[0].ravel())
                    self.push_list(obj_sta, _bbox[1].ravel())

            hf.create_dataset('rgb_path_to_id',
                              data=json.dumps(rgb_path_to_id))
            hf.create_dataset('depth_path_to_id',
                              data=json.dumps(depth_path_to_id))

        if h5_stats_path is not None:
            with h5py.File(h5_stats_path, 'w') as hf:
                hf.create_dataset(
                    'img_mean',
                    data=[r_sta.mean(), g_sta.mean(), b_sta.mean()])
                hf.create_dataset(
                    'img_std',
                    data=[r_sta.stddev(), g_sta.stddev(), b_sta.stddev()])
                hf.create_dataset('depth_mean', data=[depth_sta.mean()])
                hf.create_dataset('depth_std', data=[depth_sta.stddev()])
                hf.create_dataset(
                    'bbox_mean',
                    data=[sub_sta.mean(), obj_sta.mean()])
                hf.create_dataset(
                    'bbox_std',
                    data=[sub_sta.stddev(), obj_sta.stddev()])

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx, visualize=False):
        rel = self.relations[idx]
        width = rel['width']
        height = rel['height']

        subj_bbox = self.convert_bbox(rel['subject']['bbox'])
        obj_bbox = self.convert_bbox(rel['object']['bbox'])
        union_bbox = self.get_union_bbox(subj_bbox, obj_bbox)
        subj_t = self._getT(subj_bbox, obj_bbox)
        obj_t = self._getT(obj_bbox, subj_bbox)

        example = {
            'subject': {
                'name': rel['subject']['name'],
                'idx': self.objects.index(rel['subject']['name']),
                'bbox': np.array([
                    subj_bbox[0] / height, subj_bbox[1] / height,
                    subj_bbox[2] / width, subj_bbox[3] / width
                ], dtype=np.float32),  # x0, x1, y0, y1
                't': np.array(subj_t, dtype=np.float32),
            },
            'object': {
                'name': rel['object']['name'],
                'idx': self.objects.index(rel['object']['name']),
                'bbox': np.array([
                    obj_bbox[0] / height, obj_bbox[1] / height,
                    obj_bbox[2] / width, obj_bbox[3] / width
                ], dtype=np.float32),
                't': np.array(obj_t, dtype=np.float32),
            },
            'predicate': {
                'name': rel['predicate'],
                'idx': self.predicates.index(rel['predicate']),
                'bbox': np.array([
                    union_bbox[0] / height, union_bbox[1] / height,
                    union_bbox[2] / width, union_bbox[3] / width
                ], dtype=np.float32),
            },
            'label': rel['label'],
            'rgb_source': rel['rgb'],
            'weight': self.pred_weight[rel['predicate']]
        }

        if self.load_img:
            sample_id = self.rgb_path_to_id[rel['rgb']]
            assert sample_id == self.depth_path_to_id[rel['depth']]
            _img = self.decompress_png(
                self.rgb_dataset[sample_id], return_unit8=True)
            _depth = self.decompress_png(
                self.depth_dataset[sample_id], return_unit8=True)
            _padded_bbox_mask = self.decompress_png(
                self.bbox_dataset[sample_id], return_unit8=True)
            img = Image.fromarray(_img)
            depth = Image.fromarray(_depth)
            padded_bbox_mask = Image.fromarray(_padded_bbox_mask)
            if self.data_aug_shift:
                crop_left = random.randint(
                    0, EXTRA_RESIZE_DIM - RESIZE_DIM - 1)
                crop_top = random.randint(
                    0, EXTRA_RESIZE_DIM - RESIZE_DIM - 1)
                img_crop = TF.crop(
                    img, crop_top, crop_left, RESIZE_DIM, RESIZE_DIM)
                depth_crop = TF.crop(
                    depth, crop_top, crop_left, RESIZE_DIM, RESIZE_DIM)
                padded_bbox_mask_crop = TF.crop(
                    padded_bbox_mask, crop_top, crop_left,
                    RESIZE_DIM, RESIZE_DIM)
            else:
                img_crop = TF.center_crop(img, RESIZE_DIM)
                depth_crop = TF.center_crop(depth, RESIZE_DIM)
                padded_bbox_mask_crop = TF.center_crop(
                    padded_bbox_mask, RESIZE_DIM)

            if self.data_aug_color:
                img_crop = TF.adjust_brightness(
                    img_crop, random.uniform(0.9, 1.1))
                img_crop = TF.adjust_contrast(
                    img_crop, random.uniform(0.9, 1.1))
                img_crop = TF.adjust_gamma(img_crop, random.uniform(0.9, 1.1))
                img_crop = TF.adjust_hue(img_crop, random.uniform(-0.05, 0.05))

            if visualize:
                vis = Image.new(mode='RGB', size=(2 * 224, 2 * 224))
                vis.paste(img_crop, (0, 0))
                depth_crop_arr = (255 * np.array(
                    depth_crop, dtype=np.float32)).astype(np.uint8)
                vis.paste(Image.fromarray(depth_crop_arr), (224, 0))
                mask_arr = np.array(
                    TF.resize(padded_bbox_mask_crop, (224, 224)))
                vis.paste(
                    Image.fromarray(np.hstack([mask_arr[:, :, 0],
                                               mask_arr[:, :, 1]])),
                    (0, 224))
                draw = ImageDraw.Draw(vis)
                # font = ImageFont.truetype("sans-serif.ttf", 16)
                draw.text(
                    (0, 0),
                    f"{example['subject']['name']}"
                    f"--{example['predicate']['name']}"
                    f"--{example['object']['name']}"
                    f"--{example['label']}",
                    (255, 255, 255),
                    # font=font)
                )
                vis.save('vis_%04d.jpg' % idx)

            img_crop = TF.to_tensor(img_crop)
            depth_crop = TF.to_tensor(depth_crop)

            if self.resize_mask:
                padded_bbox_mask_crop = TF.resize(
                    padded_bbox_mask_crop, (32, 32))
            padded_bbox_mask_crop = TF.to_tensor(padded_bbox_mask_crop)
            bbox_mask_crop = padded_bbox_mask_crop[:2]
            example['subject']['bbox'] = self.get_rel_coo_from_mask(
                bbox_mask_crop[0])
            example['object']['bbox'] = self.get_rel_coo_from_mask(
                bbox_mask_crop[1])
            example['predicate']['bbox'] = np.array(
                self.get_union_bbox(example['subject']['bbox'],
                                    example['object']['bbox'])
            )

            if self.norm_data:
                img_crop = TF.normalize(
                    img_crop, mean=self.img_mean, std=self.img_std)
                depth_crop = TF.normalize(
                    depth_crop, mean=self.depth_mean, std=self.depth_std)
                bbox_mask_crop = TF.normalize(
                    bbox_mask_crop, mean=self.bbox_mean, std=self.bbox_std)

            example['img_crop'] = img_crop
            example['depth_crop'] = depth_crop
            example['bbox_mask'] = bbox_mask_crop

        if self.trans_vec:
            for key in self.trans_vec:
                example[key] = np.array(
                    rel[TRANS_VEC_STR][key]).astype('float32')
                if self.norm_data:
                    trans_mean = self.trans_stats[key]['mean']
                    trans_std = self.trans_stats[key]['std']
                    example[key] = (example[key] - trans_mean) / trans_std

        return example

    @staticmethod
    def push_list(sta, lst):
        for x in lst:
            sta.push(x)

    @staticmethod
    def convert_bbox(bbox):
        bbox = {k: float(v) for k, v in bbox.items()}
        return [bbox['top'], bbox['top'] + bbox['height'],
                bbox['left'], bbox['left'] + bbox['width']]

    @staticmethod
    def get_union_bbox(bbox_a, bbox_b):
        return [min(bbox_a[0], bbox_b[0]), max(bbox_a[1], bbox_b[1]),
            min(bbox_a[2], bbox_b[2]), max(bbox_a[3], bbox_b[3])]

    @staticmethod
    def get_bbox_mask(height, width, subj_bbox, obj_bbox):
        mask = np.zeros((height, width, 2), dtype=np.uint8)
        mask[int(subj_bbox[0]): int(subj_bbox[1]),
             int(subj_bbox[2]): int(subj_bbox[3]), 0] = 255
        mask[int(obj_bbox[0]): int(obj_bbox[1]),
             int(obj_bbox[2]): int(obj_bbox[3]), 1] = 255
        return mask

    def get_rel_coo_from_mask(self, mask):
        """
        get relative coordinates from mask
        :param mask:
        :return:
        """
        assert mask.dim() == 2
        h, w = mask.shape
        nz_mask = mask.nonzero()
        if len(nz_mask) != 0:
            t, l = nz_mask.min(0).values
            b, r = nz_mask.max(0).values
        # this can happen when we crop an object out of the image
        else:
            # Resolution: there are some samples with height=0, width=0
            # this happens when object is right at the edge
            # assert self.data_aug_shift
            # assert self.split == "train"
            t, l, b, r = [0]*4

        return np.array([
            float(t) / h, float(b) / h,
            float(l) / w, float(r) / w
        ], np.float32)

    @staticmethod
    def _getT(bbox1, bbox2):
        h1 = bbox1[1] - bbox1[0]
        w1 = bbox1[3] - bbox1[2]
        h2 = bbox2[1] - bbox2[0]
        w2 = bbox2[3] - bbox2[2]
        return [(bbox1[0] - bbox2[0]) / (float(h2) + 1e-5),
                (bbox1[2] - bbox2[2]) / (float(w2) + 1e-5),
                math.log(float(h1 + 1e-5) / (float(h2) + 1e-5)),
                math.log(float(w1 + 1e-5) / (float(w2) + 1e-5))]

    @staticmethod
    def convert_img_uint8(image):
        """
        Convert an image into uint8
        :param image:
        :return:
        """
        assert isinstance(image, Image.Image) or isinstance(image, np.ndarray)
        if isinstance(image, Image.Image):
            image = np.array(image)
        assert not image.dtype == 'uint8'
        assert image.dtype == 'float32'
        assert np.all(np.logical_and(image >= 0.0, image <= 1.0))
        return Image.fromarray((image * 255).round().astype(np.uint8))

    def compress_png(self, image, is_unit8):
        """
        :param image: a PIL image
        :param is_unit8: whether the image is unit8
        :return:
            compressed image as a numpy array
        """
        _image = np.array(image)

        if is_unit8:
            assert _image.dtype == 'uint8'
            assert np.all(np.logical_and(_image >= 0, _image <= 255))
        else:
            assert np.all(np.logical_and(_image >= 0.0, _image <= 1.0))
            assert _image.dtype == 'float32'
            image = self.convert_img_uint8(_image)

        with io.BytesIO() as b:
            imageio.imwrite(b, image, format='png')
            b.seek(0)
            image_comp = np.frombuffer(b.read(), dtype='uint8')

        return image_comp

    @staticmethod
    def decompress_png(image_comp, return_unit8):
        """
        :param image_comp: a PIL image
        :param return_unit8: whether the image is unit8
        :return:
            compressed image as a numpy array
        """
        assert image_comp.dtype == 'uint8'
        image = imageio.imread(io.BytesIO(image_comp))
        if not return_unit8:
            assert image
            image = image.astype(np.float32)
            image = image / 255

        return image


def create_dataloader(split, predicate_dim, object_dim, datapath, num_workers,
                      crop, norm_data, load_img, data_aug_shift,
                      data_aug_color, batch_size, resize_mask, trans_vec):
    dataset_args = {
        "split": split,
        "predicate_dim": predicate_dim,
        "object_dim": object_dim,
        "data_path": datapath,
        "load_img": load_img,
        "data_aug_shift": (data_aug_shift and split == "train"),
        "data_aug_color": (data_aug_color and split == "train"),
        "crop": crop,
        "norm_data": norm_data,
        "resize_mask": resize_mask,
        "trans_vec": trans_vec,
    }
    dataset = SpatialDataset(**dataset_args)
    return (
        DataLoader(
            dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            pin_memory=torch.cuda.is_available()),
        dataset.predicates,
        dataset.objects
    )


if __name__ == '__main__':
    _dataset_args = {
        "split": 'train',
        "predicate_dim": 30,
        "object_dim": 67,
        "data_path": 'data/c_0.9_c_0.1.json',
        "load_img": True,
        "data_aug_shift": False,
        "data_aug_color": False,
        "crop": True,
        "norm_data": False,
        "resize_mask": False,
        "trans_vec": [],
    }
    _dataset = SpatialDataset(**_dataset_args)
    for i in range(10):
        _dataset.__getitem__(10000 + i, visualize=True)
