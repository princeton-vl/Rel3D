from collections import defaultdict, OrderedDict, MutableMapping, Hashable
import os
import pdb
import sys
import random
import math
import csv
import tensorboardX


class TensorboardManager:
    def __init__(self, path):
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            self.writer.add_scalar('%s_%s' % (split, k), v, step)


def ds_samp_to_sub_obj(sample):
    inp = (
        sample['subject']['name'],
        sample['object']['name']
    )
    return inp


def get_rel_to_inp_dict(dataset):
    """
    :param dataset: for example data['train'] loaded form real.json
    :return:
        {
            'on': {
                 ('table', 'chair') : {
                    True: [{"rgb": "fd.jpg", "depth": "fd.tiff", "width": 1280, "height": 720,
                        "subject": {"name": "table", "bbox": {"left": "575", "top": "424", "width": "428", "height": "69"}},
                        "predicate": "on", "object": {"name": "chair", "bbox": {"left": "701", "top": "609", "width": "448", "height": "122"}},
                        "label": True}],
                    False: [{"rgb": "fe.jpg", "depth": "fe.tiff", "width": 1280, "height": 720,
                        "subject": {"name": "table", "bbox": {"left": "575", "top": "424", "width": "428", "height": "69"}},
                        "predicate": "on", "object": {"name": "chair", "bbox": {"left": "701", "top": "609", "width": "448", "height": "122"}},
                        "label": False}],
                 }
                 .
                 .
                 .
            }
            .
            .
            .

        }
    """
    rel_to_inp_dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)))

    for ds_samp in dataset:
        sub_obj = ds_samp_to_sub_obj(ds_samp)
        rel = ds_samp['predicate']
        label = ds_samp['label']
        rel_to_inp_dict[rel][sub_obj][label].append(ds_samp)

    return rel_to_inp_dict


def create_smaller_dataset(dataset, per_train, per_valid, contrastive=True, seed=26):
    """
    Create a smaller train and validation dataset from the given train dataset
    The dataset input must be contrastive
    :param dataset: for example data['train'] loaded form real.json, must be a contrastive dataset
    :param per_train: percentage of samples to be kept in the smaller dataset train dataset
    :param per_valid: percentage of samples to be kept in the validation dataset
    :param contrastive: whether the samples are chosen as contrastive pair or individually
    :param seed: random seed
    :return: new dataset
    """

    if contrastive:
        assert per_train + per_valid <= 1.0
    else:
        assert per_train + per_valid <= 0.5

    _rel_to_inp_dict = get_rel_to_inp_dict(dataset)
    # ordering so that random seed is applied in order
    rel_to_inp_dict = OrderedDict()
    for x in sorted(_rel_to_inp_dict):
        inp_dict = OrderedDict()
        for y in sorted(_rel_to_inp_dict[x]):
            inp_dict[y] = _rel_to_inp_dict[x][y]
        rel_to_inp_dict[x] = inp_dict

    print([x for x in rel_to_inp_dict])
    print([x for x in rel_to_inp_dict['near']])

    train_dataset = []
    valid_dataset = []
    i = 0
    for relation in rel_to_inp_dict:
        # changing seed in each iteration
        random.seed(seed + i)
        i += 1

        sub_obj_list = [x for x in rel_to_inp_dict[relation]]
        random.shuffle(sub_obj_list)

        if contrastive:
            num_train = math.ceil(len(sub_obj_list) * per_train)
            num_valid = math.ceil(len(sub_obj_list) * per_valid)
        else:
            num_train = min(2 * math.ceil(len(sub_obj_list) * per_train),
                            len(sub_obj_list))
            num_valid = min(2 * math.ceil(len(sub_obj_list) * per_valid),
                            len(sub_obj_list))

        # a bug might exist here
        if num_train + num_valid > len(sub_obj_list):
            num_valid -= (num_train + num_valid) - len(sub_obj_list)

        # case when total number of examples are very small
        # like sub_obj_list = 2 and per = 0.8
        # we then add one sample to the complement as well
        if (not per_valid == 0.0) and num_train == len(sub_obj_list):
            assert len(sub_obj_list) > 1
            num_train -= 1
            num_valid += 1

        sub_obj_list_train = sub_obj_list[0: num_train]
        sub_obj_list_valid = sub_obj_list[num_train: num_train + num_valid]

        for sub_obj in sub_obj_list_train:
            pos_neg_dict = rel_to_inp_dict[relation][sub_obj]
            assert (True in pos_neg_dict) and (False in pos_neg_dict)

            if contrastive:
                train_dataset.extend(
                    pos_neg_dict[True] + pos_neg_dict[False])
            else:
                true_false = random.choice([True, False])
                train_dataset.extend(pos_neg_dict[true_false])

        for sub_obj in sub_obj_list_valid:
            pos_neg_dict = rel_to_inp_dict[relation][sub_obj]
            assert (True in pos_neg_dict) and (False in pos_neg_dict)
            if contrastive:
                valid_dataset.extend(
                    pos_neg_dict[True] + pos_neg_dict[False])
            else:
                true_false = random.choice([True, False])
                valid_dataset.extend(pos_neg_dict[true_false])

        # if relation == "to the right of":
        #     pdb.set_trace()

    print(f"Num of sample initially in train: {len(dataset)}")
    print(f"Num of sample finally in train: {len(train_dataset)}")
    print(f"Num of sample finally in valid: {len(valid_dataset)}")

    # all the tests
    rel_to_inp_dict = get_rel_to_inp_dict(dataset)
    train_rel_to_inp_dict = get_rel_to_inp_dict(train_dataset)
    valid_rel_to_inp_dict = get_rel_to_inp_dict(valid_dataset)
    # pdb.set_trace()
    # same relations
    assert set(train_rel_to_inp_dict.keys()) == set(rel_to_inp_dict.keys())
    assert ((per_valid == 0.0)
            or set(train_rel_to_inp_dict.keys()) == set(valid_rel_to_inp_dict.keys()))
    for rel in rel_to_inp_dict:
        inp_dict = rel_to_inp_dict[rel]
        train_inp_dict = train_rel_to_inp_dict[rel]
        valid_inp_dict = {} if per_valid == 0.0 else valid_rel_to_inp_dict[rel]
        # about the sub_obj for each relation
        assert set(inp_dict.keys()).issuperset(set(train_inp_dict.keys()))
        assert set(inp_dict.keys()).issuperset(set(valid_inp_dict.keys()))
        # nothing common between train and validation set
        assert len(set(train_inp_dict.keys()) & set(valid_inp_dict.keys())) == 0
        if contrastive:
            for _, true_false in train_inp_dict.items():
                # whether contrastive pair present for each example
                assert True in true_false
                assert False in true_false
            if not per_valid == 0.0:
                for _, true_false in valid_inp_dict.items():
                    # whether contrastive pair present for each example
                    assert True in true_false
                    assert False in true_false
        else:
            for _, true_false in train_inp_dict.items():
                # either True or False sample present
                assert (True in true_false) or (False in true_false)
                assert not ((False in true_false) and (True in true_false))
            if not per_valid == 0.0:
                for _, true_false in valid_inp_dict.items():
                    # either True or False sample present
                    assert (True in true_false) or (False in true_false)
                    assert not ((False in true_false) and (True in true_false))

    return train_dataset, valid_dataset


class RecordExp:
    def __init__(self, file_name):
        self.file_name = file_name
        self.param_recorded = False
        self.result_recorded = False

    def record_param(self, param_dict):
        """
        all parameters must be given at the same time. parameters must be given
        before the results
        :return:
        """
        assert not self.param_recorded
        self.param_recorded = True
        self.param_dict = param_dict

    def record_result(self, result_dict):
        """
        all results must be given at the same time
        :return:
        """
        assert self.param_recorded
        assert not self.result_recorded
        self.result_recorded = True

        if os.path.exists(self.file_name):
            with open(self.file_name, 'r') as csv_file:
                reader = csv.reader(csv_file)
                fields = next(reader)
        else:
            print("This is the first record of the experiment")
            fields = list(self.param_dict.keys()) + list(result_dict.keys())
            with open(self.file_name, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(fields)

        self.param_dict.update(result_dict)

        values = []
        for field in fields:
            if field in self.param_dict:
                values.append(self.param_dict[field])
            else:
                values.append("NOT PRESENT")

        extra_fields = list(set(self.param_dict.keys() - set(fields)))
        if not len(extra_fields) == 0:
            for field in extra_fields:
                values.append(f"{field:} {self.param_dict[field]}")
                print(f"adding extra field {field}")

        with open(self.file_name, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(values)


# source: https://stackoverflow.com/questions/2363731/append-new-row-to-old-csv-file-python
def flatten_dict(d, parent_key='', sep='_', use_short_name=True):
    items = []
    for k, v in d.items():
        if use_short_name:
            k, v = short_name(k), short_name(v)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


SHORT_NAME = {
    'DATALOADER': 'DL',
    'batch_size': 'bs',
    'datapath': 'dp',
    'load_image': 'lm',
    'crop': 'cr',
    'norm_data': 'nd',
    'data_aug_shift': 'das',
    'data_aug_color': 'dac',
    'resize_mask': 'rm',
    'TRAIN' : 'TR',
    'num_epochs': 'ne',
    'learning_rate': 'lr',
    'MODEL': 'M',
    'TWO_D': '2D',
    'feature_dim': 'fd',
    'LANGUAGE': 'LG',
    'DRNET': 'DR',
    'pretrained': 'pr',
    'dropout': 'dr',
    'num_layers': 'nl',
    'backbone': 'bb',
    'two_stream': '2s',
    'only_2d': 'o2d',
    'only_appr': 'oa',
    'VTRANSE': 'VT',
    'visual_feature_size': 'vfs',
    'predicate_embedding_dim': 'ped',
    'feat_size': 'fs',
    'feat_dim': 'fd',
    'roi_size': 'roi',
    'with_rgb': 'rgb',
    'with_depth': 'depth',
    'with_bbox': 'bbox',
    'add_union_feat': 'auf',
    '20200207_c_0.9_c_0.1_c_1.0.json': '20200207_def',
    '20200215_c_0.9_c_0.1_c_1.0.json': '20200215_def',
    '20200220_c_0.9_c_0.1_c_1.0.json': '20200220_def',
    'True' : 'T',
    'False': 'F',
    'trans_vec': 'tv',
    'raw_absolute': 'ra',
    'aligned_absolute': 'aa',
    'aligned_relative': 'ar',
    "with_class": 'wc',
    True: 'T',
    False: 'F',
    'remove_near_far': 'NO_N_F'
}


def short_name(x):
    if isinstance(x, Hashable):
        if x in SHORT_NAME:
            return SHORT_NAME[x]
        else:
            return x
    else:
        return x
