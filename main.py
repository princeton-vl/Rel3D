import json
import os
import random
import argparse
from time import time
from datetime import datetime
from collections import defaultdict
from progressbar import ProgressBar

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from dataloader import create_dataloader
from utils import TensorboardManager, RecordExp, flatten_dict
from configs import get_cfg_defaults


DEVICE = ""


def logit_to_prob(logit):
    return torch.exp(logit) / (1 + torch.exp(logit))


def get_inp(model, data_batch, device):
    predi = data_batch['predicate']['idx'].to(device, non_blocking=True)
    subj_idx = data_batch['subject']['idx'].to(device, non_blocking=True)
    obj_idx = data_batch['object']['idx'].to(device, non_blocking=True)
    subj_bbox = data_batch['subject']['bbox'].to(device, non_blocking=True)
    obj_bbox = data_batch['object']['bbox'].to(device, non_blocking=True)
    so_bbox = data_batch['predicate']['bbox'].to(device, non_blocking=True)
    subj_t = data_batch['subject']['t'].to(device, non_blocking=True)
    obj_t = data_batch['object']['t'].to(device, non_blocking=True)
    if 'img_crop' in data_batch:
        rgb = data_batch['img_crop'].to(device, non_blocking=True)
        depth = data_batch['depth_crop'].to(device, non_blocking=True)
        bbox_mask = data_batch['bbox_mask'].to(device, non_blocking=True)
    pred_names = data_batch['predicate']['name']

    if isinstance(model, models.SimpleLanguageModel):
        inp = {
            'subj': subj_idx,
            'obj': obj_idx,
            'predi': predi
        }
    elif isinstance(model, models.SimpleSpatialModel):
        inp = {
            'subj': subj_bbox,
            'obj': obj_bbox,
            'predi': predi
        }
    elif isinstance(model, models.DRNet):
        inp = {
            "subj": subj_idx,
            "obj": obj_idx,
            "img_crop": rgb,
            "bbox_mask": bbox_mask,
            "predi": predi,
            "depth_crop": depth
        }
    elif isinstance(model, models.VTransE):
        inp = {
            "subj": subj_idx,
            "obj": obj_idx,
            "full_im": rgb,
            "t_s": subj_t,
            "t_o": obj_t,
            "bbox_s": subj_bbox,
            "bbox_o": obj_bbox,
            "predi": predi
        }
    elif isinstance(model, models.SimpleTransModel):
        feat_dict = {x: data_batch[x].to(device, non_blocking=True).float()
                     for x in model.trans_vec}
        feat_dict['subj'] = subj_idx
        feat_dict['obj'] = obj_idx
        inp = {
            "feat_dict": feat_dict,
            "predi": predi
        }
    elif isinstance(model, models.VipCNN):
        inp = {
            "img": rgb,
            "bbox_s": subj_bbox,
            "bbox_o": obj_bbox,
            "predicate": predi
        }
    elif isinstance(model, models.PPRFCN):
        inp = {
            "img": rgb,
            "bbox_s": subj_bbox,
            "bbox_o": obj_bbox,
            "predicate": predi
        }
    else:
        assert False

    return inp


def validate(loader, model, device):
    """
    :param loader:
    :param model:
    :param device:
    :return:
    """
    model.eval()
    correct = []
    tp = []
    fp = []
    p = []
    # dictionary storing correct list relation wise
    correct_rel = defaultdict(list)

    with torch.no_grad():
        _bar = ProgressBar(max_value=len(loader))
        for i, data_batch in enumerate(loader):
            label = data_batch['label'].to(device, non_blocking=True)
            inp = get_inp(model, data_batch, DEVICE)

            logit = model(**inp)
            batch_correct = (((logit > 0) & (label == True))
                             | ((logit <= 0) & (label == False))).tolist()
            tp.extend(((logit > 0) & (label == True)).tolist())
            fp.extend(((logit > 0) & (label == False)).tolist())

            correct.extend(batch_correct)
            p.extend((label == True).tolist())
            for pred_name, _correct in zip(data_batch['predicate']['name'],
                                           batch_correct):
                correct_rel[pred_name].append(_correct)
            _bar.update(i)

    acc = sum(correct) / len(correct)
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001)
    rec = sum(tp) / sum(p)
    f1 = (2 * pre * rec) / (pre + rec + 0.00001)
    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())

    return acc, pre, rec, f1, acc_rel, acc_rel_avg


def train(loader, model, optimizer, device, weighted_loss=False):
    model.train()
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    avg_loss = []
    correct = []
    tp = []
    fp = []
    p = []
    # dictionary storing correct list relation wise
    correct_rel = defaultdict(list)

    time_last_batch_end = time()
    for i, data_batch in enumerate(loader):
        time_start = time()
        label = data_batch['label'].to(device, non_blocking=True)
        inp = get_inp(model, data_batch, DEVICE)
        logit = model(**inp)

        if weighted_loss:
            weight = data_batch['weight'].to(device, non_blocking=True).to(
                dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logit, label.to(
                dtype=torch.float32), weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logit, label.to(
                dtype=torch.float32))
        time_forward += (time() - time_start)

        avg_loss.append(loss.item())
        batch_correct = (((logit > 0) & (label == True))
                         | ((logit <= 0) & (label == False))).tolist()
        correct.extend(batch_correct)
        tp.extend(((logit > 0) & (label == True)).tolist())
        p.extend((label == True).tolist())
        fp.extend(((logit > 0) & (label == False)).tolist())
        for pred_name, _correct in zip(data_batch['predicate']['name'],
                                       batch_correct):
            correct_rel[pred_name].append(_correct)

        optimizer.zero_grad()
        time_start = time()
        loss.backward()
        time_backward += (time() - time_start)
        optimizer.step()
        time_data_loading += (time_start - time_last_batch_end)
        time_last_batch_end = time()

        if i % 50 == 0:
            print(
                '[%d/%d] Loss = %.02f, Forward time = %.02f, Backward time = %.02f, Data loading time = %.02f' \
                % (i, len(loader), np.mean(avg_loss), time_forward,
                   time_backward, time_data_loading))

            avg_loss = []

    acc = sum(correct) / len(correct)
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001)
    rec = sum(tp) / sum(p)
    f1 = (2 * pre * rec) / (pre + rec + 0.00001)
    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())

    return acc, pre, rec, f1, acc_rel, acc_rel_avg


def save_checkpoint(epoch, model, optimizer, acc, cfg):
    model.cpu()
    path = f"./runs/{cfg.EXP.EXP_ID}/model_best.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'accuracy': acc,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_best_checkpoint(model, cfg):
    path = f"./runs/{cfg.EXP.EXP_ID}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)
    model.to(DEVICE)


def load_checkpoint(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % model_path)
    model.to(DEVICE)


def get_model(cfg):
    if cfg.EXP.MODEL_NAME == 'language':
        model = models.SimpleLanguageModel(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            object_dim=cfg.DATALOADER.object_dim,
            **cfg.MODEL.LANGUAGE
        )
    elif cfg.EXP.MODEL_NAME == '2d':
        model = models.SimpleSpatialModel(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.TWO_D
        )
    elif cfg.EXP.MODEL_NAME == 'drnet':
        model = models.DRNet(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            object_dim=cfg.DATALOADER.object_dim,
            **cfg.MODEL.DRNET
        )
    elif cfg.EXP.MODEL_NAME == 'vtranse':
        assert not cfg.DATALOADER.crop
        model = models.VTransE(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            object_dim=cfg.DATALOADER.object_dim,
            **cfg.MODEL.VTRANSE
        )
    elif cfg.EXP.MODEL_NAME == 'trans':
        model = models.SimpleTransModel(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            object_dim=cfg.DATALOADER.object_dim,
            trans_vec=cfg.DATALOADER.trans_vec,
            **cfg.MODEL.TRANS
        )
    elif cfg.EXP.MODEL_NAME == 'vipcnn':
        model = models.VipCNN(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.VIPCNN
        )
    elif cfg.EXP.MODEL_NAME == 'pprfcn':
        model = models.PPRFCN(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.PPRFCN
        )
    else:
        raise ValueError

    return model


def entry_train(cfg, record_file=""):
    loader_train, _, _ = create_dataloader(split='train', **cfg.DATALOADER)
    loader_valid, _, _ = create_dataloader('valid', **cfg.DATALOADER)
    loader_test, _, _ = create_dataloader('test', **cfg.DATALOADER)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.TRAIN.learning_rate,
                                 weight_decay=cfg.TRAIN.l2)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='max',
                                  factor=0.5,
                                  patience=10,
                                  verbose=True)

    best_acc_rel_avg_valid = -1
    best_epoch_rel_avg_valid = 0
    best_acc_rel_avg_test = -1

    log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = TensorboardManager(log_dir)
    for epoch in range(cfg.TRAIN.num_epochs):
        print('\nEpoch #%d' % epoch)

        print('Training..')
        (acc_train, pre_train, rec_train, f1_train, acc_rel_train,
         acc_rel_avg_train) = train(loader_train, model, optimizer,
                                    DEVICE, cfg.TRAIN.weighted_loss)
        print(f'Train, acc avg: {acc_rel_avg_train} acc: {acc_train},'
              f' pre: {pre_train}, rec: {rec_train}, f1: {f1_train}')
        print({x: round(y, 3) for x, y in acc_rel_train.items()})
        tb.update('train', epoch, {'acc': acc_train})

        print('\nValidating..')
        (acc_valid, pre_valid, rec_valid, f1_valid, acc_rel_valid,
         acc_rel_avg_valid) = validate(loader_valid, model, DEVICE)
        print(f'Valid, acc avg: {acc_rel_avg_valid} acc: {acc_valid},'
              f' pre: {pre_valid}, rec: {rec_valid}, f1: {f1_valid}')
        print({x: round(y, 3) for x, y in acc_rel_valid.items()})
        tb.update('val', epoch, {'acc': acc_valid})

        print('\nTesting..')
        (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
         acc_rel_avg_test) = validate(loader_test, model, DEVICE)
        print(f'Test, acc avg: {acc_rel_avg_test} acc: {acc_test},'
              f' pre: {pre_test}, rec: {rec_test}, f1: {f1_test}')
        print({x: round(y, 3) for x, y in acc_rel_test.items()})

        if acc_rel_avg_valid > best_acc_rel_avg_valid:
            print('Accuracy has improved')
            best_acc_rel_avg_valid = acc_rel_avg_valid
            best_epoch_rel_avg_valid = epoch

            save_checkpoint(epoch, model, optimizer, acc_rel_avg_valid, cfg)
        if acc_rel_avg_test > best_acc_rel_avg_test:
            best_acc_rel_avg_test = acc_rel_avg_test

        if (epoch - best_epoch_rel_avg_valid) > cfg.TRAIN.early_stop:
            print(f"Early stopping at {epoch} as val acc did not improve"
                  f" for {cfg.TRAIN.early_stop} epochs.")
            break

        scheduler.step(acc_train)

    print('\nTesting..')
    load_best_checkpoint(model, cfg)
    (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
     acc_rel_avg_test) = validate(loader_test, model, DEVICE)
    print(f'Best valid, acc: {best_acc_rel_avg_valid}')
    print(f'Best test, acc: {best_acc_rel_avg_test}')
    print(f'Test at best valid, acc avg: {acc_rel_avg_test}, acc: {acc_test},'
          f' pre: {pre_test}, rec: {rec_test}, f1: {f1_test}')
    print({x: round(y, 3) for x, y in acc_rel_test.items()})

    if record_file != "":
        exp = RecordExp(record_file)
        exp.record_param(flatten_dict(dict(cfg)))
        exp.record_result({
            "final_train": acc_rel_avg_train,
            "best_val": best_acc_rel_avg_valid,
            "best_test": best_acc_rel_avg_test,
            "final_test": acc_rel_avg_test
        })


def entry_test(cfg, model_path):
    loader_test, _, _ = create_dataloader('test', **cfg.DATALOADER)

    model = get_model(cfg)
    model.to(DEVICE)
    load_checkpoint(model, model_path)

    print('\nTesting..')
    (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
     acc_rel_avg_test) = validate(loader_test, model, DEVICE)
    print(f'Test at best valid, acc avg: {acc_rel_avg_test}, acc: {acc_test}, '
          f'pre: {pre_test}, rec: {rec_test}, f1: {f1_test}')
    print({x: round(y, 3) for x, y in acc_rel_test.items()})


def get_incorr_samp(model, loader):
    """
    Get the incorrect samples for each model
    :param model:
    :param loader:
    :return:
        incorr_samp:
            {
                'aligned to': [
                    {
                        "rgb_source": "path.png",
                        "label": True,
                        "logit": 1.0001,
                        "subj": foo,
                        "obj": bar
                    },
                    .......
                ]
                .......
            }
    """

    model.eval()
    correct_rel = defaultdict(list)
    incorr_samp = defaultdict(list)

    predictions = []
    with torch.no_grad():
        _bar = ProgressBar(max_value=len(loader))
        for i, data_batch in enumerate(loader):
            label = data_batch['label'].to(DEVICE, non_blocking=True)
            pred_name = data_batch['predicate']['name']
            subj_name = data_batch['subject']['name']
            obj_name = data_batch['object']['name']
            rgb_source = data_batch['rgb_source']
            inp = get_inp(model, data_batch, DEVICE)

            logit = model(**inp)
            correct = (((logit > 0) & (label == True))
                       | ((logit <= 0) & (label == False))).tolist()
            predictions.append(correct)
            for _pred_name, _subj_name, _obj_name, _label, _correct, \
                    _rgb_source, _logit in zip(
                        pred_name, subj_name, obj_name, label, correct,
                        rgb_source, logit):
                correct_rel[_pred_name].append(_correct)
                if not _correct:
                    incorr_samp[_pred_name].append(
                        {
                            "rgb_source": _rgb_source,
                            "label": _label.item(),
                            "logit": _logit.item(),
                            "subj": _subj_name,
                            "obj": _obj_name
                        }
                    )
            _bar.update(i)

    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())
    print(f" Average Relation Acc: {acc_rel_avg}")

    return incorr_samp, predictions


def generate_failure_log(cfg):
    loader_valid, _, _ = create_dataloader('valid', **cfg.DATALOADER)
    loader_test, _, _ = create_dataloader('test', **cfg.DATALOADER)

    model = get_model(cfg)
    model.to(DEVICE)
    load_best_checkpoint(model, cfg)
    print(model)

    incorr_samp_val, pred_val = get_incorr_samp(model, loader_valid)
    incorr_samp_test, pred_test = get_incorr_samp(model, loader_test)

    failure_log = {
        "valid": incorr_samp_val,
        "test": incorr_samp_test,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }

    path = f"./runs/{cfg.EXP.EXP_ID}/failure_log.json"
    with open(path, 'w') as file:
        print(f"Saving the failure log in {path}")
        json.dump(failure_log, file)


if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument('--exp-config', type=str, default="")
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--record-file', type=str, default="")

    cmd_args = parser.parse_args()

    if cmd_args.entry == "train":
        assert not cmd_args.exp_config == ""

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        if _cfg.EXP.EXP_ID == "":
            _cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
        _cfg.freeze()
        print(_cfg)

        torch.manual_seed(_cfg.EXP.SEED)
        np.random.seed(_cfg.EXP.SEED)
        random.seed(_cfg.EXP.SEED)

        entry_train(_cfg, cmd_args.record_file)

    elif cmd_args.entry == "test":
        assert not cmd_args.exp_config == ""

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        _cfg.freeze()
        print(_cfg)

        torch.manual_seed(_cfg.EXP.SEED)
        np.random.seed(_cfg.EXP.SEED)
        random.seed(_cfg.EXP.SEED)
        entry_test(_cfg, cmd_args.model_path)

    elif cmd_args.entry == "failure_log":
        assert not cmd_args.exp_config == ""

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        _cfg.freeze()
        print(_cfg)

        torch.manual_seed(_cfg.EXP.SEED)
        np.random.seed(_cfg.EXP.SEED)
        random.seed(_cfg.EXP.SEED)
        generate_failure_log(_cfg)

    else:
        assert False
