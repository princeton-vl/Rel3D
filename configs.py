from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.MODEL_NAME = '2d'
_C.EXP.EXP_ID = ""
_C.EXP.SEED = 0
# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.batch_size = 128
_C.DATALOADER.num_workers = 4
_C.DATALOADER.datapath = "data/real.json"
_C.DATALOADER.load_img = False
_C.DATALOADER.crop = False
_C.DATALOADER.norm_data = False
_C.DATALOADER.data_aug_shift = False
_C.DATALOADER.data_aug_color = False
_C.DATALOADER.resize_mask = False
_C.DATALOADER.trans_vec = []
_C.DATALOADER.predicate_dim = 30
_C.DATALOADER.object_dim = 67
# -----------------------------------------------------------------------------
# TRAINING DETAILS
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.num_epochs = 200
_C.TRAIN.learning_rate = 1e-3
_C.TRAIN.l2 = 0.0
_C.TRAIN.weighted_loss = True
_C.TRAIN.early_stop = 20
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# -----------------------------------------------------------------------------
# 2D MODEL
# -----------------------------------------------------------------------------
_C.MODEL.TWO_D = CN()
_C.MODEL.TWO_D.feature_dim = 256
# -----------------------------------------------------------------------------
# LANGUAGE MODEL
# -----------------------------------------------------------------------------
_C.MODEL.LANGUAGE = CN()
_C.MODEL.LANGUAGE.feature_dim = 256
# -----------------------------------------------------------------------------
# DRNET
# -----------------------------------------------------------------------------
_C.MODEL.DRNET = CN()
_C.MODEL.DRNET.feature_dim = 256
_C.MODEL.DRNET.pretrained = False
_C.MODEL.DRNET.dropout = False
_C.MODEL.DRNET.num_layers = 3
_C.MODEL.DRNET.backbone = 'resnet18'
_C.MODEL.DRNET.two_stream = False
_C.MODEL.DRNET.rgbd = False
_C.MODEL.DRNET.only_2d = False
_C.MODEL.DRNET.only_appr = False
# -----------------------------------------------------------------------------
# VTRANSE
# -----------------------------------------------------------------------------
_C.MODEL.VTRANSE = CN()
_C.MODEL.VTRANSE.feature_dim = 256
_C.MODEL.VTRANSE.visual_feature_size = 3
# same as feature dim in the Supplementary material, Table 4
_C.MODEL.VTRANSE.predicate_embedding_dim = 256
_C.MODEL.VTRANSE.pretrained = False
_C.MODEL.VTRANSE.backbone = 'resnet18'
# -----------------------------------------------------------------------------
# AE
# -----------------------------------------------------------------------------
_C.MODEL.AE = CN()
_C.MODEL.AE.feat_dim = 128
_C.MODEL.AE.roi_size = 4
_C.MODEL.AE.with_rgb = False
_C.MODEL.AE.with_depth = False
_C.MODEL.AE.with_bbox = False
_C.MODEL.AE.add_union_feat = False
_C.MODEL.AE.bn = False
# -----------------------------------------------------------------------------
# TRANS
# -----------------------------------------------------------------------------
_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.feat_dim = 128
_C.MODEL.TRANS.with_class = False
# -----------------------------------------------------------------------------
# VipCNN
# -----------------------------------------------------------------------------
_C.MODEL.VIPCNN = CN()
_C.MODEL.VIPCNN.roi_size = 3
_C.MODEL.VIPCNN.backbone = 'resnet18'
# -----------------------------------------------------------------------------
# PPRFCN
# -----------------------------------------------------------------------------
_C.MODEL.PPRFCN = CN()
_C.MODEL.PPRFCN.backbone = 'resnet18'
# there is no ROI Size parameter


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
