#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from tqdm import tqdm

# TF enumerates and initialises GPUs at first import; if CUDA_VISIBLE_DEVICES
# is not set beforehand, TF1 sessions crash with "Error loading CUDA libraries"
# even when allow_soft_placement=True (that flag covers op placement, not device
# initialisation failures).  Fallback: set CUDA_VISIBLE_DEVICES="" so TF skips
# GPU init entirely and runs on CPU.
def _select_gpu_early():
    existing = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if existing is not None:
        print(f"Using pre-set CUDA_VISIBLE_DEVICES={existing}")
        return  # already set externally — respect it
    gpu_id = None
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu_id' and i + 1 < len(sys.argv):
            try:
                gpu_id = int(sys.argv[i + 1])
            except ValueError:
                pass
            break
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Pinned GPU {gpu_id} (via --gpu_id)")
        return
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip().split('\n')
        free = [int(x) for x in out if x.strip()]
        if free:
            best = free.index(max(free))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(best)
            print(f"Auto-selected GPU {best} ({free[best]} MiB free)")
            return
    except Exception:
        pass
    # No GPU found or CUDA libs broken — force CPU-only to avoid session crash.
    # "-1" is the canonical TF/CUDA value for "no GPU"; "" (empty) still triggers
    # CUDA lib loading which crashes TF1 Session creation on this machine.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("No usable GPU detected via nvidia-smi; forcing CPU mode (CUDA_VISIBLE_DEVICES=-1)")

_select_gpu_early()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf_keras >= 2.16 removed legacy_tf_layers, but tf_slim's batch_norm lazily
# loads tensorflow.python.layers.normalization which re-exports from:
#   tf_keras.legacy_tf_layers.normalization.BatchNormalization
# tf_slim passes TF1-specific kwargs (_scope, _reuse, renorm, adjustment, fused)
# that modern keras BN doesn't accept.  We inject a stub package with a thin
# compatibility wrapper that strips those kwargs before calling keras BN.
import types as _types, tensorflow as _tf_real

class _BNCompat:
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3,
                 center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None,
                 trainable=True, name=None, **kwargs):
        # all unrecognised kwargs (TF1-only: _scope, _reuse, renorm, …) are silently dropped
        self.axis      = axis
        self.momentum  = momentum   # same convention as keras: EMA = momentum*old + (1-momentum)*new
        self.epsilon   = epsilon
        self.center    = center
        self.scale     = scale
        self.trainable = trainable
        # TF1 names use '/' as scope separator — sanitise for variable_scope
        self.name = (name.replace('/', '_').replace(':', '_') if name else 'bn')
        self.beta = self.gamma = self.moving_mean = self.moving_variance = None

    def apply(self, inputs, training=False):
        n_ch = inputs.shape[-1]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.moving_mean = tf.get_variable(
                'moving_mean', [n_ch], dtype=tf.float32,
                initializer=tf.zeros_initializer(), trainable=False)
            self.moving_variance = tf.get_variable(
                'moving_variance', [n_ch], dtype=tf.float32,
                initializer=tf.ones_initializer(), trainable=False)
            self.gamma = tf.get_variable(
                'gamma', [n_ch], dtype=tf.float32,
                initializer=tf.ones_initializer(), trainable=self.trainable) if self.scale else None
            self.beta = tf.get_variable(
                'beta', [n_ch], dtype=tf.float32,
                initializer=tf.zeros_initializer(), trainable=self.trainable) if self.center else None

            reduce_axes = list(range(len(inputs.shape) - 1))   # [0,1,2] for NHWC

            def _train():
                mean, var = tf.nn.moments(inputs, reduce_axes)
                decay = 1.0 - self.momentum
                upd_m = tf.assign_sub(self.moving_mean,     decay * (self.moving_mean     - mean))
                upd_v = tf.assign_sub(self.moving_variance, decay * (self.moving_variance - var))
                with tf.control_dependencies([upd_m, upd_v]):
                    return tf.nn.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

            def _infer():
                return tf.nn.batch_normalization(
                    inputs, self.moving_mean, self.moving_variance,
                    self.beta, self.gamma, self.epsilon)

            if isinstance(training, bool):
                return _train() if training else _infer()
            return tf.cond(training, _train, _infer)

if 'tf_keras.legacy_tf_layers.normalization' not in sys.modules:
    _legacy_pkg = _types.ModuleType('tf_keras.legacy_tf_layers')
    _legacy_pkg.__path__ = []
    _norm_mod = _types.ModuleType('tf_keras.legacy_tf_layers.normalization')
    _norm_mod.BatchNormalization = _BNCompat
    _legacy_pkg.normalization = _norm_mod
    sys.modules['tf_keras.legacy_tf_layers'] = _legacy_pkg
    sys.modules['tf_keras.legacy_tf_layers.normalization'] = _norm_mod
    try:
        import tf_keras as _tfk
        _tfk.legacy_tf_layers = _legacy_pkg
    except Exception:
        pass

try:
    import tf_slim as slim
    from tf_slim.nets import resnet_v2
except ImportError:
    raise ImportError(
        "HP-FCN requires tf-slim (the standalone fork of tf.contrib.slim).\n"
        "Install with: pip install tf-slim"
    )

WORK_DIR   = Path(__file__).resolve().parent.parent
HPFCN_DIR  = WORK_DIR / 'baselines' / 'git_repo' / 'Deep_inpainting_localization'
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
sys.path.insert(0, str(HPFCN_DIR))
from config import DATA_DIR

from hexmil.data.patch_dataset import load_split_table, MOD_LABEL
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

# Load individual files directly to avoid utils/__init__.py which imports
# vgg_mfcn.py → tensorflow.contrib (TF1-only, removed in TF2).
import importlib.util as _ilu

def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_bu      = _load_module('bilinear_upsample_weights',
                         HPFCN_DIR / 'utils' / 'bilinear_upsample_weights.py')
_rm      = _load_module('repo_metrics',
                         HPFCN_DIR / 'utils' / 'metrics.py')
bilinear_upsample_weights = _bu.bilinear_upsample_weights
repo_metrics              = _rm

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']

#  HP-FCN Model — architecture from official repo (hp_fcn.py)
#  Uses tf_slim as drop-in for tf.contrib.slim

# Filter kernels — verbatim from hp_fcn.py (Li & Huang, ICCV 2019)
FILTERS = {
    'd1': [
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 0., 1.]])],
    'd2': [
        np.array([[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]),
        np.array([[1., 0., 0.], [0., -2., 0.], [0., 0., 1.]])],
    'd3': [
        np.array([[0., 0., 0., 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., -3., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -1., 3., -3., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., -1., 0., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., 0., -3., 0.], [0., 0., 0., 0., 1.]])],
    'd4': [
        np.array([[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[1., 0., 0., 0., 0.], [0., -4., 0., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., 0., -4., 0.], [0., 0., 0., 0., 1.]])],
}


def get_residuals(image, filter_type='d1', filter_trainable=True, image_channel=1):
    residuals = []
    kernel_index = 0
    for filter_kernel in FILTERS[filter_type]:
        kernel_variable = tf.Variable(
            np.repeat(filter_kernel[:, :, np.newaxis, np.newaxis], image_channel, axis=2),
            trainable=filter_trainable, dtype='float',
            name='root_filter{}'.format(kernel_index))
        image_filtered = tf.nn.depthwise_conv2d(
            image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
        residuals.append(image_filtered)
        kernel_index += 1
    return tf.concat(residuals, 3)


def resnet_small(inputs, num_classes=None, is_training=True, global_pool=True,
                 output_stride=None, include_root_block=True, reuse=None,
                 scope='resnet_small'):
    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=32, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=128, num_units=2, stride=2),
        resnet_v2.resnet_v2_block('block4', base_depth=256, num_units=2, stride=2),
    ]
    return resnet_v2.resnet_v2(
        inputs, blocks, num_classes, is_training=is_training,
        global_pool=global_pool, output_stride=output_stride,
        include_root_block=include_root_block, reuse=reuse, scope=scope)


def build_model(images, filter_type, filter_trainable, weight_decay,
                batch_size, is_training, num_classes=2):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        inputs = get_residuals(images, filter_type, filter_trainable, image_channel=1)
        _, end_points = resnet_small(
            inputs, num_classes=None, is_training=is_training,
            global_pool=False, output_stride=None, include_root_block=False)

        net = end_points['resnet_small/block4']
        # Upsample ×4: block4 → block2 resolution
        net = tf.nn.conv2d_transpose(
            net,
            tf.Variable(bilinear_upsample_weights(4, 64, 1024),
                        dtype=tf.float32, name='bilinear_kernel0'),
            [batch_size,
             tf.shape(end_points['resnet_small/block2'])[1],
             tf.shape(end_points['resnet_small/block2'])[2], 64],
            strides=[1, 4, 4, 1], padding="SAME")
        # Upsample ×4: → input resolution
        net = tf.nn.conv2d_transpose(
            net,
            tf.Variable(bilinear_upsample_weights(4, 4, 64),
                        dtype=tf.float32, name='bilinear_kernel1'),
            [batch_size,
             tf.shape(inputs)[1],
             tf.shape(inputs)[2], 4],
            strides=[1, 4, 4, 1], padding="SAME")
        # Post-processing: BN + ReLU + 5×5 conv
        net = _BNCompat(name='post_upsample_bn').apply(net, training=is_training)
        net = tf.nn.relu(net)
        logits = slim.conv2d(net, num_classes, [5, 5],
                             activation_fn=None, normalizer_fn=None, scope='logits')
        preds = tf.cast(tf.argmax(logits, 3), tf.int32)
        preds_map = tf.nn.softmax(logits)[:, :, :, 1]
        return logits, preds, preds_map

#  Loss — focal loss from official repo (utils/losses.py)

def focal_loss_tf(logits, labels, gamma=2.0, num_classes=2):
    logits_flat = tf.reshape(logits, (-1, num_classes))
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
    weights = tf.reduce_sum(
        tf.multiply(one_hot, tf.pow(tf.subtract(1.0, tf.nn.softmax(logits_flat)), gamma)), 1)
    ce = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.stop_gradient(one_hot), logits=logits_flat)
    return tf.reduce_mean(tf.multiply(weights, ce))

#  Data loading — M3DSynth CT slices via numpy, fed to TF via feed_dict

def load_slices_numpy(data_dir, tab, target_size=224, desc='loading'):
    from skimage.transform import resize as sk_resize
    images, masks, labels, mods, img_ids = [], [], [], [], []
    for _, row in tqdm(tab.iterrows(), total=len(tab), desc=desc, ncols=80):
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir  = os.path.join(data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
        img = apply_percentile(raw.astype(np.float32), low, high)  # [0,1]

        if mod == 'real':
            mask = np.zeros_like(img, dtype=np.float32)
        else:
            ldir = os.path.join(data_dir, mod, 'label', img_id)
            mask = load_slice_tiff_scan(ldir, shape, np.bool_, cz, cz + 1)[0].astype(np.float32)

        if img.shape[0] != target_size or img.shape[1] != target_size:
            img  = sk_resize(img,  (target_size, target_size), preserve_range=True).astype(np.float32)
            mask = sk_resize(mask, (target_size, target_size), order=0, preserve_range=True).astype(np.float32)

        images.append(img[:, :, np.newaxis])   # (H, W, 1)
        masks.append(mask.astype(np.int32))    # (H, W)
        labels.append(0 if mod == 'real' else 1)
        mods.append(mod)
        img_ids.append(img_id)

    return (np.array(images), np.array(masks), np.array(labels),
            mods, img_ids)


def make_batches(images, masks, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        yield images[idx], masks[idx]

#  Evaluation — same format as other baselines

def _compute_metrics(labels, scores, mods_list) -> dict:
    y = np.array(labels)
    s = np.nan_to_num(np.array(scores, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    p = (s >= 0.5).astype(int)
    def _auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')
    out = dict(auc=_auc(y, s), acc=float(accuracy_score(y, p)),
               f1=float(f1_score(y, p, zero_division=0)),
               ap=float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float('nan'))
    per_mod = {}
    rm = np.array(mods_list) == 'real'
    for fm in ALL_FAKES:
        sel = rm | (np.array(mods_list) == fm)
        if sel.sum() < 2: continue
        per_mod[fm] = dict(auc=_auc(y[sel], s[sel]), acc=float(accuracy_score(y[sel], p[sel])),
                           f1=float(f1_score(y[sel], p[sel], zero_division=0)),
                           ap=float(average_precision_score(y[sel], s[sel]))
                              if len(np.unique(y[sel])) > 1 else float('nan'))
    out['per_mod'] = per_mod
    return out


def evaluate_localization_np(pred_maps, gt_masks, labels, mods_list) -> dict:
    _T = (0.3, 0.5, 0.7)
    pg_all, eom_all, pauc_all = [], [], []
    iou_all = {t: [] for t in _T}
    per_mod = {fm: {'pg': [], 'eom': [], 'pauc': [], **{f'iou_{t}': [] for t in _T}}
               for fm in ALL_FAKES}

    for i, lbl in enumerate(labels):
        if lbl == 0:
            continue
        pm = pred_maps[i]
        mask = gt_masks[i].astype(np.float32)
        if mask.sum() == 0:
            continue
        mod = mods_list[i]
        ay, ax = np.unravel_index(pm.argmax(), pm.shape)
        pg = int(mask[ay, ax] > 0.5)
        eom = float((pm * mask).sum() / (pm.sum() + 1e-8))
        y_flat = mask.flatten().astype(int)
        pauc = float(roc_auc_score(y_flat, pm.flatten())) \
            if len(np.unique(y_flat)) > 1 else float('nan')
        mask_bin = mask > 0.5
        ious = {}
        for t in _T:
            pred_bin = pm >= t
            inter = float((pred_bin & mask_bin).sum())
            union = float((pred_bin | mask_bin).sum())
            ious[t] = inter / (union + 1e-8)

        pg_all.append(pg); eom_all.append(eom)
        if not np.isnan(pauc): pauc_all.append(pauc)
        for t in _T: iou_all[t].append(ious[t])
        if mod in per_mod:
            per_mod[mod]['pg'].append(pg)
            per_mod[mod]['eom'].append(eom)
            if not np.isnan(pauc): per_mod[mod]['pauc'].append(pauc)
            for t in _T: per_mod[mod][f'iou_{t}'].append(ious[t])

    def _m(v): return float(np.mean(v)) if v else float('nan')
    return {
        'pointing_game': _m(pg_all), 'energy_on_mask': _m(eom_all),
        'pixel_auc': _m(pauc_all),
        'iou_0.3': _m(iou_all[0.3]), 'iou_0.5': _m(iou_all[0.5]), 'iou_0.7': _m(iou_all[0.7]),
        'n_fake_slices': len(pg_all),
        'per_mod': {mod: {
            'pointing_game': _m(v['pg']), 'energy_on_mask': _m(v['eom']),
            'pixel_auc': _m(v['pauc']),
            'iou_0.3': _m(v['iou_0.3']), 'iou_0.5': _m(v['iou_0.5']), 'iou_0.7': _m(v['iou_0.7']),
        } for mod, v in per_mod.items() if v['pg']},
    }


def get_args():
    p = argparse.ArgumentParser(description='HP-FCN (Li & Huang, ICCV 2019) — TF, 1-ch CT')
    p.add_argument('--data_dir',     type=str,   default=DATA_DIR)
    p.add_argument('--out_dir',      type=str,   default=None)
    p.add_argument('--K',            type=int,   default=16)
    p.add_argument('--target_size',  type=int,   default=224)
    p.add_argument('--filter_type',  type=str,   default='d1',
                   choices=['d1', 'd2', 'd3', 'd4'])
    p.add_argument('--filter_learnable', action='store_true', default=True)
    p.add_argument('--focal_gamma',  type=float, default=2.0)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--lr_decay',     type=float, default=0.5)
    p.add_argument('--lr_decay_freq',type=float, default=1.0)
    p.add_argument('--patience',     type=int,   default=12)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--gpu_id',       type=int,   default=None)
    p.add_argument('--train_mods',   nargs='+',  default=None)
    p.add_argument('--eval_only',    action='store_true', default=False)
    return p.parse_args()


def select_gpu(gpu_id=None):
    pass  # GPU already selected by _select_gpu_early() before TF import


def main():
    args = get_args()
    np.random.seed(args.seed)
    select_gpu(args.gpu_id)

    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))

    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'hp_fcn_K{args.K}' / f'trained_on_{mods_tag}'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    tr_mods = ['real'] + train_mods
    ts_mods = ['real'] + ALL_FAKES

    print("Loading data...")
    tab_train = load_split_table(args.data_dir, 'train', tr_mods)
    tab_valid = load_split_table(args.data_dir, 'valid', tr_mods)
    tab_test  = load_split_table(args.data_dir, 'test',  ts_mods)

    tr_imgs, tr_masks, tr_labels, tr_mods_list, _ = load_slices_numpy(
        args.data_dir, tab_train, args.target_size, desc='train')
    vl_imgs, vl_masks, vl_labels, vl_mods_list, _ = load_slices_numpy(
        args.data_dir, tab_valid, args.target_size, desc='valid')
    ts_imgs, ts_masks, ts_labels, ts_mods_list, _ = load_slices_numpy(
        args.data_dir, tab_test,  args.target_size, desc='test')
    print(f"Train: {len(tr_imgs)}, Valid: {len(vl_imgs)}, Test: {len(ts_imgs)}")

    tf.reset_default_graph()
    tf.set_random_seed(args.seed)

    images_ph   = tf.placeholder(tf.float32, [None, args.target_size, args.target_size, 1],
                                 name='images')
    labels_ph   = tf.placeholder(tf.int32, [None, args.target_size, args.target_size],
                                 name='labels')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    batch_size_dyn = tf.shape(images_ph)[0]

    logits, preds, preds_map = build_model(
        images_ph, args.filter_type, args.filter_learnable,
        args.weight_decay, batch_size_dyn, is_training, num_classes=2)

    loss = focal_loss_tf(logits, labels_ph, gamma=args.focal_gamma)
    reg_losses = tf.losses.get_regularization_losses()
    total_loss = loss + (tf.add_n(reg_losses) if reg_losses else 0.0)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    itr_per_epoch = max(1, int(np.ceil(len(tr_imgs) / args.batch_size)))
    learning_rate = tf.train.exponential_decay(
        args.lr, global_step,
        decay_steps=int(itr_per_epoch * args.lr_decay_freq),
        decay_rate=args.lr_decay, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            total_loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True      # fall back to CPU if a GPU op fails to place

    n_params = sum(int(np.prod(v.shape)) for v in tf.trainable_variables())
    print(f"HP-FCN params={n_params:,}")

    def run_inference(sess, imgs, masks):
        all_scores, all_pmaps = [], []
        indices = np.arange(len(imgs))
        for batch_imgs, batch_masks in make_batches(imgs, masks, indices, args.batch_size):
            pm = sess.run(preds_map, feed_dict={
                images_ph: batch_imgs, labels_ph: batch_masks, is_training: False})
            for i in range(len(batch_imgs)):
                all_pmaps.append(pm[i])
                all_scores.append(float(pm[i].max()))
        return all_scores, all_pmaps

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if args.eval_only:
            ckpt = tf.train.get_checkpoint_state(str(out_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(f"Restored from {ckpt.model_checkpoint_path}")
            else:
                print("ERROR: No checkpoint found for eval_only mode"); return
        else:
            best_auc, patience_counter = 0.0, 0

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                perm = np.random.permutation(len(tr_imgs))
                epoch_loss, n_batches = 0.0, 0

                for batch_imgs, batch_masks in make_batches(
                        tr_imgs, tr_masks, perm, args.batch_size):
                    _, loss_val = sess.run(
                        [train_op, total_loss],
                        feed_dict={images_ph: batch_imgs, labels_ph: batch_masks,
                                   is_training: True})
                    epoch_loss += loss_val; n_batches += 1

                vl_scores, _ = run_inference(sess, vl_imgs, vl_masks)
                val_m = _compute_metrics(vl_labels, vl_scores, vl_mods_list)
                auc = val_m.get('auc', float('nan'))
                print(f"Epoch {epoch:03d}/{args.epochs}  "
                      f"loss={epoch_loss/max(n_batches,1):.4f}  val_AUC={auc:.4f}  "
                      f"({time.time()-t0:.1f}s)")

                if not np.isnan(auc) and auc > best_auc:
                    best_auc = auc; patience_counter = 0
                    saver.save(sess, str(out_dir / 'model.ckpt'), global_step=epoch)
                    print(f"  * New best AUC={best_auc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Early stopping at epoch {epoch}"); break

            ckpt = tf.train.get_checkpoint_state(str(out_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        print("\n=== Slice-Level Test ===")
        ts_scores, ts_pmaps = run_inference(sess, ts_imgs, ts_masks)
        sm = _compute_metrics(ts_labels, ts_scores, ts_mods_list)
        print(f"  AUC={sm.get('auc',float('nan')):.4f}  "
              f"Acc={sm.get('acc',float('nan')):.4f}")

        print(f"\n=== Volume-Level Test (K={args.K}) ===")
        tab_vols = tab_test[['mod', 'img_id']].drop_duplicates().reset_index(drop=True)
        ts_tab_list = list(tab_test.itertuples(index=False))
        vol_labels, vol_scores, vol_mods = [], [], []
        for _, vrow in tab_vols.iterrows():
            vmod, vid = vrow['mod'], str(vrow['img_id'])
            idxs = [i for i, r in enumerate(ts_tab_list)
                    if r.mod == vmod and str(r.img_id) == vid]
            if not idxs: continue
            vol_labels.append(0 if vmod == 'real' else 1)
            vol_scores.append(max(ts_scores[i] for i in idxs))
            vol_mods.append(vmod)
        vm = _compute_metrics(vol_labels, vol_scores, vol_mods)
        print(f"  AUC={vm.get('auc',float('nan')):.4f}  "
              f"Acc={vm.get('acc',float('nan')):.4f}")

        print("\n=== Localization Metrics (fake slices only) ===")
        loc_m = evaluate_localization_np(ts_pmaps, ts_masks, ts_labels, ts_mods_list)
        print(f"  PG={loc_m['pointing_game']:.4f}  "
              f"EoM={loc_m['energy_on_mask']:.4f}  "
              f"pAUC={loc_m['pixel_auc']:.4f}  "
              f"IoU@0.5={loc_m['iou_0.5']:.4f}")

        eval_dir = out_dir / 'evaluation'
        eval_dir.mkdir(exist_ok=True)
        with open(eval_dir / 'metrics.json', 'w') as f:
            json.dump(dict(slice=sm, volume=vm, localization=loc_m, args=args_dict), f, indent=2)

        print(f"\nDone. Results: {eval_dir/'metrics.json'}")


if __name__ == '__main__':
    main()
