#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

#  GPU selection — MUST happen before any TensorFlow import.
#  TF enumerates GPUs at import time; setting CUDA_VISIBLE_DEVICES afterwards

def _select_gpu_early():
    existing = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if existing is not None:
        print(f"Using pre-set CUDA_VISIBLE_DEVICES={existing}")
        return

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

    print("Could not auto-select GPU via nvidia-smi; leaving CUDA_VISIBLE_DEVICES unchanged")

_select_gpu_early()

#  Keras 2.2 → tf.keras compatibility shim
#  Allows importing modelCore.py (written for standalone Keras 2.2)
#  under TF 2.x without modifying the repository code.

def _install_keras_shim():
    import tensorflow as tf

    keras = types.ModuleType('keras')
    keras.__path__ = []

    keras.layers = tf.keras.layers
    keras.models = tf.keras.models
    keras.constraints = tf.keras.constraints
    keras.activations = tf.keras.activations
    keras.initializers = tf.keras.initializers
    keras.backend = tf.keras.backend

    # keras.layers.convolutional._Conv — base Conv class that accepts rank=
    _conv_mod = types.ModuleType('keras.layers.convolutional')

    class _ConvCompat(tf.keras.layers.Conv2D):
        def __init__(self, rank=2, **kwargs):
            self._rank_compat = rank
            super().__init__(**kwargs)
        def get_config(self):
            cfg = super().get_config()
            cfg['rank'] = self._rank_compat
            return cfg

    _conv_mod._Conv = _ConvCompat
    keras.layers.convolutional = _conv_mod

    # keras.legacy.interfaces — provides legacy_conv2d_support (no-op decorator)
    _legacy = types.ModuleType('keras.legacy')
    _legacy.__path__ = []
    _iface = types.ModuleType('keras.legacy.interfaces')
    _iface.legacy_conv2d_support = lambda fn: fn   # no-op
    _legacy.interfaces = _iface
    keras.legacy = _legacy

    # keras.engine — provides InputSpec
    _engine = types.ModuleType('keras.engine')
    _engine.InputSpec = tf.keras.layers.InputSpec
    keras.engine = _engine

    sys.modules['keras'] = keras
    sys.modules['keras.layers'] = keras.layers
    sys.modules['keras.layers.convolutional'] = _conv_mod
    sys.modules['keras.models'] = keras.models
    sys.modules['keras.constraints'] = keras.constraints
    sys.modules['keras.activations'] = keras.activations
    sys.modules['keras.initializers'] = keras.initializers
    sys.modules['keras.backend'] = keras.backend
    sys.modules['keras.legacy'] = _legacy
    sys.modules['keras.legacy.interfaces'] = _iface
    sys.modules['keras.engine'] = _engine

# Install shim BEFORE any repo imports
_install_keras_shim()

import tensorflow as tf

WORK_DIR      = Path(__file__).resolve().parent.parent
MANTRANET_DIR = WORK_DIR / 'baselines' / 'git_repo' / 'ManTraNet'
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
sys.path.insert(0, str(MANTRANET_DIR))
from config import DATA_DIR

from hexmil.data.patch_dataset import load_split_table, MOD_LABEL
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

from src.modelCore import create_model, load_pretrain_model_by_index
import src.modelCore as _mc

# In TF2/Keras, `Layer.kernel` is a read-only property — assigning to it in
# build() raises AttributeError.  We patch build() and call() to store the
# concatenated kernel under `_combined_kernel` instead.
def _combined_conv2d_build(self, input_shape):
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    input_dim = input_shape[channel_axis]
    # 1. regular trainable kernels
    filters = self.filters - 9 - 3
    if filters >= 1:
        self.regular_kernel = self.add_weight(
            shape=self.kernel_size + (input_dim, filters),
            initializer=self.kernel_initializer, name='regular_kernel',
            regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
    else:
        self.regular_kernel = None
    # 2. SRM kernels (fixed)
    # Keep this as a plain tensor, not a Layer weight. The original Keras 2.2
    # implementation did not serialize SRM as a tracked weight, and the
    # pretrained .h5 expects that layout.
    srm_list = self._get_srm_list()
    srm_kernel_list = []
    for srm in srm_list:
        for ch in range(int(input_dim)):
            this_ch_kernel = np.zeros((5, 5, int(input_dim)), dtype='float32')
            this_ch_kernel[:, :, ch] = srm
            srm_kernel_list.append(this_ch_kernel)
    self._srm_kernel = tf.constant(np.stack(srm_kernel_list, axis=-1), dtype=tf.float32)
    # 3. Bayar kernels (constrained trainable)
    self.bayar_kernel = self.add_weight(
        shape=self.kernel_size + (input_dim, 3),
        initializer=self.kernel_initializer, name='bayar_kernel',
        regularizer=self.kernel_regularizer, constraint=_mc.BayarConstraint())
    # 4. Concatenate — store under _combined_kernel; .kernel is read-only in TF2
    self.input_spec = tf.keras.layers.InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

def _combined_conv2d_call(self, inputs):
    kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) \
              else (self.kernel_size, self.kernel_size)
    inputs_pad = tf.pad(inputs, [[0,0],[kh//2,kh//2],[kw//2,kw//2],[0,0]], mode='symmetric')
    # Build the concatenated kernel at call-time so gradients flow to trainable
    # regular/Bayar kernels under TF2 eager execution.
    all_kernels = ([self.regular_kernel] if self.regular_kernel is not None else []) \
                  + [self._srm_kernel, self.bayar_kernel]
    combined_kernel = tf.keras.backend.concatenate(all_kernels, axis=-1)
    outputs = tf.keras.backend.conv2d(
        inputs_pad, combined_kernel,
        strides=self.strides, padding='valid',
        data_format=self.data_format, dilation_rate=self.dilation_rate)
    if self.use_bias:
        outputs = tf.keras.backend.bias_add(outputs, self.bias, data_format=self.data_format)
    return self.activation(outputs) if self.activation is not None else outputs

_mc.CombinedConv2D.build = _combined_conv2d_build
_mc.CombinedConv2D.call  = _combined_conv2d_call

# BayarConstraint in legacy code uses in-place ops (w *= ...), which are not
# supported for tf.Variable under TF2. Keep the same math with pure tensor ops.
def _bayar_constraint_call(self, w):
    if self.mask is None:
        self._initialize_mask(w)
    w = w * (1 - self.mask)
    rest_sum = tf.keras.backend.sum(w, axis=(0, 1), keepdims=True)
    w = w / (rest_sum + tf.keras.backend.epsilon())
    w = w - self.mask
    return w

_mc.BayarConstraint.__call__ = _bayar_constraint_call

# TF2 ConvLSTM2D requires static H/W dims. ManTraNet builds with (None,None,3).
# We pin to (224,224,3) — the target_size used for all CT slices in this script.
_orig_mc_Input = _mc.Input
def _fixed_Input(*args, **kwargs):
    if kwargs.get('shape') == (None, None, 3):
        kwargs['shape'] = (224, 224, 3)
    return _orig_mc_Input(*args, **kwargs)
_mc.Input = _fixed_Input

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']
PRETRAINED_DIR = str(MANTRANET_DIR / 'pretrained_weights')


def select_gpu(gpu_id=None):
    # GPU selection is handled by _select_gpu_early() before TF import.
    # This function is kept as a no-op so the call in main() is harmless.
    pass

#  Data loading — M3DSynth CT slices as numpy

def load_slices_numpy(data_dir, tab, target_size=224):
    from skimage.transform import resize as sk_resize
    images, masks, labels, mods, img_ids = [], [], [], [], []
    for _, row in tab.iterrows():
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
            img  = sk_resize(img,  (target_size, target_size),
                             preserve_range=True).astype(np.float32)
            mask = sk_resize(mask, (target_size, target_size),
                             order=0, preserve_range=True).astype(np.float32)

        # ManTraNet expects (H, W, 3) — repeat 1-ch to 3-ch
        images.append(np.stack([img, img, img], axis=-1))  # (H, W, 3)
        masks.append(mask)                                  # (H, W)
        labels.append(0 if mod == 'real' else 1)
        mods.append(mod)
        img_ids.append(img_id)

    return (np.array(images), np.array(masks), np.array(labels),
            mods, img_ids)

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


def binary_focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)
    return loss_fn


def get_args():
    p = argparse.ArgumentParser(description='ManTraNet (Wu et al., CVPR 2019) — Keras, 1-ch CT')
    p.add_argument('--data_dir',        type=str,   default=DATA_DIR)
    p.add_argument('--out_dir',         type=str,   default=None)
    p.add_argument('--K',               type=int,   default=16)
    p.add_argument('--target_size',     type=int,   default=224)
    p.add_argument('--pretrain_index',  type=int,   default=4,
                   help='Pretrained model index 0-4 (default: 4 = frozen featex)')
    p.add_argument('--focal_gamma',     type=float, default=2.0)
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--patience',        type=int,   default=12)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--gpu_id',          type=int,   default=None)
    p.add_argument('--train_mods',      nargs='+',  default=None)
    p.add_argument('--eval_only',       action='store_true', default=False)
    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    select_gpu(args.gpu_id)

    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))

    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'mantranet_K{args.K}' / f'trained_on_{mods_tag}'
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
        args.data_dir, tab_train, args.target_size)
    vl_imgs, vl_masks, vl_labels, vl_mods_list, _ = load_slices_numpy(
        args.data_dir, tab_valid, args.target_size)
    ts_imgs, ts_masks, ts_labels, ts_mods_list, _ = load_slices_numpy(
        args.data_dir, tab_test,  args.target_size)
    print(f"Train: {len(tr_imgs)}, Valid: {len(vl_imgs)}, Test: {len(ts_imgs)}")

    # Expand masks to (H, W, 1) for Keras
    tr_masks_4d = tr_masks[..., np.newaxis]
    vl_masks_4d = vl_masks[..., np.newaxis]

    print(f"Loading ManTraNet (pretrain_index={args.pretrain_index})...")
    model = load_pretrain_model_by_index(args.pretrain_index, PRETRAINED_DIR)
    n_params = model.count_params()
    n_train = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"ManTraNet: {n_params:,} params total, {n_train:,} trainable")
    model.summary(print_fn=lambda x: None)  # suppress verbose summary

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=binary_focal_loss(gamma=args.focal_gamma),
    )

    def run_inference(imgs):
        pred_maps = model.predict(imgs, batch_size=args.batch_size, verbose=0)
        pred_maps = pred_maps[:, :, :, 0]  # (N, H, W)
        scores = [float(pm.max()) for pm in pred_maps]
        return scores, list(pred_maps)

    if not args.eval_only:
        best_auc, patience_counter = 0.0, 0
        best_weights_path = str(out_dir / 'best_model.weights.h5')

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            perm = np.random.permutation(len(tr_imgs))
            hist = model.fit(
                tr_imgs[perm], tr_masks_4d[perm],
                batch_size=args.batch_size,
                epochs=1, verbose=0,
            )
            train_loss = hist.history['loss'][0]

            vl_scores, _ = run_inference(vl_imgs)
            val_m = _compute_metrics(vl_labels, vl_scores, vl_mods_list)
            auc = val_m.get('auc', float('nan'))
            print(f"Epoch {epoch:03d}/{args.epochs}  "
                  f"loss={train_loss:.4f}  val_AUC={auc:.4f}  "
                  f"({time.time()-t0:.1f}s)")

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc; patience_counter = 0
                model.save_weights(best_weights_path)
                print(f"  * New best AUC={best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}"); break

        model.load_weights(best_weights_path)
    else:
        best_weights_path = str(out_dir / 'best_model.weights.h5')
        if os.path.exists(best_weights_path):
            model.load_weights(best_weights_path)
            print(f"Loaded weights from {best_weights_path}")
        else:
            print("WARNING: No saved weights found, using pretrained only")

    print("\n=== Slice-Level Test ===")
    ts_scores, ts_pmaps = run_inference(ts_imgs)
    sm = _compute_metrics(ts_labels, ts_scores, ts_mods_list)
    print(f"  AUC={sm.get('auc',float('nan')):.4f}  "
          f"Acc={sm.get('acc',float('nan')):.4f}")

    # Volume-level: max score per volume
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
    for mod, mm in vm.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}")

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
