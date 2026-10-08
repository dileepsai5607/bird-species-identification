#!/usr/bin/env python3
"""
Reproducible experiments for:
  "A CNN-Based Bird Species Identification System for Biodiversity Monitoring
   Using Transfer Learning"

What it does
------------
1. Loads CUB-200-2011 with its OFFICIAL train/test split (5,994 / 5,794 images)
   and holds out a stratified 10% of the training images for validation.
2. Trains VGG16, ResNet50 and EfficientNet-B0 (the proposed system) with an
   identical head and a two-stage schedule:
     stage 1 - backbone frozen, only the head is trained (lr 1e-3)
     stage 2 - top of the backbone unfrozen, BatchNorm kept frozen (lr 1e-4)
3. Trains ablations without data augmentation (VGG16, EfficientNet-B0).
4. Evaluates every model on the 5,794-image test set (all 200 classes) and
   writes, from the REAL predictions:
     generated/results.tex          numbers used in the paper text and tables
     generated/table_*.tex          per-class tables
     generated/fig_*.pdf / .png     every figure in the paper
   (inside --paper-dir, default ./paper). Compile paper/main.tex afterwards and
   the paper is filled in automatically.

Runs are resumable: a finished run is skipped next time (handy on Colab).

Usage (GPU recommended):
    python run_experiments.py --download             # all five runs + report
    python run_experiments.py --runs effnetb0        # only the proposed model
    python run_experiments.py --report-only          # rebuild tables/figures
    python run_experiments.py --postprocess-only     # re-time models, Grad-CAM, report (CPU ok)

Requirements: tensorflow>=2.15 (Keras 3 or tf.keras 2), scikit-learn, matplotlib.
"""

import argparse
import json
import os
import sys
import tarfile
import textwrap
import time
import urllib.request

import numpy as np

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

RUN_SPECS = {
    # name           backbone          augment
    "vgg16":        ("vgg16",          True),
    "resnet50":     ("resnet50",       True),
    "effnetb0":     ("efficientnetb0", True),
    "vgg16_noaug":  ("vgg16",          False),
    "effnetb0_noaug": ("efficientnetb0", False),
}
MAIN_RUN = "effnetb0"   # the proposed system (selected by validation accuracy)

# macro prefix used in the paper -> run name
MACRO_RUNS = {"EffNet": "effnetb0", "EffNetNoAug": "effnetb0_noaug", "Vgg": "vgg16",
              "VggNoAug": "vgg16_noaug", "ResNet": "resnet50"}
FEAT_DIM = {"vgg16": 512, "resnet50": 2048, "efficientnetb0": 1280}

# Grad-CAM target layer and first layer unfrozen in stage 2, per backbone.
LAST_CONV = {"vgg16": "block5_conv3", "resnet50": "conv5_block3_out",
             "efficientnetb0": "top_activation"}
# first layer unfrozen in stage 2: --unfreeze-blocks 1 -> last block, 2 -> last two
UNFREEZE_FROM = {
    1: {"vgg16": "block5_conv1", "resnet50": "conv5_block1_1_conv",
        "efficientnetb0": "block6a_expand_conv"},
    2: {"vgg16": "block4_conv1", "resnet50": "conv4_block1_1_conv",
        "efficientnetb0": "block5a_expand_conv"},
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def download_cub(data_root):
    target = os.path.join(data_root, "CUB_200_2011")
    if os.path.isdir(os.path.join(target, "images")):
        return target
    os.makedirs(data_root, exist_ok=True)
    tgz = os.path.join(data_root, "CUB_200_2011.tgz")
    if not os.path.exists(tgz):
        print(f"Downloading CUB-200-2011 (~1.1 GB) from {CUB_URL}")
        # the Caltech server rejects Python's default user agent with HTTP 403
        req = urllib.request.Request(CUB_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(tgz + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tgz + ".part", tgz)
    print("Extracting ...")
    with tarfile.open(tgz) as t:
        t.extractall(data_root)
    return target


def read_cub(cub_dir, val_fraction=0.1, seed=42):
    """Returns dict with paths/labels for train/val/test and class names."""
    def read(name):
        with open(os.path.join(cub_dir, name)) as f:
            return [line.strip().split(" ", 1) for line in f if line.strip()]

    images = {int(i): p for i, p in read("images.txt")}
    labels = {int(i): int(c) - 1 for i, c in read("image_class_labels.txt")}
    is_train = {int(i): s == "1" for i, s in read("train_test_split.txt")}
    classes = [n for _, n in sorted(read("classes.txt"), key=lambda x: int(x[0]))]

    ids = sorted(images)
    train_ids = [i for i in ids if is_train[i]]
    test_ids = [i for i in ids if not is_train[i]]

    # stratified validation hold-out from the official training split
    rng = np.random.default_rng(seed)
    val_ids, tr_ids = [], []
    for c in range(len(classes)):
        cls_ids = [i for i in train_ids if labels[i] == c]
        rng.shuffle(cls_ids)
        n_val = max(1, int(round(len(cls_ids) * val_fraction)))
        val_ids += cls_ids[:n_val]
        tr_ids += cls_ids[n_val:]

    def pack(id_list):
        return ([os.path.join(cub_dir, "images", images[i]) for i in id_list],
                np.array([labels[i] for i in id_list], dtype=np.int32))

    return {"train": pack(sorted(tr_ids)), "val": pack(sorted(val_ids)),
            "test": pack(test_ids), "classes": classes}


POSSESSIVE = {"Heermann", "Brewer", "Bewick", "Anna", "Baird", "Henslow", "Nelson", "Harris",
              "Lincoln", "Wilson", "Swainson", "Brandt", "Clark", "Scott", "Le Conte"}
COMPOUND = ("sided|breasted|throated|eyed|bellied|headed|crowned|winged|tailed|billed|backed|"
            "rumped|footed|capped|necked|faced|legged|chinned|naped|cheeked|shouldered|collared")


def pretty(name):
    """'014.Indigo_Bunting' -> 'Indigo Bunting', with standard English bird-name spelling
    ('Artic Tern' -> 'Arctic Tern', 'Heermann Gull' -> "Heermann's Gull",
    'White breasted' -> 'White-breasted')."""
    import re
    n = name.split(".", 1)[-1].replace("_", " ")
    n = n.replace("Artic ", "Arctic ").replace("Wood Pewee", "Wood-Pewee")
    n = n.replace("Forsters Tern", "Forster's Tern").replace("Chuck will Widow", "Chuck-will's-widow")
    n = re.sub(rf"\b(\w+) ({COMPOUND})\b", r"\1-\2", n)
    for p_ in sorted(POSSESSIVE, key=len, reverse=True):
        if n.startswith(p_ + " ") and not n.startswith(p_ + "'s"):
            n = p_ + "'s" + n[len(p_):]
            break
    return n


# --------------------------------------------------------------------------
# TensorFlow part (only imported when training)
# --------------------------------------------------------------------------
def train_all(args, data):
    import tensorflow as tf
    try:
        import keras  # Keras 3
    except ImportError:  # pragma: no cover
        from tensorflow import keras

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}, GPUs: {gpus}")
    if gpus and not args.no_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(args.seed)

    S, B = args.img_size, args.batch_size
    AUTOTUNE = tf.data.AUTOTUNE

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (S, S), antialias=True)
        return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8), label

    def base_ds(split):
        paths, labels = data[split]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        return ds.map(load, num_parallel_calls=AUTOTUNE).cache()

    cached = {s: base_ds(s) for s in ("train", "val", "test")}

    f32 = {"dtype": "float32"}  # keep augmentation in float32 under mixed precision
    augmenter = keras.Sequential([
        keras.layers.RandomFlip("horizontal", **f32),
        keras.layers.RandomRotation(15.0 / 360.0, **f32),          # +/- 15 degrees
        keras.layers.RandomZoom(0.2, **f32),                        # +/- 20 %
        keras.layers.RandomBrightness(0.2, value_range=(0, 255), **f32),
        keras.layers.RandomContrast(0.2, **f32),
    ], name="augmentation")

    preprocess = {
        "vgg16": keras.applications.vgg16.preprocess_input,
        "resnet50": keras.applications.resnet50.preprocess_input,
        "efficientnetb0": keras.applications.efficientnet.preprocess_input,
    }

    def make_ds(split, backbone, augment, shuffle):
        ds = cached[split]
        if shuffle:
            ds = ds.shuffle(2048, seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.batch(B).map(lambda x, y: (tf.cast(x, tf.float32), y),
                             num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        pp = preprocess[backbone]
        return ds.map(lambda x, y: (pp(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    def build(backbone, pretrained=True):
        inp = keras.Input((S, S, 3), name="image")
        weights = None if (args.no_pretrained or not pretrained) else "imagenet"
        ctor = {"vgg16": keras.applications.VGG16,
                "resnet50": keras.applications.ResNet50,
                "efficientnetb0": keras.applications.EfficientNetB0}[backbone]
        base = ctor(include_top=False, weights=weights, input_tensor=inp)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
        x = keras.layers.Dense(args.dense_units, name="fc")(x)
        x = keras.layers.BatchNormalization(name="fc_bn")(x)
        x = keras.layers.Activation("relu", name="fc_relu")(x)
        x = keras.layers.Dropout(args.dropout, name="fc_dropout")(x)
        out = keras.layers.Dense(len(data["classes"]), activation="softmax",
                                 dtype="float32", name="predictions")(x)
        return keras.Model(inp, out, name=f"{backbone}_bird"), base

    def compile_model(model, lr):
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy",
                               keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5")])

    def callbacks():
        return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                              restore_best_weights=True, verbose=1),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                  patience=2, verbose=1)]

    def predict(model, backbone):
        ds = make_ds("test", backbone, augment=False, shuffle=False)
        return model.predict(ds, verbose=0).astype(np.float32)

    def n_trainable(model):
        return int(sum(np.prod(w.shape) for w in model.trainable_weights))

    def latency_ms(model):
        """Median single-image inference time over 7 rounds of 30 calls."""
        x = tf.random.uniform((1, S, S, 3), 0, 255)
        f = tf.function(lambda t: model(t, training=False))
        for _ in range(20):
            f(x).numpy()
        rounds = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(30):
                f(x).numpy()
            rounds.append((time.perf_counter() - t0) / 30 * 1000)
        return float(np.median(rounds))

    def hist(h):
        return {k: [float(v) for v in vals] for k, vals in h.history.items()}

    y_test = data["test"][1]
    for name in args.runs:
        run_dir = os.path.join(args.out, "runs", name)
        if os.path.exists(os.path.join(run_dir, "preds.npz")):
            with open(os.path.join(run_dir, "meta.json")) as f:
                old = json.load(f)
            if (old.get("lr2", 1e-5), old.get("unfreeze_blocks", 1)) != (args.lr2, args.unfreeze_blocks):
                sys.exit(f"[{name}] in {run_dir} was trained with different stage-2 settings "
                         f"(lr2={old.get('lr2', 1e-5)}, unfreeze_blocks={old.get('unfreeze_blocks', 1)}). "
                         "Use a new --out folder so all models share one protocol.")
            print(f"[{name}] already finished - skipping")
            continue
        os.makedirs(run_dir, exist_ok=True)
        backbone, augment = RUN_SPECS[name]
        print(f"\n===== {name}: backbone={backbone}, augmentation={augment} =====")
        model, base = build(backbone)
        tr = make_ds("train", backbone, augment, shuffle=True)
        va = make_ds("val", backbone, augment=False, shuffle=False)

        # ---- stage 1: frozen backbone
        for layer in base.layers:
            layer.trainable = False
        compile_model(model, args.lr1)
        meta = {"backbone": backbone, "augment": augment, "lr1": args.lr1,
                "lr2": args.lr2, "unfreeze_blocks": args.unfreeze_blocks,
                "params_total": int(model.count_params()),
                "params_trainable_stage1": n_trainable(model)}
        t0 = time.time()
        h1 = model.fit(tr, validation_data=va, epochs=args.epochs1,
                       callbacks=callbacks(), verbose=2)
        meta["train_time_stage1_min"] = (time.time() - t0) / 60
        probs_frozen = predict(model, backbone)

        # ---- stage 2: unfreeze the last block (BatchNorm layers stay frozen)
        names = [l.name for l in base.layers]
        start = names.index(UNFREEZE_FROM[args.unfreeze_blocks][backbone])
        for layer in base.layers[start:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
        compile_model(model, args.lr2)
        meta["params_trainable_stage2"] = n_trainable(model)
        t0 = time.time()
        h2 = model.fit(tr, validation_data=va, epochs=args.epochs2,
                       callbacks=callbacks(), verbose=2)
        meta["train_time_stage2_min"] = (time.time() - t0) / 60
        probs = predict(model, backbone)

        wpath = os.path.join(run_dir, "model.weights.h5")
        model.save_weights(wpath)
        meta["weights_mb"] = os.path.getsize(wpath) / 2 ** 20
        meta["latency_ms"] = latency_ms(model)
        meta["device"] = "CPU"
        if gpus:
            try:
                meta["device"] = tf.config.experimental.get_device_details(
                    gpus[0]).get("device_name", "GPU")
            except Exception:
                meta["device"] = "GPU"
        meta["history_stage1"], meta["history_stage2"] = hist(h1), hist(h2)
        meta["epochs_stage1"] = len(h1.history["loss"])
        meta["epochs_stage2"] = len(h2.history["loss"])

        # ---- Grad-CAM examples (main model only)
        if name == MAIN_RUN:
            save_gradcam(model, backbone, data, probs, preprocess[backbone],
                         run_dir, S, tf, keras, args.seed)

        np.savez_compressed(os.path.join(run_dir, "preds.npz"), y_true=y_test,
                            probs=probs.astype(np.float16),
                            probs_frozen=probs_frozen.astype(np.float16))
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[{name}] test top-1 frozen={np.mean(probs_frozen.argmax(1) == y_test):.4f} "
              f"fine-tuned={np.mean(probs.argmax(1) == y_test):.4f}")
        keras.backend.clear_session()

    # ---- time every finished model again in this one session, so that latencies
    #      are comparable (weights do not affect speed, so no loading is needed)
    if gpus:
        try:
            dev = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "") + " GPU"
        except Exception:
            dev = "GPU"
    else:
        dev = "CPU"
        try:
            with open("/proc/cpuinfo") as f:
                model = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
            model = model.replace("(R)", "").replace("(TM)", "").replace(" CPU", "")
            dev = f"{os.cpu_count()}-core {model} CPU"
        except Exception:
            pass
    for name, (backbone, _) in RUN_SPECS.items():
        mpath = os.path.join(args.out, "runs", name, "meta.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            meta = json.load(f)
        if "latency_ms_v2" in meta and meta.get("latency_device") == dev:
            continue  # already timed on this kind of machine
        model, _ = build(backbone, pretrained=False)
        meta["latency_ms_v2"] = latency_ms(model)
        meta["latency_device"] = dev
        print(f"[{name}] single-image latency {meta['latency_ms_v2']:.1f} ms")
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=1)
        keras.backend.clear_session()

    # ---- Grad-CAM for the main model from its saved weights, if it is missing
    main_dir = os.path.join(args.out, "runs", MAIN_RUN)
    wpath = os.path.join(main_dir, "model.weights.h5")
    if (os.path.exists(os.path.join(main_dir, "preds.npz")) and os.path.exists(wpath)
            and not os.path.exists(os.path.join(main_dir, "gradcam.npz"))):
        print(f"\nComputing Grad-CAM examples for {MAIN_RUN} from its saved weights ...")
        try:
            backbone = RUN_SPECS[MAIN_RUN][0]
            model, _ = build(backbone)
            model.load_weights(wpath)
            probs = np.load(os.path.join(main_dir, "preds.npz"))["probs"].astype(np.float32)
            check = predict(model, backbone)
            agree = np.mean(check.argmax(1) == probs.argmax(1))
            print(f"  reloaded model agrees with saved predictions on {100 * agree:.1f}% of test images")
            if agree < 0.98:
                raise RuntimeError("reloaded weights do not reproduce the saved predictions")
            save_gradcam(model, backbone, data, probs, preprocess[backbone],
                         main_dir, S, tf, keras, args.seed)
            print("  done.")
        except Exception as e:  # never lose a finished run because of the figure
            print(f"  WARNING: Grad-CAM could not be computed ({e}).")
        keras.backend.clear_session()


def save_gradcam(model, backbone, data, probs, pp, run_dir, S, tf, keras, seed):
    paths, y = data["test"]
    pred = probs.argmax(1)
    rng = np.random.default_rng(seed)
    correct = rng.choice(np.where(pred == y)[0], 4, replace=False)
    wrong_pool = np.where(pred != y)[0]
    wrong = rng.choice(wrong_pool, min(2, len(wrong_pool)), replace=False)
    idx = np.concatenate([correct, wrong])

    imgs = []
    for i in idx:
        img = tf.io.decode_jpeg(tf.io.read_file(paths[i]), channels=3)
        imgs.append(tf.cast(tf.image.resize(img, (S, S), antialias=True), tf.float32))
    imgs = tf.stack(imgs)

    grad_model = keras.Model(model.inputs,
                             [model.get_layer(LAST_CONV[backbone]).output, model.output])
    x = tf.convert_to_tensor(pp(tf.identity(imgs)))
    with tf.GradientTape() as tape:
        tape.watch(x)
        conv, out = grad_model(x, training=False)
        score = tf.gather(tf.cast(out, tf.float32), pred[idx], axis=1, batch_dims=1)
    grads = tf.cast(tape.gradient(score, conv), tf.float32)
    conv = tf.cast(conv, tf.float32)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * w, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    cam = tf.image.resize(cam[..., None], (S, S))[..., 0]
    np.savez_compressed(os.path.join(run_dir, "gradcam.npz"),
                        images=np.clip(imgs.numpy(), 0, 255).astype(np.uint8),
                        cams=cam.numpy().astype(np.float16), idx=idx,
                        y_true=y[idx], y_pred=pred[idx], conf=probs[idx, pred[idx]])


# --------------------------------------------------------------------------
# Report: metrics, tables, figures, results.tex  (numpy/sklearn/matplotlib only)
# --------------------------------------------------------------------------
def metrics(y, probs):
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    C = probs.shape[1]
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    onehot = np.eye(C)[y]
    return {
        "top1": float(np.mean(pred == y)),
        "top5": float(np.mean((top5 == y[:, None]).any(1))),
        "macro_p": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_r": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                   labels=list(range(C)))),
        "map": float(average_precision_score(onehot, probs, average="macro")),
    }


def make_report(args, classes, n_split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                                 precision_recall_fscore_support, roc_curve)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#d1661f", "#2a9d5c", "#7a7a7a"
    gen = os.path.join(args.paper_dir, "generated")
    os.makedirs(gen, exist_ok=True)
    C = len(classes)
    names = [pretty(c) for c in classes]

    def load(name):
        d = os.path.join(args.out, "runs", name)
        if not os.path.exists(os.path.join(d, "preds.npz")):
            return None
        z = np.load(os.path.join(d, "preds.npz"))
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        norm = lambda a: (a.astype(np.float64) / a.astype(np.float64).sum(1, keepdims=True))
        return {"y": z["y_true"].astype(int), "probs": norm(z["probs"]),
                "probs_frozen": norm(z["probs_frozen"]), "meta": meta, "dir": d}

    runs = {n: load(n) for n in RUN_SPECS}
    if runs[MAIN_RUN] is None:
        sys.exit(f"The main '{MAIN_RUN}' run has not finished yet - nothing to report.")

    pct = lambda v: f"{100 * v:.1f}"
    lines = ["% AUTO-GENERATED by run_experiments.py -- do not edit by hand.",
             f"% Generated {time.strftime('%Y-%m-%d %H:%M')}"]
    defn = lambda k, v: lines.append(f"\\renewcommand{{\\{k}}}{{{v}}}")
    na = "n/a"

    M, MF = {}, {}  # fine-tuned / frozen metrics per macro prefix
    for key, run in MACRO_RUNS.items():
        r = runs[run]
        if r is None:
            for k in ("TopOneFrozen", "TopOne", "TopFive", "MacroP", "MacroR", "MacroF",
                      "AUC", "MAP", "Params", "Trainable", "SizeMB", "Latency", "Val"):
                defn(f"{key}{k}", na)
            continue
        m = M[key] = metrics(r["y"], r["probs"])
        mf = MF[key] = metrics(r["y"], r["probs_frozen"])
        mt = r["meta"]
        defn(f"{key}TopOneFrozen", pct(mf["top1"]))
        defn(f"{key}TopOne", pct(m["top1"]))
        defn(f"{key}TopFive", pct(m["top5"]))
        defn(f"{key}MacroP", pct(m["macro_p"]))
        defn(f"{key}MacroR", pct(m["macro_r"]))
        defn(f"{key}MacroF", pct(m["macro_f1"]))
        defn(f"{key}AUC", f"{m['auc']:.3f}")
        defn(f"{key}MAP", pct(m["map"]))
        defn(f"{key}Params", f"{mt['params_total'] / 1e6:.1f}M")
        defn(f"{key}Trainable", f"{mt['params_trainable_stage2'] / 1e6:.1f}M")
        defn(f"{key}SizeMB", f"{mt['params_total'] * 4 / 2 ** 20:.0f}")  # float32 weights
        defn(f"{key}Latency", f"{mt.get('latency_ms_v2', mt['latency_ms']):.1f}")
        # validation accuracy of the restored (lowest validation loss) stage-2 epoch
        h2 = mt["history_stage2"]
        best = int(np.argmin(h2["val_loss"]))
        defn(f"{key}Val", pct(h2["val_accuracy"][best]))

    signed = lambda a, b: f"{100 * (a - b):+.1f}"
    defn("FineTuneGain", signed(M["EffNet"]["top1"], MF["EffNet"]["top1"]))
    for key, ref in (("Vgg", "Vgg"), ("ResNet", "ResNet")):
        defn(f"FineTuneGain{key}", signed(M[key]["top1"], MF[key]["top1"]) if key in M else na)
    defn("AugGain", signed(M["EffNet"]["top1"], M["EffNetNoAug"]["top1"])
         if "EffNetNoAug" in M else na)
    defn("AugGainVgg", signed(M["Vgg"]["top1"], M["VggNoAug"]["top1"])
         if {"Vgg", "VggNoAug"} <= set(M) else na)

    main = runs[MAIN_RUN]
    vm = main["meta"]
    defn("MainFrozenTrainable", f"{vm['params_trainable_stage1'] / 1e6:.2f}M")
    defn("EpochsStageOne", str(vm["epochs_stage1"]))
    defn("EpochsStageTwo", str(vm["epochs_stage2"]))
    defn("TrainDevice", vm.get("device", "GPU"))
    defn("LatencyDevice", vm.get("latency_device", vm.get("device", "GPU") + " GPU")
         .replace("_", " ").replace("@", "at"))
    defn("AugGainVggTopFive", signed(M["Vgg"]["top5"], M["VggNoAug"]["top5"])
         if {"Vgg", "VggNoAug"} <= set(M) else na)
    if "EffNetNoAug" in M:
        cells = ["EfficientNet-B0"] + [f"\\EffNetNoAug{k}" for k in (
            "Val", "TopOneFrozen", "TopOne", "TopFive", "MacroP", "MacroR", "MacroF", "AUC",
            "Params", "SizeMB", "Latency")]
        defn("EffNetNoAugRow", " & ".join(cells) + r" \\")
        defn("AugSentenceEffNet", f"For EfficientNet-B0 the change in top-1 accuracy is "
             f"{signed(M['EffNet']['top1'], M['EffNetNoAug']['top1'])} percentage points.")
    else:
        defn("EffNetNoAugRow", "")
        defn("AugSentenceEffNet", "")
    defn("NTrain", f"{n_split['train']:,}")
    defn("NVal", f"{n_split['val']:,}")
    defn("NTest", f"{n_split['test']:,}")
    defn("NTotal", f"{sum(n_split.values()):,}")

    # parameter breakdown of the main model (head computed from its layer sizes)
    D, F = args.dense_units, FEAT_DIM[vm["backbone"]]
    fc, bn, out = F * D + D, 4 * D, D * C + C
    total = vm["params_total"]
    defn("ArchBackbone", f"{total - fc - bn - out:,}")
    defn("ArchFc", f"{fc + bn:,}")
    defn("ArchOut", f"{out:,}")
    defn("ArchTotal", f"{total:,}")
    defn("ArchFeat", str(F))

    # ---------- per-class analysis of the main model
    y, probs = main["y"], main["probs"]
    pred = probs.argmax(1)
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(C)),
                                                    zero_division=0)
    defn("NClassesPerfect", str(int(np.sum(f1 >= 0.999))))
    defn("NClassesBelowHalf", str(int(np.sum(f1 < 0.5))))
    defn("MedianClassF", pct(float(np.median(f1))))
    defn("MinClassF", pct(float(f1.min())))
    defn("MaxClassF", pct(float(f1.max())))

    order = np.lexsort((-sup, -f1))  # by F1 desc, then support desc
    best, worst = order[:5], order[::-1][:5]
    t = ["% AUTO-GENERATED", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Species & P (\%) & R (\%) & F1 (\%) & $n$ \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{Five best-recognised species}}\\"]
    fmt = lambda c: (f"{names[c]} & {pct(p[c])} & {pct(r[c])} & {pct(f1[c])} & {sup[c]} \\\\")
    t += [fmt(c) for c in best]
    t += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Five worst-recognised species}}\\"]
    t += [fmt(c) for c in worst]
    t += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(gen, "table_bestworst.tex"), "w") as f:
        f.write("\n".join(t) + "\n")

    cm = confusion_matrix(y, pred, labels=list(range(C)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None, kind="stable")[:6]
    t = ["% AUTO-GENERATED", r"\begin{tabular}{@{}llr@{}}", r"\toprule",
         r"True species & Predicted as & Count \\", r"\midrule"]
    for k in flat:
        i, j = np.unravel_index(k, off.shape)
        if off[i, j] == 0:
            break
        t.append(f"{names[i]} & {names[j]} & {off[i, j]} of {cm[i].sum()} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(gen, "table_confused.tex"), "w") as f:
        f.write("\n".join(t) + "\n")

    # ---------- Fig: training curves of the main model (both stages)
    h1, h2 = vm["history_stage1"], vm["history_stage2"]
    e1 = len(h1["loss"])
    ep = np.arange(1, e1 + len(h2["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for a, key, lab in ((ax[0], "accuracy", "Accuracy"), (ax[1], "loss", "Cross-entropy loss")):
        a.plot(ep, h1[key] + h2[key], color=BLUE, lw=1.4, label="Train")
        a.plot(ep, h1["val_" + key] + h2["val_" + key], color=ORANGE, lw=1.4, label="Validation")
        a.axvline(e1 + 0.5, color=GREY, ls="--", lw=0.8)
        a.text(e1 + 0.7, a.get_ylim()[1], "fine-tuning starts", fontsize=6.5,
               color=GREY, va="top")
        a.set_xlabel("Epoch")
        a.set_ylabel(lab)
        a.grid(alpha=0.25, lw=0.5)
    ax[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(gen, "fig_curves.pdf"))
    plt.close(fig)

    # ---------- Fig: confusion matrix (row-normalised, 200x200)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_xticks([0, 49, 99, 149, 199], ["1", "50", "100", "150", "200"])
    ax.set_yticks([0, 49, 99, 149, 199], ["1", "50", "100", "150", "200"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Fraction of true class")
    fig.tight_layout()
    fig.savefig(os.path.join(gen, "fig_confusion.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig: per-class F1 histogram
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.hist(100 * f1, bins=np.arange(0, 105, 5), color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(100 * np.mean(f1), color=ORANGE, ls="--", lw=1,
               label=f"Macro F1 = {100 * np.mean(f1):.1f}%")
    ax.set_xlabel("Per-class F1-score (%)")
    ax.set_ylabel("Number of species")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(gen, "fig_f1hist.pdf"))
    plt.close(fig)

    # ---------- Fig: micro-averaged ROC and PR curves
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    curves = (("EffNet", "probs", "EfficientNet-B0 fine-tuned (ours)", BLUE, "-"),
              ("EffNet", "probs_frozen", "EfficientNet-B0 frozen", BLUE, ":"),
              ("ResNet", "probs", "ResNet50 fine-tuned", ORANGE, "-"),
              ("Vgg", "probs", "VGG16 fine-tuned", GREEN, "-"))
    for key, which, lab, col, ls in curves:
        run = runs[MACRO_RUNS[key]]
        if run is None:
            continue
        mm = M[key] if which == "probs" else MF[key]
        onehot = np.eye(C)[run["y"]].ravel()
        fpr, tpr, _ = roc_curve(onehot, run[which].ravel())
        prec, rec, _ = precision_recall_curve(onehot, run[which].ravel())
        ax[0].plot(fpr, tpr, color=col, ls=ls, lw=1.2, label=f"{lab} (macro AUC {mm['auc']:.3f})")
        ax[1].plot(rec, prec, color=col, ls=ls, lw=1.2, label=f"{lab} (mAP {100 * mm['map']:.1f}%)")
    ax[0].plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_xlim(1e-4, 1)
    ax[0].set_xlabel("False positive rate (log scale)")
    ax[0].set_ylabel("True positive rate")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.legend(frameon=False, fontsize=6, loc="lower left" if a is ax[1] else "lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(gen, "fig_roc_pr.pdf"))
    plt.close(fig)

    # ---------- Fig: Grad-CAM examples of the main model
    gpath = os.path.join(main["dir"], "gradcam.npz")
    if os.path.exists(gpath):
        g = np.load(gpath)
        n = len(g["idx"])
        fig, ax = plt.subplots(2, n, figsize=(7.0, 3.6))
        ax = np.atleast_2d(ax).reshape(2, n)
        wrap = lambda t: "\n".join(textwrap.wrap(t, 20))
        for k in range(n):
            ok = g["y_true"][k] == g["y_pred"][k]
            ax[0, k].imshow(g["images"][k])
            ax[0, k].set_title("True: " + wrap(names[g["y_true"][k]]), fontsize=6)
            ax[1, k].imshow(g["images"][k])
            ax[1, k].imshow(g["cams"][k].astype(np.float32), cmap="jet", alpha=0.45)
            ax[1, k].set_title("Pred: " + wrap(names[g["y_pred"][k]])
                               + f"\n({100 * g['conf'][k]:.0f}%)" + ("" if ok else " - wrong"),
                               fontsize=6, color="black" if ok else "#c0392b")
            for a in ax[:, k]:
                a.set_xticks([])
                a.set_yticks([])
                for s in a.spines.values():
                    s.set_visible(False)
        fig.tight_layout(h_pad=1.4, w_pad=0.3)
        fig.savefig(os.path.join(gen, "fig_gradcam.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"NOTE: no Grad-CAM file for {MAIN_RUN} yet - run any training command "
              "(e.g. --runs effnetb0_noaug) on a machine with TensorFlow to create it.")

    with open(os.path.join(gen, "results.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(gen, "summary.json"), "w") as f:
        json.dump({"fine_tuned": M, "frozen": MF}, f, indent=1)

    print("\nTest-set results (200 classes, %d images):" % len(y))
    for key, m in M.items():
        print(f"  {key:12s} frozen {100 * MF[key]['top1']:5.1f}  top-1 {100 * m['top1']:5.1f}  "
              f"top-5 {100 * m['top5']:5.1f}  macro-F1 {100 * m['macro_f1']:5.1f}  AUC {m['auc']:.3f}")
    print(f"  {MAIN_RUN} parameter check: backbone {total - fc - bn - out:,} + head "
          f"{fc + bn + out:,} = {total:,}")
    print(f"\nWrote tables, figures and results.tex to {gen}/ - now compile main.tex.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", help="folder that holds CUB_200_2011/")
    ap.add_argument("--download", action="store_true", help="download CUB-200-2011 if missing")
    ap.add_argument("--out", default="experiment_outputs", help="where runs are saved")
    ap.add_argument("--paper-dir", default="paper", help="folder containing main.tex")
    ap.add_argument("--runs", nargs="+", default=list(RUN_SPECS), choices=list(RUN_SPECS))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--postprocess-only", action="store_true",
                    help="no training: re-time all finished models on this machine, create the "
                         "Grad-CAM figure from saved weights, rebuild the report (works on CPU)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dense-units", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs1", type=int, default=30)
    ap.add_argument("--epochs2", type=int, default=30)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--unfreeze-blocks", type=int, default=2, choices=[1, 2],
                    help="backbone blocks fine-tuned in stage 2 (2 = last two, as in the paper)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="(testing only)")
    ap.add_argument("--no-mixed-precision", action="store_true")
    args = ap.parse_args()

    cub = os.path.join(args.data_root, "CUB_200_2011")
    if args.download:
        cub = download_cub(args.data_root)
    if not os.path.isdir(os.path.join(cub, "images")):
        sys.exit(f"CUB-200-2011 not found in {cub}. Use --download or --data-root.")
    data = read_cub(cub, seed=args.seed)
    n_split = {s: len(data[s][1]) for s in ("train", "val", "test")}
    print("Split sizes:", n_split)

    if args.postprocess_only:
        args.runs = []
    if not args.report_only:
        train_all(args, data)
    make_report(args, data["classes"], n_split)


if __name__ == "__main__":
    main()
