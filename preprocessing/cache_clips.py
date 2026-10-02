"""
Pre-extract face clips ONCE so training never touches the mp4 files.

For every CSV row (one 16-frame clip) this writes
    <cache_dir>/<row_index:06d>.npy      uint8  [T, 224, 224, 3]  (RGB)

It produces exactly the same pixels as DeepfakeDataset._read_frame
(frame_id - 1 -> OpenCV position, clamp bbox, crop, INTER_LINEAR resize),
but each video is decoded ONCE, sequentially, and shared by all of its clips.

Usage (from project root):
    python preprocessing/cache_clips.py --csv data/split/train.csv --out data/cache/train
    python preprocessing/cache_clips.py --csv data/split/val.csv   --out data/cache/val
    python preprocessing/cache_clips.py --csv data/split/train.csv --out data/cache/train --limit 200   # timing test
    python preprocessing/cache_clips.py --csv data/split/train.csv --out data/cache/train --verify 20   # compare with old reader

It is resumable: rows whose .npy already exists are skipped.
The cache is tied to the row ORDER of the CSV. If you regenerate the CSV, delete the cache.
"""
import argparse
import ast
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)


def build_video_index(video_root):
    index = {}
    for dirpath, _, files in os.walk(video_root):
        for f in files:
            if f.lower().endswith(".mp4"):
                index.setdefault(f, os.path.join(dirpath, f))
    return index


def parse_ids(v):
    return [int(x) for x in str(v).split(",") if x.strip()]


def crop_face(frame_bgr, bbox, size):
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w));     y2 = max(0, min(y2, h))
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"invalid bbox {bbox}")
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)


def process_video(job):
    """job = (video_path, [(row_idx, frame_ids, bboxes), ...], out_dir, size, n_frames)"""
    video_path, rows, out_dir, size, n_frames = job
    errors = []

    # needed[pos] -> list of (row_idx, slot, bbox)
    needed = defaultdict(list)
    buffers, remaining = {}, {}
    for row_idx, fids, boxes in rows:
        buffers[row_idx] = np.zeros((n_frames, size, size, 3), np.uint8)
        remaining[row_idx] = len(fids)
        for slot, (fid, bb) in enumerate(zip(fids, boxes)):
            needed[fid - 1].append((row_idx, slot, bb))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, [f"cannot open {video_path}"]

    done = 0
    try:
        positions = sorted(needed)
        last = positions[-1]
        wanted = set(positions)
        first = positions[0]
        if first > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)   # one seek only
        pos = first
        while pos <= last:
            if not cap.grab():
                errors.append(f"{video_path}: ran out of frames at {pos}")
                break
            if pos in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    errors.append(f"{video_path}: retrieve failed at {pos}")
                else:
                    for row_idx, slot, bb in needed[pos]:
                        try:
                            buffers[row_idx][slot] = crop_face(frame, bb, size)
                            remaining[row_idx] -= 1
                        except Exception as e:
                            errors.append(f"{video_path} row {row_idx}: {e}")
                        if remaining[row_idx] == 0:
                            np.save(os.path.join(out_dir, f"{row_idx:06d}.npy"), buffers.pop(row_idx))
                            done += 1
            pos += 1
    finally:
        cap.release()
    return done, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-root", default="data/raw")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (timing test)")
    ap.add_argument("--verify", type=int, default=0, help="compare N cached clips against the old reader and exit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.iloc[: args.limit]
    vindex = build_video_index(args.video_root)
    print(f"{len(df):,} rows, {len(vindex):,} mp4 files found under {args.video_root}")

    if args.verify:
        sys.path.insert(0, os.getcwd())
        from data.raw.dataset import DeepfakeDataset
        ds = DeepfakeDataset(metadata_csv=args.csv, video_root=args.video_root)
        rng = np.random.default_rng(0)
        for i in rng.choice(len(df), size=min(args.verify, len(df)), replace=False):
            old, _ = ds[int(i)]                       # [T,3,H,W] float 0..1
            old = (old.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
            new = np.load(os.path.join(args.out, f"{int(i):06d}.npy"))
            print(f"row {i}: max abs diff = {np.abs(old.astype(int) - new.astype(int)).max()}")
        return

    groups = defaultdict(list)
    skipped = 0
    for i, r in enumerate(df.itertuples(index=False)):
        if os.path.exists(os.path.join(args.out, f"{i:06d}.npy")):
            skipped += 1
            continue
        fids = parse_ids(r.frame_ids)
        boxes = ast.literal_eval(str(r.bboxes))
        if len(fids) != len(boxes) or len(fids) != args.frames:
            print(f"row {i}: bad frame/bbox count, skipped")
            continue
        vp = vindex.get(str(r.video_name))
        if vp is None:
            print(f"row {i}: video not found: {r.video_name}")
            continue
        groups[vp].append((i, fids, boxes))
    print(f"already cached: {skipped:,}   videos to decode: {len(groups):,}")

    jobs = [(vp, rows, args.out, args.size, args.frames) for vp, rows in groups.items()]
    t0, total, n_err = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for k, (done, errs) in enumerate(pool.imap_unordered(process_video, jobs), 1):
            total += done
            n_err += len(errs)
            for e in errs[:3]:
                print("  !", e)
            if k % 25 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"videos {k}/{len(jobs)}  clips {total:,}  {total / max(el, 1e-9):.1f} clips/s  elapsed {el / 60:.1f} min")
    print(f"done. clips written: {total:,}   errors: {n_err}")


if __name__ == "__main__":
    main()
