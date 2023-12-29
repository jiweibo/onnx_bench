import os
import argparse
import pathlib
import glob
from typing import List

import numpy as np
from sess import Session

import cv2

np.random.seed(1998)


def parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("onnx", type=str, help="")
    # parser.add_argument('--batch', type=int, default=1, help='')
    parser.add_argument("--provider", choices=["cpu", "cuda", "trt"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--filter_ops", type=str, default=None)
    parser.add_argument("--img", type=str, default=None, help="")
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="save_out")
    return parser.parse_args()


def generate_ort_data(shape, dtype):
    if dtype == "tensor(float)":
        return np.random.rand(*shape).astype(np.float32)
    elif dtype == "tensor(int32)":
        return np.random.randint(-128, 127, shape).astype(np.int32)
    elif dtype == "tensor(bool)":
        return np.random.randint(0, 1, shape).astype(np.bool)
    else:
        raise NotImplementedError("not support for %s" % dtype)


def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    tgt_shape = list(img.shape)
    tgt_shape.insert(0, 1)
    return img.reshape(tgt_shape)


def post_process(src_img_path, det_bbox, det_scores, threshold=0.5):
    img2 = cv2.imread(str(src_img_path))
    r, g, b = cv2.split(img2)
    img = cv2.merge([b, g, r])
    height = img.shape[0]
    width = img.shape[1]
    idx = det_scores >= threshold
    for i in range(det_bbox[idx].shape[0]):
        lx = int(det_bbox[i][1] * width)
        ly = int(det_bbox[i][0] * height)
        rx = int(det_bbox[i][3] * width)
        ry = int(det_bbox[i][2] * height)
        cv2.rectangle(img, (lx, ly), (rx, ry), (0, 255, 0), 1)

    save_path = str(pathlib.Path(args.save_dir).joinpath(src_img_path.name))
    cv2.imwrite(save_path, img)


def main(onnx_model, args):
    sess = Session(
        onnx_model, args.provider, args.precision, args.cache_dir, args.filter_ops
    )

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.img_dir is not None:
        for src_img_path in pathlib.Path(args.img_dir).glob("*.jpg"):
            ins = [read_image(str(src_img_path))]
            out = sess.run(ins)
            det_bbox = out[0][0]  # [1, 100 ,4] -> [100, 4]
            det_scores = out[1][0]  # [1, 100] -> [100]
            det_classes = out[2][0]  # [1, 100] - > [100]
            post_process(src_img_path, det_bbox, det_scores)

    if args.img is not None:
        src_img_path = pathlib.Path(args.img)
        ins = [read_image(str(src_img_path))]
        out = sess.run(ins)
        det_bbox = out[0][0]  # [1, 100 ,4] -> [100, 4]
        det_scores = out[1][0]  # [1, 100] -> [100]
        det_classes = out[2][0]  # [1, 100] - > [100]
        post_process(src_img_path, det_bbox, det_scores)


if __name__ == "__main__":
    args = parse()
    main(args.onnx, args)
