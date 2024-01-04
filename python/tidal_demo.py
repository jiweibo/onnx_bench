import os
import argparse
from pathlib import Path
import glob
from typing import List

import numpy as np
from sess import Session

import cv2

np.random.seed(1998)


def parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("onnx", type=str, help="")
    parser.add_argument("onnx2", type=str, help="")
    # parser.add_argument('--batch', type=int, default=1, help='')
    parser.add_argument("--provider", choices=["cpu", "cuda", "trt"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--min_subgraph_size", type=int, default=1)
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


repeats = 1

start_h = 100
start_w = 400
in_height = 288
in_width = 608

model2_in_height = 192
model2_in_width = 192


def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = img[start_h : start_h + in_height, start_w : start_w + in_width, :]
    tgt_shape = list(img.shape)
    tgt_shape.insert(0, 1)
    return img.reshape(tgt_shape)


def crop(img):
    img = img[start_h : start_h + in_height, start_w : start_w + in_width, :]
    return img[np.newaxis, :, :, :]


def draw(img, img_path, det_bbox, det_scores, det_classes, threshold=0.5):
    height = img.shape[0]
    width = img.shape[1]
    idx = det_scores >= threshold
    bboxes = []
    for i in range(det_bbox[idx].shape[0]):
        ly = int(det_bbox[i][0] * height)
        lx = int(det_bbox[i][1] * width)
        ry = int(det_bbox[i][2] * height)
        rx = int(det_bbox[i][3] * width)
        mid_w = (lx + rx) / 2.0
        mid_h = (ly + ry) / 2.0
        cen_w = int(mid_w + start_w)
        cen_h = int(mid_h + start_h)
        bboxes.append((cen_h, cen_w))
        cv2.rectangle(img, (lx, ly), (rx, ry), (0, 255, 0), 1)
    save_path = str(Path(args.save_dir).joinpath(Path(img_path).name))
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    cv2.imwrite(save_path, img)
    return bboxes


def main(onnx_model, onnx_model2, args):
    sess = Session(
        onnx_model,
        args.provider,
        args.precision,
        args.cache_dir,
        args.min_subgraph_size,
        args.filter_ops,
    )
    sess2 = Session(
        onnx_model2,
        args.provider,
        args.precision,
        args.cache_dir,
        args.min_subgraph_size,
        args.filter_ops,
    )

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.img is not None:
        src_img_path = Path(args.img)
        base_img = cv2.imread(args.img)
        img = cv2.resize(base_img, None, fx=0.5, fy=0.5)
        img = crop(img)
        ins = [img]

        for i in range(repeats):
            out = sess.run(ins)

        det_bbox = out[0][0]  # [1, 100 ,4] -> [100, 4]
        det_scores = out[1][0]  # [1, 100] -> [100]
        det_classes = out[2][0]  # [1, 100] - > [100]
        bboxes = draw(img[0], args.img, det_bbox, det_scores, det_classes)

        resized_img = cv2.resize(base_img, None, fx=0.5, fy=0.5)
        resized_img = resized_img[np.newaxis, :, :, :]

        for i in range(len(bboxes)):
            cen_h, cen_w = bboxes[i]
            left = int(cen_w - model2_in_width / 2)
            right = int(cen_w + model2_in_width / 2)
            top = int(cen_h - model2_in_height / 2)
            down = int(cen_h + model2_in_height / 2)
            img = resized_img[
                :,
                top:down,
                left:right,
                :,
            ]
            out = sess2.run([img])
            bbox = out[0][0]
            scores = out[1][0]
            classes = out[2][0]
            draw(img[0], str(i) + ".jpg", bbox, scores, classes, 0.1)


if __name__ == "__main__":
    args = parse()
    main(args.onnx, args.onnx2, args)
