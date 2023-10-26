import numpy as np
import argparse

def read_data(filename):
    data = np.load(filename)
    dic = {}
    for name in data.files:
        dic[name] = data[name]
    return dic

def parse():
    parser = argparse.ArgumentParser('')
    parser.add_argument('base', type=str, help='')
    parser.add_argument('ref', type=str, help='')
    return parser.parse_args()

def check(base, ref):
    if base.shape != ref.shape:
        print('base.shape is %s but ref.shape is %s' % (str(base.shape), str(ref.shape)))
        return -1
    print('shape is ', base.shape)
    if base.size == 0:
        return
    if base.dtype == np.bool:
        base = base.astype(np.int32)
        ref = ref.astype(np.int32)
    base = base.flatten()
    ref = ref.flatten()
    diff = base - ref
    abs_diff = np.abs(diff)
    rerr = abs_diff / (base + 1e-9)

    print('- avg abs_diff: {0:.6f}%'.format(np.mean(abs_diff) * 100))
    print('- avg relative_diff: {0:.6f}%'.format(np.mean(rerr) * 100))

    idx = np.argmax(abs_diff)
    print("max_diff: %f (base val is %f, ref val is %f)" % (abs_diff[idx], base[idx], ref[idx]))

    k = min(10, abs_diff.size)
    top_k_idx = np.argsort(abs_diff)[::-1][:k]
    print("top_k max_diff (k = %d):" % k)
    print("  base val is %s" % str(base[top_k_idx]))
    print("  ref val is %s" % str(ref[top_k_idx]))

    ridx = np.argmax(rerr)
    print("max_relative_diff: %f (base val is %f, ref val is %f)" % (rerr[ridx], base[ridx], ref[ridx]))
    top_k_idx = np.argsort(rerr)[::-1][:k]
    print("top_k relative_diff (k = %d)" % k)
    print("  base val is %s" % str(base[top_k_idx]))
    print("  ref val is %s" % str(ref[top_k_idx]))


if __name__ == '__main__':
    args = parse()
    base = read_data(args.base)
    ref = read_data(args.ref)
    print(base.keys())
    for n, v in base.items():
        print('---------------- check %s ----------------' % n)
        check(v, ref[n])
