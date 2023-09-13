import numpy as np
import argparse

def read_data(filename):
    sep = "-SEP-"
    dic = {}
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            info = line.split(sep)
            name = info[0]
            data = np.fromstring(info[1], sep=' ')
            dic[name] = data
            line = f.readline().strip()
    return dic

def parse():
    parser = argparse.ArgumentParser('')
    parser.add_argument('base', type=str, help='')
    parser.add_argument('ref', type=str, help='')
    return parser.parse_args()

def check(base, ref):
    assert base.shape == ref.shape, 'base.shape is %s but ref.shape is %s' % (str(base.shape), str(ref.shape))
    print('shape is ', base.shape)
    diff = base - ref
    abs_diff = np.abs(diff)
    rerr = abs_diff / (base + 1e-9)

    print('avg abs_diff: %f ' % np.mean(abs_diff))
    print('avg relative_diff: %f ' % np.mean(rerr))

    idx = np.argmax(abs_diff)
    print("max_diff: %f (base val is %f, ref val is %f)" % (abs_diff[idx], base[idx], ref[idx]))

    k = min(10, abs_diff.size)
    top_k_idx = np.argsort(abs_diff)[::-1][:k]
    print("top_k max_diff (k = %d):" % k)
    print("base val is %s" % str(base[top_k_idx]))
    print("ref val is %s" % str(ref[top_k_idx]))

    ridx = np.argmax(rerr)
    print("max_relative_diff: %f (base val is %f, ref val is %f)" % (rerr[ridx], base[ridx], ref[ridx]))
    top_k_idx = np.argsort(rerr)[::-1][:k]
    print("top_k relative_diff (k = %d)" % k)
    print("base val is %s" % str(base[top_k_idx]))
    print("ref val is %s" % str(ref[top_k_idx]))


if __name__ == '__main__':
    args = parse()
    base = read_data(args.base)
    ref = read_data(args.ref)
    print(base.keys())
    for n, v in base.items():
        print('---------------- check %s ----------------' % n)
        check(v, ref[n])
