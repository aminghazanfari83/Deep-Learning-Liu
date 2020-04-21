"""A reader for the MNIST data format.

Information about the MNIST data format is available at
http://yann.lecun.com/exdb/mnist/

"""

import gzip
import struct


def read_images(input):
    magic = struct.unpack('>BBBB', input.read(4))
    assert magic[0] == 0 and magic[1] == 0 and magic[2] == 8 and magic[3] == 3
    n_imgs = struct.unpack('>i', input.read(4))[0]
    n_rows = struct.unpack('>i', input.read(4))[0]
    n_cols = struct.unpack('>i', input.read(4))[0]
    n_pxls = n_rows * n_cols
    for i in range(n_imgs):
        yield struct.unpack('>%dB' % n_pxls, input.read(n_pxls))


def read_labels(input):
    magic = struct.unpack('>BBBB', input.read(4))
    assert magic[0] == 0 and magic[1] == 0 and magic[2] == 8 and magic[3] == 1
    n_labs = struct.unpack('>i', input.read(4))[0]
    for i in range(n_labs):
        yield struct.unpack('>B', input.read(1))[0]


def read_data(images_file, labels_file):
    with gzip.open(images_file) as fi, gzip.open(labels_file) as fl:
        yield from zip(read_images(fi), read_labels(fl))


def read_training_data():
    yield from read_data('train-images-idx3-ubyte.gz',
                         'train-labels-idx1-ubyte.gz')


def read_test_data():
    yield from read_data('t10k-images-idx3-ubyte.gz',
                         't10k-labels-idx1-ubyte.gz')


if __name__ == "__main__":
    print("Training data: %d items" % len(list(read_training_data())))
    print("Test data: %d items" % len(list(read_test_data())))
