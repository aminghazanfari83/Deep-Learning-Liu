import bz2

def _read_tagged_sentences(fp):
    result = []
    for line in fp:
        line = line.rstrip()
        if len(line) == 0:
            yield result
            result = []
        else:
            columns = line.split()
            assert len(columns) == 2
            result.append(tuple(columns))

def read_tagged_sentences(filename):
    with bz2.open(filename) as fp:
        yield from _read_tagged_sentences(fp)
            
def read_training_data():
    yield from read_tagged_sentences('train.txt.bz2')

def read_development_data():
    yield from read_tagged_sentences('dev.txt.bz2')

def read_test_data():
    yield from read_tagged_sentences('test.txt.bz2')
