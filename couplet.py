
from model import Model

m = Model(
        '../couplet/train/in.txt',
        '../couplet/train/out.txt',
        '../couplet/test/in.txt',
        '../couplet/test/out.txt',
        '../couplet/vocabs',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir='../models/tf-lib/temp',
        restore_model=False)

m.train(2000,start=0)
