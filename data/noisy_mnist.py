def data(path='mnist-with-awgn.gz'):
    from keras.utils.data_utils import get_file
    path = get_file(path,
                    origin='https://csc.lsu.edu/~saikat/n-mnist/data/mnist-with-awgn.gz',
                    file_hash='f33c8f78534c1e4e0173744183e68f72a360bc28eeaf4f43f07d6d4c4ddb635b')

    import os
    base_path = os.path.dirname(path)
    mat_path = base_path+'/mnist-with-awgn.mat'
    if (not os.path.isfile(mat_path)):
        import tarfile
        tar = tarfile.open(path)
        tar.extractall(path=base_path)
        tar.close()

    import scipy.io
    f = scipy.io.loadmat(mat_path)
    x_train, y_train = f['train_x'], f['train_y']
    x_test, y_test = f['test_x'], f['test_y']

    y_train = y_train.nonzero()[1]
    y_test = y_test.nonzero()[1]

    from keras.utils import to_categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
