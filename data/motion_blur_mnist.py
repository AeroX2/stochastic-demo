def data(path='mnist-with-motion-blur.gz'):
    from keras.utils.data_utils import get_file
    path = get_file(path,
                    origin='https://csc.lsu.edu/~saikat/n-mnist/data/mnist-with-motion-blur.gz',
                    file_hash='70a7ccabd53e5b50ba8e7b9358cefadea8ab499a4738eda67caabe66289cb17e')

    import os
    base_path = os.path.dirname(path)
    mat_path = base_path+'/mnist-with-motion-blur.mat'
    if (not os.path.isfile(mat_path)):
        import tarfile
        tar = tarfile.open(path)
        tar.extractall(path=base_path)
        tar.close()

    import scipy.io
    f = scipy.io.loadmat(mat_path)
    x_train, y_train = f['train_x'], f['train_y']
    x_test, y_test = f['test_x'], f['test_y']

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = y_train.nonzero()[1]
    y_test = y_test.nonzero()[1]

    from keras.utils import to_categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
