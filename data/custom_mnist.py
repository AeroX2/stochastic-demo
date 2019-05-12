from PIL import Image
import numpy as np

def data():
    img = Image.open('./images/dataset_low_1.png').convert('L')
    img2 = Image.open('./images/dataset_high_1.png').convert('L')

    x_test_1 = np.array(img)
    x_test_2 = np.array(img2)
    x_test = np.concatenate((x_test_1, x_test_2), 1)

    img.close()
    img2.close()

    dataset_size = x_test.shape[1]//28 
    x_test = np.hsplit(x_test, dataset_size)
    x_test = np.reshape(x_test, (dataset_size, 28*28))
    x_test = x_test.astype('float32')
    x_test /= 255
    x_test = 1.0-x_test

    #from keras.utils import to_categorical
    y_test = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
    #y_test = to_categorical(y_test, 10)

    return (None, None), (x_test, y_test)


