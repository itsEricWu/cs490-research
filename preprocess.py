import numpy as np
import cv2
from matplotlib import pyplot as plt


class Preprocess():
    def im2double(im):
        min_val = np.min(im.ravel())
        max_val = np.max(im.ravel())
        out = (im.astype('float') - min_val) / (max_val - min_val)
        return out

    def preprocess_image(img_filename, v, alpha):
        """
        processs the image with v matrix and noise
        input:
        img_filename: the path of an image in ppm format
        v matrix: 3 by 3 numpy array
        alpha: coefficient used for generating gaussian noise
        """
        # Comments on v matrix:
        # For some global color mixing matrix v, figure out what the input to the
        # projector should be, assuming zero illumination other than the projector.
        # Note : Not all scenes are possible for all v matrices. For some matrices
        # some scenes are not possible at all.

        img = cv2.imread(img_filename)  # numpy array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Captured Image - original #
        y = Preprocess.im2double(img)  # Convert to normalized floating point

        ss = y.shape  # (480,640,3)
        ylong = np.zeros((ss[0] * ss[1], 3))  # (307200,3)

        y1 = y[:, :, 0]  # (480,640)
        ylong[:, 0] = y1.flatten()
        y2 = y[:, :, 1]  # (480,640)
        ylong[:, 1] = y2.flatten()
        y3 = y[:, :, 2]  # (480,640)
        ylong[:, 2] = y3.flatten()

        xlong = np.transpose(np.matmul(np.linalg.pinv(v), np.transpose(ylong)))
        xlong[xlong > 1] = 1
        xlong[xlong < 0] = 0

        # Projector input - original #
        x = np.zeros(y.shape)
        x[:, :, 0] = xlong[:, 0].reshape(ss[0], ss[1])
        x[:, :, 1] = xlong[:, 1].reshape(ss[0], ss[1])
        x[:, :, 2] = xlong[:, 2].reshape(ss[0], ss[1])

        # Now you can get any perturbed image y = v(x+\delta x)

        xlong_new = xlong + alpha * \
            np.random.rand(xlong.shape[0], xlong.shape[1])
        # Projector input - Attacked #
        x_new = np.zeros(x.shape)
        x_new[:, :, 0] = xlong_new[:, 0].reshape(ss[0], ss[1])
        x_new[:, :, 1] = xlong_new[:, 1].reshape(ss[0], ss[1])
        x_new[:, :, 2] = xlong_new[:, 2].reshape(ss[0], ss[1])

        ylong_new = np.transpose(np.matmul(v, np.transpose(xlong_new)))
        # Captured Image - Attacked #
        y_new = np.zeros(y.shape)
        y_new[:, :, 0] = ylong_new[:, 0].reshape(ss[0], ss[1])
        y_new[:, :, 1] = ylong_new[:, 1].reshape(ss[0], ss[1])
        y_new[:, :, 2] = ylong_new[:, 2].reshape(ss[0], ss[1])

        # Captured original, Projector original, Captured attracked, Projector attacked
        return y, x, y_new, x_new

    # , original_filename, actual_label, v_number, n_number):
    def show_np_array_as_jpg(matrix, number):
        # filename = "/home/zhan3447/CS490_DSC/jpg/{actual_label}/v{v_number}/n{n_number}/{original_filename}"
        filename = f"show{number}.jpg"
        # cv2.imwrite(filename, matrix) # does not work
        plt.imshow(matrix)
        plt.savefig(filename)
