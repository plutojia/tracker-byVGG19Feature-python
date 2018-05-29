import numpy as np
import tensorflow as tf
import cv2
import sys
import utils
import vgg19
import matplotlib.pyplot  as plt
import skimage.io

vgg_height=224
vgg_width=224

class CNNfeature:
    def __init__(self):
        self.vgg = vgg19.Vgg19()
        self.sess = tf.Session()
        self.vgg_placeholder = tf.placeholder("float", [1, 224, 224, 3])
        self.vgg.build(self.vgg_placeholder,0)
        self.featrue_dict={"conv1_1":self.vgg.conv1_1,"conv1_2":self.vgg.conv1_2,"conv2_1":self.vgg.conv2_1,
                           "conv2_2": self.vgg.conv2_2,"conv3_1":self.vgg.conv3_1,"conv3_2":self.vgg.conv3_2,
                           "conv3_3": self.vgg.conv3_3, "conv3_4": self.vgg.conv3_4, "conv4_1": self.vgg.conv4_1,
                           "conv4_2": self.vgg.conv4_2, "conv4_3": self.vgg.conv4_3, "conv4_4": self.vgg.conv4_4,
                           "conv5_1": self.vgg.conv5_1, "conv5_2": self.vgg.conv5_2, "conv5_3": self.vgg.conv5_3,
                           "conv5_4": self.vgg.conv5_4
                           }
        print("CNNfeature inited!")

    def getfeature(self,img,str_list):
        resized_img = cv2.resize(img, (224, 224))
        batch = resized_img.reshape((1, 224, 224, 3))
        batch = batch / 255.0
        conv_list=list(self.featrue_dict[conv] for conv in str_list)
        feature_list = self.sess.run(conv_list, feed_dict={self.vgg_placeholder: batch})
        feature_list=[f[0] for f in feature_list]
        return feature_list

    def normalize(self,conv):
        n_conv = (conv / conv.max()-0.5)
        return n_conv

    def normalize_list(self, conv_list):
        return [conv / conv.max()-0.5 for conv in conv_list]

    def resize_list(self,conv_list,shape):
        return [cv2.resize(conv,shape,interpolation=cv2.INTER_LINEAR) for conv in conv_list]

if __name__ == "__main__":
    img1 = cv2.imread("test_data/BlurFace/BlurFace/img/0001.jpg")
    vgg=CNNfeature()
    l=("conv3_2",)
    f=vgg.getfeature(img1,l)
    conv3_2=f[0][0]
    print(conv3_2.shape)
    feature_img3_2 = np.dstack((conv3_2[:, :, 0], conv3_2[:, :, 40], conv3_2[:, :, 80]))
    feature_img3_2 = (feature_img3_2 / feature_img3_2.max() * 255).astype("uint8")
    skimage.io.imshow(feature_img3_2)
    plt.show()

    cv2.imshow("feature_img",feature_img3_2)
    cv2.imshow("con1",conv3_2[:,:,0])
    cv2.imshow("con2",conv3_2[:,:,20])
    cv2.waitKey(0)
    sys.exit(0)