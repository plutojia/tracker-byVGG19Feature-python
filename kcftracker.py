import numpy as np
import cv2
import CNNfeature

import fhog
import sys
PY3 = sys.version_info >= (3,)

if PY3:
    xrange = range


# ffttools
def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags=(
        (cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!


def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]


def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))

    assert (img.ndim == 2)
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_



# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2] > limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if (rect[1] < limit[1]):
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if (rect[2] < 0):
        rect[2] = 0
    if (rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


# KCF tracker
class KCFTracker:
    def __init__(self, multiscale=False):
        self.lambdar = 0.0001  # regularization
        self.padding = 2.5  # extra area surrounding the target
        self.output_sigma_factor = 0.125  # bandwidth of gaussian target
        self.interp_factor = 0.01  # linear interpolation factor for adaptation  0.075
        self.sigma = 0.2  # gaussian kernel bandwidth

        if (multiscale):
            self.template_size = 224  # template size
            self.scale_step = 1.05  # scale step for multi-scale estimation
            self.scale_weight = 0.96  # to downweight detection scores of other scales for added stability
        else:
            self.template_size = 224
            self.scale_step = 1

        self._tmpl_sz = [224, 224]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.  # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.vgg=CNNfeature.CNNfeature()
        self.convlist=("conv3_4","conv4_4","conv5_4")                      #要获取的VGG卷积层
        self.nweights=[1.0, 0.5, 0.25]                    #加权融合权重
        self.numLayers = len(self.convlist)

        self.cell_size = 4  # CNN cell size

    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0][0], 0:self.size_patch[0][1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[0][1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0][0] - 1)))
        hann2d = hann2t * hann1t

        hann1d = hann2d.reshape(self.size_patch[0][0] * self.size_patch[0][1])
        self.hann = [np.zeros((self.size_patch[i][2], 1), np.float32) + hann1d for i in range(self.numLayers)]    #相当于把1D汉宁窗复制成多个通道

        self.hann = [self.hann[i].astype(np.float32) for i in range(self.numLayers)]

    def createGaussianPeak(self, sizey, sizex):                                      #构建响应图，只在初始化用
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def gaussianCorrelation(self, x1, x2):
        c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
        for i in xrange(self.size_patch[2]):
            x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
            x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
            caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)            # 'conjB=' is necessary!在做乘法之前取第二个输入数组的共轭.
            caux = real(fftd(caux, True))
            # caux = rearrange(caux)
            c += caux
        c = rearrange(c)

        if (x1.ndim == 3 and x2.ndim == 3):
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif (x1.ndim == 2 and x2.ndim == 2):
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d
    def linear_correlation(self,x1,x2,index):
        size_patch=self.size_patch[index]
        c = np.zeros((size_patch[0], size_patch[1]), np.float32)
        for i in xrange(size_patch[2]):
            x1aux = x1[i, :].reshape((size_patch[0], size_patch[1]))
            x2aux = x2[i, :].reshape((size_patch[0], size_patch[1]))
            caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)  # 'conjB=' is necessary!在做乘法之前取第二个输入数组的共轭.
            caux = real(fftd(caux, True))
            # caux = rearrange(caux)
            c += caux
        c = rearrange(c)

        return c

    def getFeatures(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float         self._roi [229.28751704333027, 193.2002430105049, 167.0, 249.0]
        cy = self._roi[1] + self._roi[3] / 2  # float

        if (inithann):                                              #只有初始化时执行，算出模板大小和PATCH大小
            padded_w = self._roi[2] * self.padding                      #padded_w\h 是应在原图中裁剪下多大窗口
            padded_h = self._roi[3] * self.padding

            if (padded_w >= padded_h):
                self._scale = padded_w / float(self.template_size)    #把最大的边缩小到224，_scale是缩小比例
                padded_h=padded_w
            else:
                self._scale = padded_h / float(self.template_size)
                padded_w=padded_h

            # _tmpl_sz是输入到CNN来获取特征的图像的大小也是裁剪下的PATCH被缩放到的大小,为224,224
            self._tmpl_sz[0] = 224 // (2 * self.cell_size) * 2 * self.cell_size
            self._tmpl_sz[1] = 224 // (2 * self.cell_size) * 2 * self.cell_size

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])        #选取从原图中扣下的图片位置大小
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)                       #z是当前帧被裁剪下的搜索区域
        if (z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]):          #缩小到96
            z = cv2.resize(z, tuple(self._tmpl_sz))
        #获取CNNfeature
        feature_list=self.vgg.getfeature(z,self.convlist)
        self.size_patch = [[feature_list[0].shape[0], feature_list[0].shape[1],
                            feature_list[i].shape[2]] for i in range(self.numLayers)] # size_patch为列表，保存裁剪下来的特征图的【长，宽，1】

        feature_list=self.vgg.resize_list(feature_list,(self.size_patch[0][0], self.size_patch[0][1]))
        FeaturesMap = self.vgg.normalize_list(feature_list)                                # 从此FeatureMap从-0.5到0.5

        FeaturesMap=[FeaturesMap[i].reshape((self.size_patch[i][0] * self.size_patch[i][1],
                                               self.size_patch[i][2])).T for i in range(self.numLayers)]

        if (inithann):
            self.createHanningMats()  # createHanningMats need size_patch
        FeaturesMap = [self.hann[i] * FeaturesMap[i] for i in range(self.numLayers)]            #加汉宁（余弦）窗减少频谱泄露
        return FeaturesMap

    def detect(self, z, x):                                             #z是_tmpl即特征的平均，x是当前帧的特征
        #k = self.gaussianCorrelation(x, z)
        k = [self.linear_correlation(x[i], z[i],i) for i in range(self.numLayers)]
        res = [self.nweights[i]*real(fftd(complexMultiplication(self._alphaf[i], fftd(k[i])), True))
                                for i in range(self.numLayers)]  #得到响应图
        #res=np.sum(res, 0)/np.sum(self.nweights)
        res = np.sum(res, 0)
        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int                  #pv:响应最大值 pi:相应最大点的索引数组
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]          #得到响应最大的点索引的float表示

        if (pi[0] > 0 and pi[0] < res.shape[1] - 1):
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])  #使用幅值做差来定位峰值的位置
        if (pi[1] > 0 and pi[1] < res.shape[0] - 1):
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.
                                                                                        #得出偏离采样中心的位移
        return p, pv                                                                    #返回偏离采样中心的位移和峰值

    def train(self, x, train_interp_factor):
        # k = self.gaussianCorrelation(x, x)
        k = [self.linear_correlation(x[i], x[i],i) for i in range(self.numLayers)]
        alphaf = [complexDivision(self._prob, fftd(k[i]) + self.lambdar) for i in range(self.numLayers)] #alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
                                                                                        #_prob是初始化时的高斯响应图，相当于y

        # _tmpl是截取的特征的加权平均
        self._tmpl = [(1 - train_interp_factor) * self._tmpl[i] + train_interp_factor * x[i] for i in range(self.numLayers)]
        #_alphaf是频域中相关滤波模板的加权平均
        self._alphaf = [(1 - train_interp_factor) * self._alphaf[i] + train_interp_factor * alphaf[i] for i in range(self.numLayers)]

    def init(self, roi, image):
        self._roi = list(map(float,roi))
        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.getFeatures(image, 1)                                                 #_tmpl是截取的特征的加权平均
        self._prob = self.createGaussianPeak(self.size_patch[0][0], self.size_patch[0][1])            #_prob是初始化时的高斯响应图
        self._alphaf = [np.zeros((self.size_patch[0][0], self.size_patch[0][1], 2), np.float32) for i in range(self.numLayers)]    #_alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        self.train(self._tmpl, 1.0)
        return True

    def update(self, image):
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1                #修正边界
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.                                                   #尺度框中心
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        if (self.scale_step != 1):
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

            if (self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif (self.scale_weight * new_peak_value2 > peak_value):
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step

        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale         #loc是中心相对移动量
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 1
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 1
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self._roi
