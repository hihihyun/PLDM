import numpy as np
import cv2
import skimage
from skimage import color
import matplotlib
import matplotlib.pyplot as plt
import math


def show(img, title):
    # 이미지를 화면에 표시하고 파일로 저장하는 함수
    plt.title(title)
    plt.imshow(img)
    plt.savefig(title+'.jpg')
    plt.show()


def color_compensation(img):
    # 색상 보정 함수
    img = np.double(img)
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    # 세 가지 색상 채널의 평균값 계산 (I¯r, I¯g, I¯b에 해당)
    Irm = np.mean(R, axis=0)
    Irm = np.mean(Irm)/256.0
    Igm = np.mean(G, axis=0)
    Igm = np.mean(Igm)/256.0
    Ibm = np.mean(B, axis=0)
    Ibm = np.mean(Ibm)/256.0
    a = 1
    Irc = R + a * (Igm-Irm)*(1-Irm)*G  # 빨간색 채널 보정
    Irc = np.array(Irc.reshape(G.shape), np.uint8)
    Ibc = B + a * (Igm-Ibm)*(1-Ibm)*G  # 파란색 채널 보정
    Ibc = np.array(Ibc.reshape(G.shape), np.uint8)

    G = np.array(G, np.uint8)
    img = cv2.merge([Irc, G, Ibc])
    show(img, "color_compensation")
    return img


def gray_world(img):
    # 그레이 월드(Gray World) 화이트 밸런스 함수
    dim = np.shape(img)[2]
    img = np.array(img, dtype='uint8')
    out = np.zeros(np.shape(img))
    avg = np.mean(np.mean(img))
    # 원본 Gray World 알고리즘과 약간의 수정이 있음
    for j in range(0, dim):
        m = np.sum(np.sum(img[:, :, j], axis=0), axis=0)
        n = np.size(img[:, :, j])
        scale = n/m
        g_weight = (avg*scale)
        out[:, :, j] = img[:, :, j]*g_weight
    out = np.array(out, dtype='uint8')
    show(out, "gray_world")
    return out


def gamma_correction(img, gamma=1.3):
    # 감마 보정 함수
    img = np.array(256*(img / 256) ** gamma, np.uint8)
    show(img, "gamma_correction")
    return img


def sharpen(img):
    # 이미지 선명화(샤프닝) 함수
    filter1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, filter1)
    show(img, "sharpen")
    return img


def gauss(shape=(3, 3), sigma=0.5):
    # 가우시안 필터 커널 생성 함수
    # 이미지 크기에 따라 shape를 적절히 조절할 수 있음
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def saliency_weight(img):
    # 현저성(Saliency) 가중치 맵 계산 함수
    kernel = np.array(gauss((3, 3), 1))
    gfrgb = cv2.filter2D(img, -1, kernel, cv2.BORDER_WRAP)
    lab = color.rgb2lab(gfrgb)
    l = np.double(lab[:, :, 0])
    a = np.double(lab[:, :, 1])
    b = np.double(lab[:, :, 2])
    lm = np.mean(np.mean(l))
    am = np.mean(np.mean(a))
    bm = np.mean(np.mean(b))
    w = np.square(l-lm) + np.square(a-am) + np.square((b-bm))
    return w


def laplacian_weight(img):
    # 라플라시안(Laplacian) 가중치 맵 계산 함수
    w = cv2.Laplacian(img, cv2.CV_64F)
    w = cv2.convertScaleAbs(w)
    return w


def saturation_weight(img, sigma=0.25, avg_=0.5):
    # 채도(Saturation) 가중치 맵 계산 함수
    w = np.exp(-(img-avg_)**2/(2*np.square(sigma)))
    return w


def weight_maps(img):
    # 라플라시안, 현저성, 채도 가중치 맵을 모두 계산하는 함수
    R = np.double(img[:, :, 0])/256
    WL = laplacian_weight(R)
    WS = saliency_weight(img)
    WSat = saturation_weight(R)
    return WL, WS, WSat


def iexpand(image):
    # 이미지를 확장하는 함수 (피라미드 업샘플링)
    h = np.array([1, 4, 6, 4, 1])/16
    filt = (h.T).dot(h)
    outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = cv2.filter2D(outimage, cv2.CV_64F, filt)
    return out


def ireduce(image):
    # 이미지를 축소하는 함수 (피라미드 다운샘플링)
    h = np.array([1, 4, 6, 4, 1])/16
    filt = (h.T).dot(h)
    outimage = cv2.filter2D(image, cv2.CV_64F, filt)
    out = outimage[::2, ::2]
    return out


def gaussian_pyramid(image, depth=5):
    # 가우시안 피라미드 생성 함수
    output = []
    output.append(image)
    tmp = image
    for i in range(0, depth):
        tmp = ireduce(tmp)
        output.append(tmp)
    return output


def lapl_pyramid(img_pyr):
    # 라플라시안 피라미드 생성 함수
    output = []
    k = len(img_pyr)
    for i in range(0, k-1):
        gu = img_pyr[i]
        egu = iexpand(img_pyr[i+1])
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(img_pyr.pop())
    return output


def collapse(img_pyr):
    # 피라미드 붕괴(합성) 함수
    for i in range(len(img_pyr)-1, 0, -1):
        lap = iexpand(img_pyr[i])
        lap_last_dim = img_pyr[i-1]
        if lap.shape[0] > lap_last_dim.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lap_last_dim.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lap_last_dim
    output = tmp
    return output


def split_rgb(img):
    # opencv의 색상 채널 순서는 BGR
    (blue, green, red) = cv2.split(img)
    return red, green, blue


def pyramid_op(img):
    # 색상 채널별로 가우시안 및 라플라시안 피라미드 생성
    (R, G, B) = split_rgb(img)
    R = gaussian_pyramid(R)
    G = gaussian_pyramid(G)
    B = gaussian_pyramid(B)
    R = lapl_pyramid(R)
    G = lapl_pyramid(G)
    B = lapl_pyramid(B)
    return R, G, B


# 이미지 파일 불러오기
img = cv2.imread('initial_image.jpg')
# 1. 색상 보정 수행
img = color_compensation(img)
# 2. Gray World 화이트 밸런스 적용
img = gray_world(img)

# 3. 감마 보정 (첫 번째 입력 이미지, input1)
img1 = gamma_correction(img)
# 4. 이미지 선명화 (두 번째 입력 이미지, Iinput2)
img2 = sharpen(img)

# 5. 가중치 맵(weight maps) 계산
(WL1, WS1, WSat1) = weight_maps(img1)
(WL2, WS2, WSat2) = weight_maps(img2)

# 6. 정규화된 가중치 맵
W1 = (WL1 + WS1 + WSat1)/(WL1 + WS1 + WSat1 + WL2 + WS2 + WSat2)
W2 = (WL2 + WS2 + WSat2)/(WL1 + WS1 + WSat1 + WL2 + WS2 + WSat2)

# 7. 다중 스케일(multi-scale) 융합
W1 = gaussian_pyramid(W1, 5)
W2 = gaussian_pyramid(W2, 5)

r1, g1, b1 = pyramid_op(img1)
r2, g2, b2 = pyramid_op(img2)

R = np.array(W1) * r1 + np.array(W2) * r2
G = np.array(W1) * g1 + np.array(W2) * g2
B = np.array(W1) * b1 + np.array(W2) * b2

R = collapse(R)
G = collapse(G)
B = collapse(B)

# 8. 최종 결과 이미지 생성
R[R < 0] = 0
R[R > 255] = 255
R = np.array(R, np.uint8)

G[G < 0] = 0
G[G > 255] = 255
G = np.array(G, np.uint8)

B[B < 0] = 0
B[B > 255] = 255
B = np.array(B, np.uint8)

result = cv2.merge([B, G, R])  # OpenCV의 색상 채널 순서는 BGR
show(result, "result")