# import cv2
# import numpy as np
# import sys

# #实现图像的极坐标的转换 center代表及坐标变换中心‘;r是一个二元元组，代表最大与最小的距离；theta代表角度范围
# #rstep代表步长； thetastap代表角度的变化步长
# def polar(image,center,r,theta=(0,360),rstep=0.5,thetastep=360.0/(180*4)):
#     #得到距离的最小值、最大值
#     minr,maxr=r
#     #角度的最小范围
#     mintheta,maxtheta=theta
#     #输出图像的高、宽 O:指定形状类型的数组float64
#     H=int((maxr-minr)/rstep)+1
#     W=int((maxtheta-mintheta)/thetastep)+1
#     O=125*np.ones((H,W),image.dtype)
#     #极坐标转换  利用tile函数实现W*1铺成的r个矩阵 并对生成的矩阵进行转置
#     r=np.linspace(minr,maxr,H)
#     r=np.tile(r,(W,1))
#     r=np.transpose(r)
#     theta=np.linspace(mintheta,maxtheta,W)
#     theta=np.tile(theta,(H,1))
#     x,y=cv2.polarToCart(r,theta,angleInDegrees=True)
#     #最近插值法
#     for i in range(H):
#         for j in range(W):
#             px=int(round(x[i][j])+cx)
#             py=int(round(y[i][j])+cy)
#             if((px>=0 and px<=w-1) and (py>=0 and py<=h-1)):
#                 O[i][j]=image[py][px]

#     return O

# if __name__=="__main__":
img_path = r"C:\qianlinjun\_0aXSFFHkHwOGrp1pSf1dQ(22.278969,114.191412).jpg"
#     img = cv2.imread(r"C:\qianlinjun\_0aXSFFHkHwOGrp1pSf1dQ(22.278969,114.191412).jpg", cv2.IMREAD_GRAYSCALE)
#     # 传入的图像宽：600  高：400
#     h, w = img.shape[:2]
#     print("h:%s w:%s"%(h,w))
#     # 极坐标的变换中心（300，200）
#     cx, cy = h//2, w//2#300, 200
#     # 圆的半径为10 颜色：灰 最小位数3
#     cv2.circle(img, (int(cx), int(cy)), 10, (255, 0, 0, 0), 3)
#     L = polar(img, (cx, cy), (0,1000)) #(100, 350)
#     # 旋转
#     L = cv2.flip(L, 0)
#     # 显示与输出
#     cv2.imshow('img', img)
#     cv2.imshow('O', L)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#!/usr/bin/env python

import numpy as np
import cv2
# main window
WINDOW_NAME = "carToPol"
cv2.namedWindow(WINDOW_NAME)
# read input
src  = cv2.imread(img_path)
# get dimenions and compute half (for center/radius)
dims = src.shape[:-1]
cols,rows = dims
half = cols // 2

def carToPol(src,center,maxRadius,interpolation,rotate90=True):
    # rotate 90 degrees (cv2.warpAffine seems to mess up a column)
    if(rotate90):
        src = np.rot90(src)
    # cartesian to polar (WARP_INVERSE_MAP)
    return cv2.linearPolar(src,(centerX,centerY),maxRadius,interpolation + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

# interpolation names: just for debugging
interpolations = ["NEAREST","LINEAR","CUBIC","AREA","LANCZOS4"]

# slider callbacks: just for debugging
def printCenterX(x):
    print("center",x)
def printCenterY(x):
    print("centerY",x)
def printMaxRadius(x):
    print("maxRadius",x)
def printInterpolation(x):
    global interpolations
    print("interpolation",interpolations[x])
# create sliders
cv2.createTrackbar("centerX"  ,WINDOW_NAME,half,cols,printCenterX)
cv2.createTrackbar("centerY"  ,WINDOW_NAME,half,cols,printCenterY)
cv2.createTrackbar("maxRadius",WINDOW_NAME,half,cols * 4,printMaxRadius)
cv2.createTrackbar("interpolation",WINDOW_NAME,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4,printInterpolation)

# continously process for quick feedback
while 1:
    # exit on ESC key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # read slider values
    centerX   = cv2.getTrackbarPos("centerX",WINDOW_NAME)
    centerY   = cv2.getTrackbarPos("centerY",WINDOW_NAME)
    maxRadius = cv2.getTrackbarPos("maxRadius",WINDOW_NAME)
    interpolation = cv2.getTrackbarPos("interpolation",WINDOW_NAME)

    dst = carToPol(src,(centerX,centerY),maxRadius,interpolation,True)
    # show result
    cv2.imshow(WINDOW_NAME,dst)
    # save image with 's' key
    if k == ord('s'):
        cv2.imwrite("output.png",dst)

# exit
cv2.destroyAllWindows()