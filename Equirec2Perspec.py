# -*- coding:utf-8 -*-
import os
import sys
import cv2
import numpy as np
import time
class Equirectangular:
    def __init__(self):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
        self.paramdict = {}

    def set_path(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
    
    def set_img(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._img = img
        if type(self._img) is np.ndarray:
            [self._height, self._width, _] = self._img.shape
        else:
            print('please input numpy array')

    
    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS = 128):
        #f, h, v, 640, 480
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        param_present=False
        try:
            self.paramdict['{}-{}-{}-{}-{}'.format(FOV, THETA, PHI, height, width)]  
        except KeyError:
            param_present = False
        else:
            param_present = True

        if param_present is False: 
            equ_h = self._height
            equ_w = self._width
            equ_cx = (equ_w - 1) / 2.0
            equ_cy = (equ_h - 1) / 2.0

            wFOV = FOV#50
            hFOV = float(height) / width * wFOV
            # 640 480 /2
            c_x = (width - 1) / 2.0
            c_y = (height - 1) / 2.0

            wangle = (180 - wFOV) / 2.0
            # https://arxiv.org/pdf/1811.05304.pdf
            # sinx/consx = sinx/

            # 成像平面宽度
            w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
            w_interval = w_len / (width - 1)
            time0 = time.clock()
            # print('time is {}\n'.format(time4-time3))
            hangle = (180 - hFOV) / 2.0
            
            h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
            h_interval = h_len / (height - 1)
            # RADIUS离成像平面距离
            x_map = np.zeros([height, width], np.float32) + RADIUS
            y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
            z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
            D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
            xyz = np.zeros([height, width, 3], np.float)
            # RADIUS / D cons
            xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
            xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
            xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
            
            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
            z_axis = np.array([0.0, 0.0, 1.0], np.float32)
            [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
            [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))
            
            xyz = xyz.reshape([height * width, 3]).T
            xyz = np.dot(R1, xyz)
            xyz = np.dot(R2, xyz).T
            time1 = time.clock()
            # print('time1 is {}'.format(time1-time0))
            # print(len(xyz[:, 2]))
            lat = np.arcsin(xyz[:, 2] / RADIUS)
            # print(lat)
            lon = np.zeros([height * width], np.float)
            theta = np.arctan(xyz[:, 1] / xyz[:, 0])
            time2 = time.clock()
            # print('time2 is {}'.format(time2-time1))
            idx1 = xyz[:, 0] > 0
            idx2 = xyz[:, 1] > 0
            
            idx3 = ((1 - idx1) * idx2).astype(np.bool)
            idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
            
            lon[idx1] = theta[idx1]
            lon[idx3] = theta[idx3] + np.pi
            lon[idx4] = theta[idx4] - np.pi

            lon = lon.reshape([height, width]) / np.pi * 180
            lat = -lat.reshape([height, width]) / np.pi * 180
            lon = lon / 180 * equ_cx + equ_cx
            lat = lat / 90 * equ_cy + equ_cy
            #for x in range(width):
            #    for y in range(height):
            #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
            #return self._img 
            self.paramdict['{}-{}-{}-{}-{}'.format(FOV, THETA, PHI, height, width)]   = [lon, lat]
        else:
            lon, lat=self.paramdict['{}-{}-{}-{}-{}'.format(FOV, THETA, PHI, height, width)]
            # print(self.paramdict.keys()) 
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        time3 = time.clock()
        # print('time3 is {}\n'.format(time3-time2))
        return persp




if __name__ == '__main__':
    # img_path = sys.stdin.readline().strip()
    img_path = r"C:\qianlinjun\projection\wecnwr\_0aXSFFHkHwOGrp1pSf1dQ(22.278969,114.191412)\_0aXSFFHkHwOGrp1pSf1dQ(22.278969,114.191412).jpg"
    img_name = img_path.replace(".json", "").split("\\")[-1]

    equ = Equirectangular()    # Load equirectangular image
    equ.set_path(img_path)

    #
    # FOV unit is degree 
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension 
    fov = 60
    theta = 0 #180
    phi=0
    height = 720
    width = 1080
    img = equ.GetPerspective(fov, theta, phi, height, width) # Specify parameters(FOV, theta, phi, height, width)
    img_save_path = r"C:\qianlinjun\projection\save_dir\pos{}_fov{}_theta{}_phi{}.png".format(img_name, fov, theta, phi)
    cv2.imwrite(img_save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY),90])
