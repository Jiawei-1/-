import cv2 
import numpy as np
np.set_printoptions(threshold=np.inf) 

path='C:\\Users\\93115\\Desktop\\mat\\3-6canny1.jpg' # 保存边缘提取后的图片位置

def DistanceCalculate(path): #测量上下边界的宽度 传入保存边缘的路径 输出宽度值
    # 读取边缘，并灰度化
    img=cv2.imread(path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 选取要测量的部分
    cv2.namedWindow('Canny Edge', cv2.WINDOW_NORMAL) 
    bbox = cv2.selectROI('Canny Edge',img ,False)
    cut = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\cut.jpg', cut)#保存切割后的图片
    
    img=cut

    distance = 600
    h, w = img.shape
    line = int(h/2)     #取中间轴
    conture_up=np.zeros(w)       #上方轮廓
    conture_down=np.zeros(w)     #下方轮廓
    for j in range(w):          #找到上部的轮廓
        for i in range(line):
            pv=img[i,j]
            if pv>0:
                conture_up[j] = i
                continue
    for j in range(w):          #找到下部的轮廓
        for i in range(line,h):
            pv=img[i,j]
            if pv>0:
                conture_down[j] = i
                continue
    for j in range(w):             #计算
        for i in range(w):
            d2=(i-j)*(i-j)+(conture_down[j]-conture_up[i])*(conture_down[j]-conture_up[i])
            if np.sqrt(d2)<distance:
                distance=np.sqrt(d2)
    print(distance)
    return distance

# class UseCv:
#     def __init__(self):
#         self.path = 'C:\\Users\\93115\\Desktop\\mat\\3-6canny1.jpg'

#     def cut(self):
#         img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
#         cv2.namedWindow('Canny Edge', cv2.WINDOW_NORMAL) 
#         bbox = cv2.selectROI('Canny Edge',img ,False)
#         cut = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
#         cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\cut.jpg', cut)

# if __name__ == '__main__':
#     UseCv().cut()

# img2 = cv2.imread('C:\\Users\\93115\\Desktop\\mat\\cut.jpg')
DistanceCalculate(path)

