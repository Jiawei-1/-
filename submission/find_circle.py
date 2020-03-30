import  cv2

# def Find_Circle(path):
#载入并显示图片
img=cv2.imread('C:\\Users\\93115\\Desktop\\mat\\circle2.jpg')
cv2.imshow('img',img)
#灰度化
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
#输出图像大小，方便根据图像大小调节minRadius和maxRadius
print(img.shape)
#霍夫变换圆检测
circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=5,maxRadius=300)
#输出返回值，方便查看类型
print(circles)
#输出检测到圆的个数
print(len(circles[0]))

print('-------------我是条分割线-----------------')
#根据检测到圆的信息，画出每一个圆
for circle in circles[0]:
    #圆的基本信息
    print(circle[2])
    #坐标行列
    x=int(circle[0])
    y=int(circle[1])
    #半径
    r=int(circle[2])
    #在原图用指定颜色标记出圆的位置
    img=cv2.circle(img,(x,y),r,(0,0,255),-1)
#显示新图像
cv2.imshow('res',img)

#按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()



# # from PIL import Image
# import cv2
# # import matplotlib.pyplot as plt
# import numpy as np

# img = cv2.imread('C:\\Users\\93115\\Desktop\\mat\\circle.jpg')
# GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# GrayImage= cv2.medianBlur(GrayImage,5)
# ret,th1 = cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,  
#                     cv2.THRESH_BINARY,3,5)  
# th3 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
#                     cv2.THRESH_BINARY,3,5)


# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(th2,kernel,iterations=1)
# dilation = cv2.dilate(erosion,kernel,iterations=1)

# imgray=cv2.Canny(erosion,30,100)

# circles = cv2.HoughCircles(imgray,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=20,maxRadius=40)

# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
# print(len(circles[0,:]))

# import sys
# import cv2
# import numpy
# from scipy.ndimage import label
# # Application entry point
# #img = cv2.imread("02_adj_grey.jpg")
# img = cv2.imread("C:\\Users\\93115\\Desktop\\mat\\circle2.jpg")
# # Pre-processing.
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# cv2.imwrite("SO_0_gray.png", img_gray)
# #_, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
# _, img_bin = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY)
# cv2.imwrite("SO_1_threshold.png", img_bin)
# #blur = cv2.GaussianBlur(img,(5,5),0)
# img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, numpy.ones((3, 3), dtype=int))
# cv2.imwrite("SO_2_img_bin_morphoEx.png", img_bin)
# border = img_bin - cv2.erode(img_bin, None)
# cv2.imwrite("SO_3_border.png", border)
# circles = cv2.HoughCircles(border,cv2.HOUGH_GRADIENT,50,80, param1=80,param2=40,minRadius=10,maxRadius=150)
# # print circles
# cimg = img
# for i in circles[0,:]:
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#     cv2.putText(cimg,str(i[0])+str(',')+str(i[1]), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
#     cv2.imwrite("SO_8_cimg.png", cimg)
#     # cv2.imshow('detected circles',img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()