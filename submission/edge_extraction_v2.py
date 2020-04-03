import cv2
import numpy as np
def edge(filePath,dirPath):
    
    img = cv2.imread(filePath,0)

    # 腐蚀
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    # cv2.imwrite(dirPath + "erosion.jpg",erosion)# 保存腐蚀后的图片

    # 膨胀
    dilate = cv2.dilate(erosion,kernel,iterations = 1)
    # cv2.imwrite(dirPath + "3-6dilate.jpg",dilate)

    #  边缘提取 并保存图片
    canny1=cv2.Canny(dilate,150,200)
    # cv2.imwrite(dirPath + "3-6canny.jpg",canny1)

    #  去除孤立点
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(canny1)

    i=0
    for istat in stats:
        if istat[4]<120:
            if istat[3]>istat[4]:
                r=istat[3]
            else:r=istat[4]
            cv2.rectangle(canny1,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  # 26
        i=i+1

    #  保存去除孤立点后的边缘提取图片 并展示
    cv2.imwrite(dirPath, canny1)   
    return