import cv2
import numpy as np
filename="C:\\Users\\93115\\Desktop\\mat\\3-28test.jpg" #原图的路径
img = cv2.imread(filename,0)
print(np.shape(img))
# 腐蚀
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\3-27test.jpg',erosion)# 保存腐蚀后的图片
# 膨胀
dilate = cv2.dilate(erosion,kernel,iterations = 1)

# 显示图片
# ## 效果展示
cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
cv2.imshow('origin', img)
 
 
cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\3-27dtest.jpg',dilate)

#  边缘提取 并保存图片
canny1=cv2.Canny(dilate,150,200)
cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\3-27cannytest.jpg',canny1)
#   展示提取后的图片
cv2.namedWindow('canny1', cv2.WINDOW_NORMAL)
cv2.imshow('canny1', canny1)
# kernel2 = np.ones((2,1),np.uint8)
# erosion = cv2.erode(canny,kernel2,iterations = 1)
# cv2.imwrite('lishuwang_erosion.jpg',erosion)


#  去除孤立点
_, labels, stats, centroids = cv2.connectedComponentsWithStats(canny1)
print(centroids)
print("stats",stats)
i=0
for istat in stats:
    if istat[4]<120:
        #print(i)
        print(istat[0:2])
        if istat[3]>istat[4]:
            r=istat[3]
        else:r=istat[4]
        cv2.rectangle(canny1,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  # 26
    i=i+1
 
#  保存去除孤立点后的边缘提取图片 并展示
cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\3-27canny_test.jpg',canny1)
cv2.namedWindow('canny',cv2.WINDOW_NORMAL)
cv2.imshow('canny', canny1)
#  按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()