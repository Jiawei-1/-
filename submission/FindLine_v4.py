import cv2 
import numpy as np
import math
def Line(path, Left, Right): 
    # 测量边缘提取后的图片的线段长度，
    # 输入边缘提取后的照片路径，测量区域的左上点坐标和右下点坐标 
    # 返回是线段的点集和线段的长度
    #与v3相比只是加了点注释，还没有加上xml文件部分
    point = [] # 点集，用来储存是线段的点的坐标
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) 
    # 把unit8格式的图片数据二值化
    edges = edges[Left[1]:Right[1], Left[0]:Right[0]]
    # cv2.namedWindow('lines2',cv2.WINDOW_NORMAL)
    # cv2.imshow("lines2", edges)
    img = img[Left[1]:Right[1], Left[0]:Right[0]]
    # 对图像进行裁剪，缩小计算区域，提高精度
    Max = 0 # 用于计算点集中相聚最远的点
    lines = cv2.HoughLines(edges,1,np.pi/180, 20)
    # 霍夫变换提取直线
    for r,theta in lines[0]:
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
        # x0 stores the value rcos(theta)
        x0 = a*r
        # y0 stores the value rsin(theta)
        y0 = b*r
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
        #y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be 
        #drawn. In this case, it is red. 
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1) #把直线用红色画出来，1倍宽度，方便观测，后面主要用(x1,y1)和(x2,y2)两个点计算

    

    for x in range(edges.shape[0]):   # 图片的高
        for y in range(edges.shape[1]):   # 图片的宽
            px = edges[x,y]
            if px == 255: # 如果是边缘点，计算与拟合直线的距离
                if get_point_line_distance([y,x],[[x1,y1],[x2,y2]]) <= 1: #距离小于阈值就加入点集，1是阈值可以随便改

                # if img[x,y][0]==0 and img[x,y][1]==0 and img[x,y][2]==255:
                    img[x,y][1], img[x,y][2] = 255, 0 #标绿方便观察
                    point.append([y+Left[0], x+Left[1]]) #把目标点加入点集
                  
    for i,p in enumerate(point):
        for j in range(i+1,len(point)):
            p2 = point[j]
            d = math.sqrt((p[0] - p2[0]) * (p[0] - p2[0]) + (p[1] - p2[1]) * (p[1] - p2[1])) #计算距离最大的两个点
            if d > Max :
                Max = d
                xx1 = i #保存数据
                xx2 = j
    # cv2.imshow("lines", img)
    # cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\3-25line.jpg', img)
    # print(count)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return Max, point[xx1], point[xx2]

def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0][0]
    line_s_y = line[0][1]
    line_e_x = line[1][0]
    line_e_y = line[1][1]
    #若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    #若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    #斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    #截距
    b = line_s_y - k * line_s_x
    #带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis