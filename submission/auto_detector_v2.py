import cv2
import numpy as np
import imutils 
import distanceC_v3
import FindLine_v3

def auto_detection(path_tem_edge, path_test_edge, path_tem, path_test, line, distance):
    Move=center_moving(path_tem_edge,path_test_edge)

    for n in range(len(lines)):
        lines[n][0][0]+=Move[0]
        lines[n][1][0]+=Move[0]
        lines[n][0][1]+=Move[1]
        lines[n][1][1]+=Move[1]

    for n in range(len(distances)):
        distances[n][0][0]+=Move[0]
        distances[n][1][0]+=Move[0]
        distances[n][0][1]+=Move[1]
        distances[n][1][1]+=Move[1]


    distance=[]
    point1=[]
    point2=[]
    count=[]
    point=[]
    angle,corr = compare(path_tem_edge,path_test_edge,path_tem,path_test)
    print(angle,corr)
    im = cv2.imread(path_test)
    thresh = rotate(im,angle,find_center(path_test_edge)[0])
    path='C:\\Users\\93115\\Desktop\\mat\\3-25test.jpg'
    cv2.imwrite(path,thresh)

    filename=path #原图的路径
    img = cv2.imread(filename,0)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    # 膨胀
    dilate = cv2.dilate(erosion,kernel,iterations = 1)
    #  边缘提取 并保存图片
    canny1=cv2.Canny(dilate,150,200)
    #  去除孤立点
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(canny1)
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
    cv2.imwrite(path,canny1)

    distance=[]
    point1=[]
    point2=[]
    count=[]
    point3=[]
    point4 = []
    for d in distances:
        x=distanceC_v3.DistanceCalculate(path,d[0],d[1])
        distance.append(x[0])
        point1.append(x[1])
        point2.append(x[2])
    for l in lines:
        x=FindLine_v3.Line(path,l[0],l[1])
        count.append(x[0])
        point3.append(x[1]) 
        point4.append(x[2])
    
    img=cv2.imread(path)
    # for i,p in enumerate(point):
    #     for x,y in p:
    #         img[x,y][1],img[x,y][2]=255,0
    #     cv2.putText(img,str(count[i])+'pixels', (int((p[0][1]+p[-1][1])/2),int((p[0][0]+p[-1][0])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)
    for i in range(len(point3)):
        x1,y1=point3[i]
        x2,y2=point4[i]
        cv2.line(img,(int(x1),int(y1)), (int(x2),int(y2)), (0,0,255),2)
        cv2.putText(img,str(count[i])+'pixels', (int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)
    for i in range(len(point1)):
        x1,y1=point1[i]
        x2,y2=point2[i]
        cv2.line(img,(int(x1),int(y1)), (int(x2),int(y2)), (0,0,255),2)
        cv2.putText(img,str(distance[i])+'pixels', (int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)
    cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
    cv2.imshow("lines", img)
    cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\4-1test.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 

def find_center(path):
    # 找到一个边缘图像的最小外接圆的圆心位置
    # 输入边缘图片的路径，返回圆心位置
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    # 大津阈值，二值化原来的unit8图像

    # 腐蚀一圈，不腐蚀的话边缘不连贯，不能提取准确的外接圆
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(thresh,kernel,iterations = 1)

    contours = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
    #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    
    cnts = contours[0] if imutils.is_cv2() else contours[1]  #用imutils来判断是opencv是2还是2+
    
    for cnt in cnts:
        # # 外接矩形框，没有方向角
        # x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # # 最小外接矩形框，有方向角
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
    
        # 最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(im, center, radius, (255, 0, 0), 2)
    
        # # 椭圆拟合
        # ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(im, ellipse, (255, 255, 0), 2)
    
        # # 直线拟合
        # rows, cols = im.shape[:2]
        # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((cols - x) * vy / vx) + y)
        # im = cv2.line(im, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
    return center,radius
    
def center_moving(path_tem,path_test):
    # 输入模板和测试图的路径，返回侧视图需要平移多少距离[x,y]才能和模板重合
    center_tem=find_center(path_tem)[0]
    center_test=find_center(path_test)[0]
    return [center_test[0]-center_tem[0],center_test[1]-center_tem[1]]

def rotate(image, angle, center=None, scale=1.0): #1
    # 输入中心点，旋转的角度，对图像进行旋转
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
 
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
 
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7

def compare(path_tem,path_test,p1,p2):
    # 计算两个图对齐要旋转的角度，通过相关性计算寻找角度
    # path_tem 指的是模板图的边缘图像的路径，p1 指的是模板图原图像的路径
    # path_test 指的是要测量的图的边缘图像的路径，p2 指的是要测量的图的原图像的路径
    img_tem = cv2.imread(path_tem)
    img_test = cv2.imread(path_test)

    c_tem,r_tem = find_center(path_tem)
    c_test,r_test = find_center(path_test)
    r=max(r_tem,r_test)
    # 取两个图最大外接圆半径的最大值
    im_tem=cv2.imread(p1)
    im_tem=cv2.cvtColor(im_tem,cv2.COLOR_BGR2GRAY)
    im_test=cv2.imread(p2)
    im_test=cv2.cvtColor(im_test,cv2.COLOR_BGR2GRAY)
    # 原图都要灰度化才能进行相关性计算
    im_tem=im_tem[c_tem[1]-r:c_tem[1]+r,c_tem[0]-r:c_tem[0]+r]
    # 从模板的原图裁剪包裹住图像的正方形区域
    Max=0
    angle=0

    for i in range(360):
        im=rotate(im_test,i,c_test)
        # 旋转测量图
        im=im[c_test[1]-r:c_test[1]+r,c_test[0]-r:c_test[0]+r]
        # 裁剪
        corr=calcPearsonCorr2(im_tem,im)
        if corr>Max:
            angle=i
            Max=corr
    return angle,corr

def calcPearsonCorr2(a,b):
    # 计算两个图像的相关性，需要满足a、b两个图的尺寸相同
    img0= a.reshape(a.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img1= b.reshape(b.size, order='C')

    corr = np.corrcoef(img0, img1)[0][1]# 计算相关系数

    return corr

distances=[[[664,611],[676,719]],[[813,614],[828,716]]]
# lines=[[[663,565],[788,592]],[[665,649],[785,670]],[[485,564],[629,591]],[[493,646],[651,684]]]
lines=[[[578,602],[756,654]],[[580,688],[748,725]],[[744,604],[912,654]],[[580,688],[748,725]],[[750,677],[914,731]]]
p1="C:\\Users\\93115\\Desktop\\mat\\3-26template.jpg" 
p2="C:\\Users\\93115\\Desktop\\mat\\3-28test.jpg"
p3="C:\\Users\\93115\\Desktop\\mat\\3-27canny_tem.jpg"  
p4="C:\\Users\\93115\\Desktop\\mat\\3-27canny_test.jpg" 

auto_detection(p3,p4,p1,p2,lines,distances)