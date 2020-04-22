import cv2
import numpy as np
import imutils 
from xml.dom import minidom
import os

# path='C:\\Users\\93115\\Desktop\\mat\\3-6canny1.jpg' # 保存边缘提取后的图片位置
XMLfile= "C:\\Users\\93115\\Desktop\\dimension.xml"
# 顶点坐标Left，Right的数据类型是list，例如Left=[10,20]表示左顶点的横坐标是10，纵坐标是20
def DistanceCalculate(path,Left,Right): #测量上下边界的宽度 传入保存边缘的路径 ROI的左右顶点位置 输出宽度值和两点坐标
    # 读取边缘，并灰度化
    img=cv2.imread(path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 选取要测量的部分
    # cv2.namedWindow('Canny Edge', cv2.WINDOW_NORMAL) 
    # bbox = cv2.selectROI('Canny Edge',img ,False)
    cut = img[Left[1]:Right[1], Left[0]:Right[0]]
    # cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\cut.jpg', cut)#保存切割后的图片
    ret,cut = cv2.threshold(cut,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    img=cut

    distance = 10000 #预设的最大值
    
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
                point1=[i+Left[0],conture_up[i]+Left[1]]
                point2=[j+Left[0],conture_down[j]+Left[1]]
    print(distance)
    DistanceWriteXml(XMLfile,point1,point2,distance)
    # 将点和长度写入xml文件
    return distance,point1,point2

def DistanceWriteXml(filename,x,y,length):
    i=0
    # doc=parse("./customer.xml")
    if not os.path.exists(filename): 
        # 如果文件没有根节点 创建根节点
        doc = minidom.Document()
        dimension = doc.createElement('Dimension')
        i=1
    else:
        # 如果有根节点，访问根节点，在根节点下写入参数
        doc = minidom.parse(filename)
        dimension=doc.documentElement
    line=doc.createElement("Distance")
    dimension.appendChild(line)
    point1 = doc.createElement("startpoint")
    # distance的子节点，属性是xy坐标
    point1.setAttribute("x", str(x[0]))
    point1.setAttribute("y", str(x[1]))
    point2 = doc.createElement("endpoint")
    # distance的子节点，属性是xy坐标
    point2.setAttribute("x", str(y[0]))
    point2.setAttribute("y", str(y[1]))
    line.appendChild(point1)
    line.appendChild(point2)
    line.setAttribute("length",str(length))
    # distance的属性length，代表测量得到的宽度的长度
    doc.appendChild(dimension)
    # filename = "C:\\Users\\gjw\\Desktop\\dimension.xml"
    if i==1:
        f = open(filename, "a")
        doc.writexml(f, addindent='  ', encoding='utf-8')
        f.close()
    else:
        with open(filename, 'w') as f:
            # 缩进 - 换行 - 编码
            doc.writexml(f, addindent='  ', encoding='utf-8')
