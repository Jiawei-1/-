def Line(path, Left, Right): 
    # 测量边缘提取后的图片的线段长度，
    # 输入边缘提取后的照片路径，测量区域的左上点坐标和右下点坐标 
    # 返回是线段的点集和线段的长度

    point=[] # 点集，用来储存是线段的点的坐标
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,edges = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    # 把unit8格式的图片数据二值化
    edges = edges[Left[1]:Right[1], Left[0]:Right[0]]
    # cv2.namedWindow('lines2',cv2.WINDOW_NORMAL)
    # cv2.imshow("lines2", edges)
    img=img[Left[1]:Right[1], Left[0]:Right[0]]
    # 对图像进行裁剪，缩小计算区域，提高精度

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
        cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) #把直线用红色画出来，2倍宽度

    count =0 # 计算数量

    for x in range(edges.shape[0]):   # 图片的高
        for y in range(edges.shape[1]):   # 图片的宽
            px = edges[x,y]
            if px== 255: # 如果是边缘点，就判断边缘点在不在拟合出的红色直线上
                if img[x,y][0]==0 and img[x,y][1]==0 and img[x,y][2]==255:
                    img[x,y][1],img[x,y][2]=255,0
                    point.append([x+Left[1],y+Left[0]]) #把目标点加入点集
                    count+=1 # 加法器

    # cv2.imshow("lines", img)
    # cv2.imwrite('C:\\Users\\93115\\Desktop\\mat\\3-25line.jpg', img)
    # print(count)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return point,count
