import cv2
import numpy as np 
def mathc_img(image, Target, save_path, value=0.9):
    # image是测量的图片的地址，target是模板的地址，
    # save_path是对齐后的图片的输出地址，value是给定的匹配阈值默认0.9
    img_rgb = cv2.imread(image) 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
    template = cv2.imread(Target,0) 
    w, h = template.shape[::-1] 
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)  
    if max(map(max,res)) <value:
        return 
    loc = np.where( res >= max(map(max,res))) 
    for pt in zip(*loc[::-1]): 
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,155), 2) 
        img_crop = img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
    # cv2.imshow('Detected',img_crop) 
    cv2.imwrite(save_path, img_crop)
    # print(max(map(max,res)))
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

# def rotate(image, angle, center=None, scale=1.0): #1
#     # 输入中心点，旋转的角度，对图像进行旋转
#     (h, w) = image.shape[:2] #2
#     if center is None: #3
#         center = (w // 2, h // 2) #4
 
#     M = cv2.getRotationMatrix2D(center, angle, scale) #5
 
#     rotated = cv2.warpAffine(image, M, (w, h),borderValue=(255,255,255) )#6
#     return rotated #7

# def angle(image, Target):
#     img_rgb = cv2.imread(image)
#     angle = 0
#     Max = 0
#     template = cv2.imread(Target,0) 
#     w, h = template.shape[::-1] 
#     for i in range(360): 
#         img_rg = rotate(img_rgb,i)
#         img_gray = cv2.cvtColor(img_rg, cv2.COLOR_BGR2GRAY) 
#         res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
#         print(i, max(map(max,res)))
#         if max(map(max,res)) > Max:
#             Max = max(map(max,res))
#             angle = i
#     return angle




# image=p1
# Target=p2
# value=0.9
# p1 = "C:\\Users\\93115\\Desktop\\mat\\test\\3-26template.jpg"
# p2 = "C:\\Users\\93115\\Desktop\\mat\\test\\3-26template2.jpg"
# p3 = 'C:\\Users\\93115\\Desktop\\mat\\test\\4-21test.jpg'
# mathc_img(image,Target,value, p3)
