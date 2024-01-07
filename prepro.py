import cv2
import cv2 as cv
import os
from PIL import ImageEnhance,Image
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager

###第一步检验
#视频转成图像序列
def frame_extraction(adr):
    #cap = cv2.VideoCapture('demos.mp4')   #获取视频
    cap = cv2.VideoCapture(adr)
    success = True
    x=cap.isOpened()
    print (x)      #视频是否打开
    num=0

    while(success): 
        success,frame = cap.read() #循环获取视频帧
        print(frame)
        if frame is None:
            break
            #print(frame)
        Img_Name = "result_preprocessing/frame_extraction/"+ str(num).zfill(4)+".jpg" # 保存图片png/jpg/bmp/tif
        num = num+1
        cv2.imwrite(Img_Name,frame)

    cap.release()                  #释放视频资源

#视频随机抽取图像帧
def image_sequence(adr):
    START_TIME= 4 #设置开始时间(单位秒)可设置
    END_TIME= 28800 #设置结束时间(单位秒)可设置

    # vidcap = cv2.VideoCapture(r'demos.mp4')
    vidcap = cv2.VideoCapture(adr)

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))  # 获取视频每秒的帧数
    print(fps)

    frameToStart = START_TIME*fps #开始帧 = 开始时间*帧率
    print(frameToStart)
    frametoStop = END_TIME*fps #结束帧 = 结束时间*帧率
    print(frametoStop)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart) #设置读取的位置,从第几帧开始读取视频 可设置
    print(vidcap.get(cv2.CAP_PROP_POS_FRAMES))  # 查看当前的帧数

    success,image = vidcap.read()  # 获取第一帧

    count = 0
    while success and frametoStop >= count:
        if count % (25*15) == 0:  # 每15秒保存一次 可设置
            cv2.imwrite(r"./result_preprocessing/Capture/1result%d.jpg" % int(count / 375), image)
            cv2.imwrite(r"./result_preprocessing/Capture/2result%d.png" % int(count / 375), image) 
            cv2.imwrite(r"./result_preprocessing/Capture/3result%d.bmp" % int(count / 375), image) 
            cv2.imwrite(r"./result_preprocessing/Capture/4result%d.tif" % int(count / 375), image)    # 保存图片png/jpg/bmp/tif 可设置加gif
            print('Process %dth seconds: ' % int(count / 375), success) #可设置
        success,image = vidcap.read()  # 每次读取一帧
        count += 1
    address="./result_preprocessing/Capture/1result0.jpg"
    print("end!")
    return address

###第二步检验
def preprocseeing(address):
    adr1 = address
    adr2 =Salt_pepper(adr1)
    balance(adr2)

#（1）增噪声去噪声    
# 椒盐噪声 (Salt-pepper)
def Salt_pepper(adr1):
    #将第一步与第二步串起来
    #frame_extraction(adr1)
    adr=image_sequence(adr1)
   
    #读取从第一步视频中抽取的帧
    img = cv2.imread(adr, 1)  # flags=0 读取为灰度图像
    
    def pepper_and_salt(img,percentage):
        i=0
        num=int(percentage*img.shape[0]*img.shape[1])#  椒盐噪声点数量
        random.randint(0, img.shape[0])
        img2=img.copy()
        for i in range(num):
            X=random.randint(0,img2.shape[0]-1)#从0到图像长度之间的一个随机整数,因为是闭区间所以-1
            Y=random.randint(0,img2.shape[1]-1)
            if random.randint(0,1) ==0: #黑白色概率55开
                img2[X,Y] = (255,255,255)#白色
            else:
                img2[X,Y] =(0,0,0)#黑色
            i=i+1
        return img2
    img2 = pepper_and_salt(img,0.04)#百分之4的椒盐噪音
    cv2.imwrite(r"./result_preprocessing/Addnoisy/Salt.jpg",img2)

    #中值滤波器
    img_median = cv2.medianBlur(img2,3) #中值滤波
    cv2.imwrite(r"./result_preprocessing/Addnoisy/filter.jpg",img_median)
    adr="./result_preprocessing/Addnoisy/filter.jpg"
    return adr

#(2)均衡
def balance(adr2):
# def balance(adr2):
        # adr2='./result_preprocessing/Addnoisy/filter.jpg'
        img_rbg = cv2.imread(adr2, 1)  # flags=0 读取为灰度图像
            # b,g,r = cv2.split(img_rbg)
            # test = cv2.merge([r,g,b])

        B,G,R = cv2.split(img_rbg) #get single 8-bits channel
        EB=cv2.equalizeHist(B)
        EG=cv2.equalizeHist(G)
        ER=cv2.equalizeHist(R)
        equal_test=cv2.merge((EB,EG,ER))  #merge it back
        cv2.imwrite(r"./result_preprocessing/Balance/origin.jpg",img_rbg)
        cv2.imwrite(r"./result_preprocessing/Balance/result.jpg",equal_test)

            #直方图
            #原来整体直方图
            # color = ('b','g','r')
            # for i, col in enumerate(color):
            #     histr = cv.calcHist([img_rbg], [i], None, [256], [0,256])
            #     plt.plot(histr, color=col)
            #     plt.xlim([0,256])

            #各通道直方图
        # hist_B=cv.calcHist([B],[0],None,[256],[0,256]) 
        # hist_G=cv.calcHist([G],[0],None,[256],[0,256]) 
        # hist_R=cv.calcHist([R],[0],None,[256],[0,256]) 
        # hist_EB=cv.calcHist([EB],[0],None,[256],[0,256]) 
        # hist_EG=cv.calcHist([EG],[0],None,[256],[0,256]) 
        # hist_ER=cv.calcHist([ER],[0],None,[256],[0,256]) 
        
        hist_B= cv2.calcHist([img_rbg], [0], None, [256], [0, 255])
        hist_EB=cv2.calcHist([equal_test],[0],None,[256],[0,256])
        hist_G= cv2.calcHist([img_rbg], [1], None, [256], [0, 255])
        hist_EG=cv2.calcHist([equal_test],[1],None,[256],[0,256])
        hist_R= cv2.calcHist([img_rbg], [2], None, [256], [0, 255])
        hist_ER=cv2.calcHist([equal_test],[2],None,[256],[0,256])
        
        plt.subplot(3,1,1)
        plt.plot(hist_B,color = 'red', label = "Blue channel histogram", linestyle = '--', alpha = 1)
        plt.legend()
        plt.plot(hist_EB,color='deepskyblue', label = "Blue channel_hist histogram", linestyle = '--', alpha = 1)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(hist_G,color = 'red', label = "Green channel histogram", linestyle = '--', alpha = 1)
        plt.legend()
        plt.plot(hist_EG,color='deepskyblue', label = "Green channel_hist histogram", linestyle = '--', alpha = 1)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(hist_R,color = 'red', label = "Red channel histogram", linestyle = '--', alpha = 1)
        plt.legend()
        plt.plot(hist_ER,color='deepskyblue', label = "Red channel_hist histogram", linestyle = '--', alpha = 1)
        plt.legend()
        plt.savefig('./result_preprocessing/Balance/hist.jpg', dpi=300, bbox_inches='tight')
        print('yes!')   
#(3)视频整体的增强
def img_enhance(image):
    # # 亮度增强
    # enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.5
    # image_brightened = enh_bri.enhance(brightness)
    # # image_brightened.show()
    image_brightened =image
 
    # 色度增强
    enh_col = ImageEnhance.Color(image_brightened)
    color = 1.5
    image_colored = enh_col.enhance(color)
    # image_colored.show()
 
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()
 
    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.show()
    #去噪声
    # image = np.asarray(image)
    # kernel = (5,5)
    # sigma = 1.5
    # image = cv2.GaussianBlur(image, kernel , sigma)
    # #直方图均衡化
    # B,G,R = cv2.split(image) #get single 8-bits channel
    # EB=cv2.equalizeHist(B)
    # EG=cv2.equalizeHist(G)
    # ER=cv2.equalizeHist(R)
    # equal_test=cv2.merge((EB,EG,ER))  #merge it back
    # #去模糊
    # #frame = cv2.GaussianBlur(equal_test,(5.5),0)
    # # #去过曝光
    # gamma = 0.8
    # frame = np.power(equal_test/ 255.0, gamma)
    # frame = np.uint8(frame * 255)
    return image_sharped
def video_enhancement(adr):
    print(adr)
    adr1=adr
    preprocseeing(adr1)
    cap = cv2.VideoCapture(adr1)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
# 设置需要保存视频的格式“xvid”
# 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 按照设置的格式来out输出
    out = cv2.VideoWriter("./result_preprocessing/Enhanced_vid/result.avi",fourcc, 10, (frame_width, frame_height))  # 保存视频
    while(cap.isOpened()):
        ret, frame = cap.read()
        ret, frame = cap.read() 
        # ret, frame = cap.read() # 读出来的frame是ndarray类型
        

        img_out = op_one_img(frame)
        out.write(img_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('预处理完毕')
#逐帧处理同步进行增强画质处理，并显示
def op_one_img(frame):
    images = []
    image = Image.fromarray(np.uint8(frame))  # 转换成PIL可以处理的格式
    images.append(image)
    image_enhanced = img_enhance(image)  # 调用编写的画质增强函数
    return np.asarray(image_enhanced) # 显示的时候要把格式转换回来



