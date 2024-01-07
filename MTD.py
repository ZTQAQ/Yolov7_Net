###混合高斯模型测试
import cv2#opencv
import numpy as np#NumPy是python的扩展程序库，支持大量的维度数组与矩阵运算。用np代替numpy


def MTD(adr):

  #行人检测参数
  # min_countour_area= 500 #最小轮廓面积阈值
  # max_countour_area= 3000 #最大轮廓面积阈值
  # min_aspect_ratio= 0.3#最小纵横比阈值
  # max_aspect_ratio= 8.0#最大纵横比阈值
  # 第一步：使用cv2.VideoCapture读取视频
  # camera = cv2.VideoCapture('demos.mp4')#0为本机，如果有已有视频则为数据路径
  camera = cv2.VideoCapture(adr)#0为本机，如果有已有视频则为数据路径
  threshold = 2000
  x = threshold
  # 判断视频是否打开
  if (camera.isOpened()):
    print('视频已打开')
  else:
    print('视频未打开')
  # frame_width = int(camera.get(3))
  # frame_height = int(camera.get(4))
  # fourcc = cv2.VideoWriter_fourcc(*'XVID')
  #out1 = cv2.VideoWriter("./result_preprocessing/MTD/back_mod.avi",fourcc, 10, (frame_width, frame_height))
  #out2 = cv2.VideoWriter("./result_preprocessing/MTD/mo_det.avi",fourcc, 10, (frame_width, frame_height))
  size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print('size:'+repr(size))#转换为整数型，用repr()函数转换成string格式输出
    
  # 第二步：cv2.getStructuringElement构造形态学使用的kernel
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#椭圆
  # 第三步：构造高斯混合模型，以高斯混合模型为基础的背景/前景分割算法
  model = cv2.createBackgroundSubtractorMOG2()
  
  while(True):
      # 第四步：读取视频中的图片，并使用高斯模型进行拟合
      ret, frame = camera.read()#参数ret 为True 或者False,代表有没有读取到图片。第二个参数frame表示截取到一帧的图片
      ret, frame = camera.read()
      ret, frame = camera.read()
      # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255，将获取到的每一帧图像都应用到当前的背景提取当中，前景置为255，背景置为0
      fgmk = model.apply(frame)
      # 第五步：使用形态学的开运算做背景的去除，开运算，先腐蚀后膨胀
      fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
      # 第六步：cv2.findContours计算fgmk的轮廓
      contours, hierarchy = cv2.findContours(fgmk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓（外轮廓，四顶点）
      for c in contours:
          if cv2.contourArea(c) < x:#当前c值太小的话找下一个
              continue
          # area = cv2.contourArea(c)
          (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
          # aspect_ratio = float(w)/h
          # if min_countour_area<area<max_countour_area and min_aspect_ratio < aspect_ratio <max_aspect_ratio:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  
      # 第八步：进行图片的展示
      cv2.imshow('fgmk', fgmk)
      #out1.write(fgmk)
      cv2.imshow('frame', frame)
      #img = np.hstack((fgmk, frame))
      #out2.write(frame)
      if cv2.waitKey(150) & 0xff == 27:
          break
          
  camera.release()
  cv2.destroyAllWindows()

# MTD('demos.mp4')

