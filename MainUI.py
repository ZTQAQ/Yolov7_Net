import os
import shutil
from PyQt5 import QtWidgets, uic
import sys
import random
import numpy as np
import torch
import cv2
import argparse
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage, QMovie
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame
from MTD import MTD
# from prepro import image_sequence,frame_extraction

from prepro import video_enhancement
from matplotlib import pyplot as plt
import matplotlib
from track import parse_opt, main, run
from pathlib import Path

class Ui(QFrame,QtWidgets.QMainWindow):
    #  看ui生成的py文件，是继承wideget，class_UI需要加上QFrame，不然不能栅格布局
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('MainUi2.ui', self)
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setWindowTitle("行人检测系统")
        self.cap = cv2.VideoCapture()
        self.timer_video = QtCore.QTimer()
        self.timer_camera = QtCore.QTimer()
        self.out = None
        # 权重初始文件名
        self.openfile_name_model = None
        self.pushButton_1.clicked.connect(self.button_video_open)
        self.pushButton_2.clicked.connect(self.button_preprocessing)
        self.pushButton_3.clicked.connect(self.button_preprocessing_result)
        self.pushButton_4.clicked.connect(self.button_MTD_open)
        self.pushButton_5.clicked.connect(self.button_pedestrian_detection)
        self.timer_camera.timeout.connect(self.show_video)
        self.show()

    # 选择视频
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        self.textEdit.setText(video_name)
        self.label_3.setText("视频导入成功")
        self.cap = cv2.VideoCapture(video_name)
        self.timer_camera.start(33)            # 这行代码直接注释就可以不在label_3上播放视频

    # 预处理
    def button_preprocessing(self):
        self.label_3.setText("正在预处理中，请稍后")
        print(self.textEdit.toPlainText())
        try:
            video_enhancement(self.textEdit.toPlainText())
        except:
            pass
        self.label_3.setText("预处理已完成")

    # 预处理结果展示
    def button_preprocessing_result(self):
        self.finish_detect()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开文件", "", "图像文件 (*.jpg *.png *.bmp);;视频文件 (*.mp4 *.avi)")

        if file_path:
            # 判断选择的文件是图片还是视频
            if file_path.endswith((".jpg", ".png", ".bmp")):
                # 处理图片文件
                pixmap = QtGui.QPixmap(file_path)
                self.label_3.setPixmap(pixmap)
                self.label_3.setScaledContents(True)
            elif file_path.endswith((".mp4", ".avi")):
                # 处理视频文件
                self.cap = cv2.VideoCapture(file_path)
                self.timer_camera.start(33)
                pass

    # 运动区域检测展示
    def button_MTD_open(self):
        print(self.textEdit.toPlainText())
        MTD(self.textEdit.toPlainText())
        # MTD_filename()
        # flag = self.cap.open(video_name)
        '''
        flag, img = self.cap.read()
        if flag:
            # 将帧转换为 QImage 对象，并显示到 QLabel 控件上
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            elif len(img.shape) == 1:
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Indexed8)
            else:
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.label_2.setPixmap(pixmap)
            self.label_2.setScaledContents(True)
            '''

    # 行人检测区域展示
    def button_pedestrian_detection(self):
        self.finish_detect()
        self.label_3.setText("正在检测视频，请稍后")
        opt = parse_opt(self.textEdit.toPlainText())
        main(opt)
        # 去找到前两行生成的新视频文件
        print("6666666")
        directory = "runs/track"
        folder_path = "runs/track/exp"  # 指定"exp"文件夹的路径
        # if os.path.exists(folder_path):
          #  shutil.rmtree(folder_path)
        folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name != 'exp']
        print("folder:",folders)
        latest_folder = max(folders, key=lambda x: int(x[3:]))
        folder_path = os.path.join(directory, latest_folder)  # 文件夹路径
        files = os.listdir(folder_path)
        print(files)
        file_name = files[0]
        print(file_name)
        # 构建最新生成文件夹中的视频文件路径
        #video_path = os.path.join(directory, latest_folder, "test.mp4")
        video_path = directory +'/'+latest_folder+'/'+file_name
        print(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.timer_camera.start(33)

    # 逐帧播放
    def show_video(self):
        flag, img = self.cap.read()
        # print(flag)
        if flag:
            # 将帧转换为 QImage 对象，并显示到 QLabel 控件上
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            elif len(img.shape) == 1:
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Indexed8)
            else:
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            # print(pixmap)
            self.label_3.setPixmap(pixmap)
            self.label_3.setScaledContents(True)
        else:
            print("None")
            self.timer_camera.stop()
            self.cap.release()
            self.label_3.clear()

    # 结束播放，清空label，避免同时放俩视频
    def finish_detect(self):
            self.cap.release()  # 释放video_capture资源
            self.label_3.clear()  # 清空label画布
            self.timer_camera.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()
    sys.exit(app.exec())