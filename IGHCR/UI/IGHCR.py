# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 20:41:08 2023

@author: Sanath Ezhuthachan
"""

from PyQt5 import QtCore, QtGui, uic 
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image
# from PIL.ImageQt import ImageQt 
# from PIL import Image, ImageQt
import sys, cv2, numpy as np, ModelTesting as mt

class Window(QMainWindow): 
    
    def __init__(self): 
        super().__init__()
#        self.setGeometry(52, 80, 1820, 910) 
        uic.loadUi("IGHCR.ui",self)
        self.setWindowTitle("MCA2_04 - IGHCR  :  Isolated Gujarati Handwritten Character Recognition")
        self.setFocus()
        self.btn_loadImg.clicked.connect(self.load_img)
        self.btn_binnarizeImg.clicked.connect(self.binnarize_img)
        self.btn_thinningImg.clicked.connect(self.thinning_img)
        self.btn_predictImg.clicked.connect(self.predict_img)
        self.show()
     
    def cv_read_img(self) :
        return cv2.imread(self.filename[0])
    
    def clear_vars(self) :
        self.lbl_originalImg.clear()
        self.lbl_binnarizeImg.clear()
        self.lbl_thinnedImg.clear()
        self.txt_predictedImg.setText("")
        
    def load_img(self) :
        try :
            self.clear_vars()
            self.filename = QFileDialog.getOpenFileName(self, "Open File")
            self.img = self.cv_read_img()
            self.pixmap = QPixmap(self.filename[0])
            self.lbl_originalImg.setPixmap(self.pixmap)   
        except : print("Exception Occured At Image Acquisition.")

    def preprocess_img(self) :
        try : 
            self.preprocessed = mt.preprocessImg(self.cv_read_img())
            return self.preprocessed
        except : print("Exception Occured At Pre-Processing.")
        
    def binnarize_img(self) :
        try : 
            self.preprocess_img()
            self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]   
            self.lbl_binnarizeImg.setStyleSheet("border: 2px solid white;")
            PIL_image = Image.fromarray(self.binary_image)
            cv2.imwrite("temp-binary.png", self.binary_image)
            self.lbl_binnarizeImg.setPixmap(QPixmap("temp-binary.png"))
        except : print("Exception Occured At Binnarization.")
        
    def thinning_img(self) : 
        try :
            self.thinned = cv2.ximgproc.thinning(self.binary_image)
            self.lbl_thinnedImg.setStyleSheet("border: 2px solid white;")
            PIL_image = Image.fromarray(self.thinned)
            cv2.imwrite("temp-thinned.png", self.thinned)
            self.lbl_thinnedImg.setPixmap(QPixmap("temp-thinned.png"))
        except : print("Exception Occured At Thinning.")
        
    def predict_img(self) :
        try :
            self.predicted = mt.classifyImg(self.img, mt.model)
            print(self.predicted)
            if self.predicted[1] > 0.65 :
                self.txt_predictedImg.setText(mt.consonants[self.predicted[0]])
                self.txt_predictedImg.setAlignment(Qt.AlignCenter)
            else :
                self.txt_predictedImg.setText(mt.consonants[30])
                self.txt_predictedImg.setAlignment(Qt.AlignCenter)
        except : print("Exception Occured At Prediction.")

app = QCoreApplication.instance()
if not app : app = QApplication(sys.argv)
 
window = Window() 

app.exec()
app.quit()

#sys.exit(app.exec()) 