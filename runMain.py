# -*- coding: utf-8 -*-
"""
The library it needs：
    keras 2.2.4
    PyQt5 5.11.3
    pandas 0.24.2
    scikit-learn 0.21.2
    tensorflow 1.13.1
    imutils 0.5.2
    opencv-python 4.10.25
    matplotlib 3.2.1 

here is the main function: runMain.py
"""
# @Time    : 12/07/2020
# @Author  : Yibo li
# @Email   : lslsls0001@gmail.com
# @Software: Anaconda
# @version : 2.2 

import warnings
import os
# ignore the warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from EmotionRecongnition import Ui_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication,QMainWindow


if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
