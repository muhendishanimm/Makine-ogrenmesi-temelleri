# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:01:17 2020

@author: aylin
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from denemekod import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()