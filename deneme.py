# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'deneme.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1111, 856)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1081, 841))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tableWidget = QtWidgets.QTableWidget(self.tab)
        self.tableWidget.setGeometry(QtCore.QRect(20, 50, 1041, 701))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(30, 770, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.label_11 = QtWidgets.QLabel(self.tab)
        self.label_11.setGeometry(QtCore.QRect(410, 10, 161, 21))
        self.label_11.setObjectName("label_11")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab)
        self.pushButton_6.setGeometry(QtCore.QRect(150, 770, 93, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.textEdit = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit.setGeometry(QtCore.QRect(20, 170, 431, 181))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_2.setGeometry(QtCore.QRect(530, 170, 431, 181))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_3.setGeometry(QtCore.QRect(20, 420, 431, 171))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_4.setGeometry(QtCore.QRect(530, 420, 431, 171))
        self.textEdit_4.setObjectName("textEdit_4")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(30, 140, 71, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setGeometry(QtCore.QRect(30, 390, 55, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setGeometry(QtCore.QRect(530, 140, 81, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(540, 400, 55, 16))
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_2.setGeometry(QtCore.QRect(370, 60, 121, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_7.setGeometry(QtCore.QRect(890, 760, 121, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_10 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_10.setGeometry(QtCore.QRect(762, 760, 111, 28))
        self.pushButton_10.setObjectName("pushButton_10")
        self.comboBox = QtWidgets.QComboBox(self.tab_2)
        self.comboBox.setGeometry(QtCore.QRect(190, 40, 121, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_2.setGeometry(QtCore.QRect(190, 90, 121, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.radioButton = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton.setGeometry(QtCore.QRect(30, 40, 141, 20))
        self.radioButton.setStyleSheet("font: 75 8pt \"MS Shell Dlg 2\";")
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton_2.setGeometry(QtCore.QRect(30, 90, 111, 20))
        self.radioButton_2.setStyleSheet("font: 75 8pt \"MS Shell Dlg 2\";")
        self.radioButton_2.setObjectName("radioButton_2")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(20, 650, 55, 16))
        self.label_12.setObjectName("label_12")
        self.textEdit_9 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_9.setGeometry(QtCore.QRect(100, 650, 361, 21))
        self.textEdit_9.setObjectName("textEdit_9")
        self.textEdit_10 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_10.setGeometry(QtCore.QRect(100, 690, 361, 21))
        self.textEdit_10.setObjectName("textEdit_10")
        self.label_14 = QtWidgets.QLabel(self.tab_2)
        self.label_14.setGeometry(QtCore.QRect(20, 730, 55, 16))
        self.label_14.setObjectName("label_14")
        self.textEdit_11 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_11.setGeometry(QtCore.QRect(100, 730, 361, 21))
        self.textEdit_11.setObjectName("textEdit_11")
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(20, 690, 55, 16))
        self.label_13.setObjectName("label_13")
        self.label_15 = QtWidgets.QLabel(self.tab_2)
        self.label_15.setGeometry(QtCore.QRect(20, 770, 55, 16))
        self.label_15.setObjectName("label_15")
        self.textEdit_12 = QtWidgets.QTextEdit(self.tab_2)
        self.textEdit_12.setGeometry(QtCore.QRect(100, 770, 361, 21))
        self.textEdit_12.setObjectName("textEdit_12")
        self.label_10 = QtWidgets.QLabel(self.tab_2)
        self.label_10.setGeometry(QtCore.QRect(50, 610, 101, 31))
        self.label_10.setObjectName("label_10")
        self.comboBox_4 = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_4.setGeometry(QtCore.QRect(560, 70, 111, 22))
        self.comboBox_4.setObjectName("comboBox_4")
        self.label_34 = QtWidgets.QLabel(self.tab_2)
        self.label_34.setGeometry(QtCore.QRect(200, 10, 101, 21))
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.tab_2)
        self.label_35.setGeometry(QtCore.QRect(550, 40, 261, 16))
        self.label_35.setObjectName("label_35")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.textEdit_13 = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit_13.setGeometry(QtCore.QRect(470, 40, 411, 191))
        self.textEdit_13.setObjectName("textEdit_13")
        self.label_16 = QtWidgets.QLabel(self.tab_3)
        self.label_16.setGeometry(QtCore.QRect(480, 10, 55, 16))
        self.label_16.setObjectName("label_16")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(900, 80, 151, 31))
        self.label_9.setObjectName("label_9")
        self.textEdit_7 = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit_7.setGeometry(QtCore.QRect(900, 110, 161, 121))
        self.textEdit_7.setObjectName("textEdit_7")
        self.label_17 = QtWidgets.QLabel(self.tab_3)
        self.label_17.setGeometry(QtCore.QRect(910, 10, 81, 21))
        self.label_17.setObjectName("label_17")
        self.textEdit_8 = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit_8.setGeometry(QtCore.QRect(900, 40, 141, 21))
        self.textEdit_8.setObjectName("textEdit_8")
        self.pushButton_11 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_11.setGeometry(QtCore.QRect(20, 760, 93, 28))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_5.setGeometry(QtCore.QRect(120, 760, 93, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_33 = QtWidgets.QLabel(self.tab_3)
        self.label_33.setGeometry(QtCore.QRect(540, 360, 521, 311))
        self.label_33.setText("")
        self.label_33.setObjectName("label_33")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setGeometry(QtCore.QRect(44, 380, 371, 291))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 451, 301))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_3.setGeometry(QtCore.QRect(20, 45, 251, 31))
        self.comboBox_3.setObjectName("comboBox_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(20, 10, 271, 21))
        self.label.setObjectName("label")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_4.setGeometry(QtCore.QRect(300, 50, 91, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_12.setGeometry(QtCore.QRect(300, 140, 91, 31))
        self.pushButton_12.setObjectName("pushButton_12")
        self.comboBox_5 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_5.setGeometry(QtCore.QRect(20, 140, 251, 31))
        self.comboBox_5.setObjectName("comboBox_5")
        self.label_32 = QtWidgets.QLabel(self.groupBox_3)
        self.label_32.setGeometry(QtCore.QRect(20, 100, 311, 31))
        self.label_32.setObjectName("label_32")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_13.setGeometry(QtCore.QRect(300, 240, 93, 31))
        self.pushButton_13.setObjectName("pushButton_13")
        self.comboBox_6 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_6.setGeometry(QtCore.QRect(20, 240, 251, 31))
        self.comboBox_6.setObjectName("comboBox_6")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(20, 205, 221, 21))
        self.label_6.setObjectName("label_6")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_4.setGeometry(QtCore.QRect(470, 240, 551, 71))
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_25 = QtWidgets.QLabel(self.groupBox_4)
        self.label_25.setGeometry(QtCore.QRect(10, 40, 31, 21))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.groupBox_4)
        self.label_26.setGeometry(QtCore.QRect(190, 40, 41, 16))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.groupBox_4)
        self.label_27.setGeometry(QtCore.QRect(380, 40, 41, 16))
        self.label_27.setObjectName("label_27")
        self.textEdit_20 = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit_20.setGeometry(QtCore.QRect(430, 40, 101, 21))
        self.textEdit_20.setObjectName("textEdit_20")
        self.label_24 = QtWidgets.QLabel(self.groupBox_4)
        self.label_24.setGeometry(QtCore.QRect(170, 0, 241, 31))
        self.label_24.setObjectName("label_24")
        self.textEdit_18 = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit_18.setGeometry(QtCore.QRect(50, 40, 101, 21))
        self.textEdit_18.setObjectName("textEdit_18")
        self.textEdit_19 = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit_19.setGeometry(QtCore.QRect(240, 40, 101, 21))
        self.textEdit_19.setObjectName("textEdit_19")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.label_28 = QtWidgets.QLabel(self.tab_4)
        self.label_28.setGeometry(QtCore.QRect(530, 20, 421, 31))
        self.label_28.setObjectName("label_28")
        self.pushButton_8 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_8.setGeometry(QtCore.QRect(30, 760, 93, 28))
        self.pushButton_8.setObjectName("pushButton_8")
        self.label_29 = QtWidgets.QLabel(self.tab_4)
        self.label_29.setGeometry(QtCore.QRect(220, 390, 611, 351))
        self.label_29.setText("")
        self.label_29.setObjectName("label_29")
        self.groupBox = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox.setGeometry(QtCore.QRect(10, 70, 491, 281))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.textEdit_14 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_14.setGeometry(QtCore.QRect(60, 70, 231, 81))
        self.textEdit_14.setObjectName("textEdit_14")
        self.label_18 = QtWidgets.QLabel(self.groupBox)
        self.label_18.setGeometry(QtCore.QRect(20, 40, 291, 16))
        self.label_18.setObjectName("label_18")
        self.textEdit_15 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_15.setGeometry(QtCore.QRect(60, 190, 231, 81))
        self.textEdit_15.setObjectName("textEdit_15")
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setGeometry(QtCore.QRect(20, 160, 321, 16))
        self.label_19.setObjectName("label_19")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_14.setGeometry(QtCore.QRect(360, 30, 93, 28))
        self.pushButton_14.setObjectName("pushButton_14")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_2.setGeometry(QtCore.QRect(520, 70, 511, 281))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(90, 60, 93, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_31 = QtWidgets.QLabel(self.groupBox_2)
        self.label_31.setGeometry(QtCore.QRect(240, 70, 101, 21))
        self.label_31.setObjectName("label_31")
        self.textEdit_21 = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_21.setGeometry(QtCore.QRect(350, 70, 101, 21))
        self.textEdit_21.setObjectName("textEdit_21")
        self.label_30 = QtWidgets.QLabel(self.groupBox_2)
        self.label_30.setGeometry(QtCore.QRect(20, 110, 261, 21))
        self.label_30.setObjectName("label_30")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_9.setGeometry(QtCore.QRect(150, 220, 93, 28))
        self.pushButton_9.setObjectName("pushButton_9")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(40, 150, 121, 21))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_5.setGeometry(QtCore.QRect(240, 150, 121, 21))
        self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QtCore.QRect(40, 180, 111, 21))
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_6.setGeometry(QtCore.QRect(240, 180, 95, 20))
        self.radioButton_6.setObjectName("radioButton_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(10, 20, 141, 21))
        self.label_7.setObjectName("label_7")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(160, 20, 113, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.label_20 = QtWidgets.QLabel(self.tab_4)
        self.label_20.setGeometry(QtCore.QRect(30, 30, 191, 16))
        self.label_20.setObjectName("label_20")
        self.tabWidget.addTab(self.tab_4, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "IMPORT"))
        self.label_11.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#aa0000;\">DATASET IMPORT ETME</span></p></body></html>"))
        self.pushButton_6.setText(_translate("Dialog", "ILERI"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Dataset Import"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; font-style:italic;\">X Training</span></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; font-style:italic;\">X Test</span></p></body></html>"))
        self.label_4.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; font-style:italic;\">Y Training</span></p></body></html>"))
        self.label_5.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; font-style:italic;\">Y Test</span></p></body></html>"))
        self.pushButton_2.setText(_translate("Dialog", "APPLY"))
        self.pushButton_7.setText(_translate("Dialog", "ILERI"))
        self.pushButton_10.setText(_translate("Dialog", "GERI"))
        self.radioButton.setWhatsThis(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.radioButton.setText(_translate("Dialog", "Hold-Out Uygula"))
        self.radioButton_2.setText(_translate("Dialog", "K-Fold Uygula"))
        self.label_12.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0000ff;\">X Train</span></p></body></html>"))
        self.label_14.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0000ff;\">Y Train</span></p></body></html>"))
        self.label_13.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0055ff;\">X Test</span></p></body></html>"))
        self.label_15.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0055ff;\">Y Test</span></p></body></html>"))
        self.label_10.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#aa0000;\">Veri Kesitleri </span></p></body></html>"))
        self.label_34.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">Split Se??imleri</span></p></body></html>"))
        self.label_35.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">K-Fold g??rmek istedi??imiz Kesitin Se??imi</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "Dataseti Ay??rma"))
        self.label_16.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Tahmin</span></p></body></html>"))
        self.label_9.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:7pt; font-weight:600;\">Model Ba??ar??m ??l????tleri</span></p></body></html>"))
        self.label_17.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Accuracy</span></p></body></html>"))
        self.pushButton_11.setText(_translate("Dialog", "GERI"))
        self.pushButton_5.setText(_translate("Dialog", "ILERI"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0000ff;\">Model S??n??fland??rma Algoritmas??n?? Se??iniz</span></p></body></html>"))
        self.pushButton_4.setText(_translate("Dialog", "APPLY"))
        self.pushButton_12.setText(_translate("Dialog", "ISLEM SEC"))
        self.label_32.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0000ff;\">Veri Setine Uygulamak ??stedi??iniz ????lemi Se??iniz </span></p></body></html>"))
        self.pushButton_13.setText(_translate("Dialog", "UYGULA"))
        self.label_6.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0000ff;\">Toplu ????renme Yakla????m?? Se??iniz</span></p></body></html>"))
        self.label_25.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600; color:#0000ff;\">R??</span></p></body></html>"))
        self.label_26.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600; color:#0000ff;\">MAE</span></p></body></html>"))
        self.label_27.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600; color:#0000ff;\">MSE</span></p></body></html>"))
        self.label_24.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600; color:#aa0000;\">Model Ba??ar?? De??erlendirmesi</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Dialog", "Details"))
        self.label_28.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:7pt; font-weight:600; color:#aa0000;\">Yapay Sinir A??lar?? ve Derin Sinir A??lar?? ile ????renme ve test i??lemleri</span></p></body></html>"))
        self.pushButton_8.setText(_translate("Dialog", "GERI"))
        self.label_18.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">GridSearch En iyi Ba??ar?? Ve En ??yi Parametre</span></p></body></html>"))
        self.label_19.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">RandomSearch En iyi Ba??ar?? Ve En ??yi Parametre</span></p></body></html>"))
        self.pushButton_14.setText(_translate("Dialog", "UYGULA"))
        self.pushButton_3.setText(_translate("Dialog", "Testi Ba??lat"))
        self.label_31.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Ba??ar?? Sonucu</span></p></body></html>"))
        self.label_30.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600; color:#0000ff;\">G??r??nt??lemek ??stedi??iniz Grafi??i Se??iniz</span></p></body></html>"))
        self.pushButton_9.setText(_translate("Dialog", "Grafi??i G??ster"))
        self.radioButton_3.setText(_translate("Dialog", "Accuracy Grafi??i"))
        self.radioButton_5.setText(_translate("Dialog", "Confusion Matrix "))
        self.radioButton_4.setText(_translate("Dialog", "Loss Grafi??i"))
        self.radioButton_6.setText(_translate("Dialog", "ROC E??risi"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Epoch De??eri Giriniz</span></p></body></html>"))
        self.label_20.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:7pt; font-weight:600; color:#aa0000;\">GridSearch ve RandomSearch  </span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Dialog", "Result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

