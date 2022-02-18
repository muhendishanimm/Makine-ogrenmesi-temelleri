# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:01:54 2020

@author: aylin
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QMainWindow, QMessageBox, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
import pandas as pd
import openpyxl
from deneme import Ui_Dialog
from sklearn import preprocessing
import seaborn as sns
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy import genfromtxt
from PyQt5.QtCore import QAbstractTableModel, Qt
import random
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC as svc
from scipy import stats
from sklearn.preprocessing import StandardScaler#standardizasyon
from sklearn.ensemble import BaggingRegressor#toplu öğrenme bagging
from sklearn.ensemble import BaggingClassifier
from sklearn import tree#toplu öğrenme için
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2#kikare 
from sklearn.preprocessing import MinMaxScaler#normalizasyon
from sklearn.neighbors import KNeighborsClassifier#knn
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import NearMiss
from collections import Counter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import scikitplot.metrics as splt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_confusion_matrix

class MainWindow(QWidget,Ui_Dialog):
    dataset_file_path = ""
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.importet)
        self.pushButton_2.clicked.connect(self.apply)
        self.pushButton_3.clicked.connect(self.testibaslat)
        self.pushButton_4.clicked.connect(self.dinfo)
        self.pushButton_5.clicked.connect(self.dorduncusayfayagec)
        self.pushButton_6.clicked.connect(self.ikincisayfayagec)
        self.pushButton_7.clicked.connect(self.ucuncusayfayagec)
        self.pushButton_10.clicked.connect(self.birincisayfayadon)
        self.pushButton_11.clicked.connect(self.ikincisayfayadon)
        self.pushButton_8.clicked.connect(self.ucuncusayfayadon)
        self.pushButton_9.clicked.connect(self.grafikgoster)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.tabWidget.setCurrentIndex(0)#her zaman ilk sayfada acilmasi icin
        self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.comboBox.addItem("0.1")
        self.comboBox.addItem("0.2")
        self.comboBox.addItem("0.3")
        self.comboBox_2.addItem("2")
        self.comboBox_2.addItem("5")
        self.comboBox_2.addItem("10")
        self.comboBox_3.addItem("SVC")
        self.comboBox_3.addItem("Logistic Regression")
        self.comboBox_3.addItem("KNN")
        self.comboBox_3.addItem("Naive Bayes")
        self.comboBox_3.addItem("Decision Tree Classification")  
        self.comboBox_4.addItem("1")
        self.comboBox_4.addItem("2")
        self.comboBox_4.addItem("3")
        self.comboBox_4.addItem("4")
        self.comboBox_4.addItem("5")
        self.comboBox_4.addItem("6")
        self.comboBox_4.addItem("7")
        self.comboBox_4.addItem("8")
        self.comboBox_4.addItem("9")
        self.comboBox_4.addItem("10")
        self.comboBox_4.setEnabled(False)
        self.pushButton_12.clicked.connect(self.islemsec)
        self.comboBox_5.addItem("Seçim Yapınız")
        self.comboBox_5.addItem("Veri Setini Dengeli Yap")
        self.comboBox_5.addItem("Normalizasyon Uygula")
        self.comboBox_5.addItem("Standardizasyon Uygula")
        self.comboBox_5.addItem("Öznitelik Seçimi Uygula")
        self.comboBox_5.addItem("Öznitelik Dönüşümü Uygula")
        self.pushButton_13.clicked.connect(self.toplu)
        self.comboBox_6.addItem("BaggingClassifier")
        self.comboBox_6.addItem("Gradient Boosting Classifier")
        self.pushButton_14.clicked.connect(self.searchler)
        self.dataset=[]
   
    def importet(self):
#verisetini ekleme        
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        print("Seçilen veriseti:",path)  
        self.dataset=pd.read_csv(path, comment='#')               
        print("Veri seti hakkında bilgi:\n")
        self.dataset.info()#dataset hakkinda bilgi veriyor
        self.dataset.head() 
        print("Shape:",self.dataset.shape)
        print("\n************************************\n")
       
        self.pushButton_6.setEnabled(True)
       
#Eksik veriyi saptama                            
        print("eksik veri var mı?\n",self.dataset.isnull().sum())  
        print("\n************************************\n")
        self.eksik_deger_tablosu()#fonksiyonu çağırdım
#tekrarları kaldır
        revised_data = self.dataset.drop_duplicates()     
        print(revised_data)
#verisetini table widgete doldurma        
        veriler=self.dataset.values
        #print(veriler.shape[1]-1)
        self.verilerim=veriler[:,0:veriler.shape[1]]
        #y=veriler[:,veriler.shape[1]-1]
        self.tableWidget.setRowCount(self.verilerim.shape[0])#satır sayısını alır veriler.shape[0]
        self.tableWidget.setColumnCount(self.verilerim.shape[1])#sutun sayısını alır veriler.shape[1]
        #for i in range(0,veriler.shape[0]):           
              #self.tableWidget.setItem(i,0,QtWidgets.QTableWidgetItem(str(veriler[i][0])))
        #self.tableWidget.setItem(0,1,"Phrases")      
        for i in range(0, self.verilerim.shape[0]):
           for j in range(0, self.verilerim.shape[1]):
              self.tableWidget.setItem(i,j ,QtWidgets.QTableWidgetItem(str(self.verilerim[i][j]))) 
        
    def eksik_deger_tablosu(self):
        eksik_deger = self.dataset.isnull().sum()
        eksik_deger_yuzde = 100 * self.dataset.isnull().sum()/len(self.dataset)
        eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
        eksik_deger_tablo_son = eksik_deger_tablo.rename(
                columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
        print(eksik_deger_tablo_son)
        self.dataset.dropna()
        self.dataset.dropna(inplace=True)
       
        
    def ikincisayfayagec(self):        
        self.tabWidget.setTabEnabled(1,True)
        self.tabWidget.setCurrentIndex(1)
        self.veriseti=self.dataset.values
        self.labels=self.dataset.columns
        self.X=self.veriseti[:,0:len(self.labels)-1]
        self.y=self.veriseti[:,len(self.labels)-1]
        
    def apply(self): 
         self.pushButton_7.setEnabled(True)
         self.textEdit.setText(" ")
         self.textEdit_2.setText(" ")
         self.textEdit_3.setText(" ")
         self.textEdit_4.setText(" ")
         self.textEdit_9.setText(" ")
         self.textEdit_10.setText(" ")
         self.textEdit_11.setText(" ")
         self.textEdit_12.setText(" ") 
#hold-out ile verisetini test train olarak ayırma     
         if(self.radioButton.isChecked()):
            #X=self.dataset.drop('target', axis=1) #data
            #y=self.dataset['target'] # target
            test_size=self.comboBox.currentText()
            self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.X,self.y, 
                                                    test_size=float(test_size),random_state=42)
            self.textEdit.setText(str(self.X_train))
            self.textEdit_2.setText(str(self.y_train))
            self.textEdit_3.setText(str(self.X_test))
            self.textEdit_4.setText(str(self.y_test))
            veri_kesitleri = {"x_train"  :self. X_train
                  ,"x_test"  : self.X_test
                  ,"y_train" : self.y_train
                  ,"y_test"  : self.y_test}
            
            for i in veri_kesitleri:
                #print(f"{i}: satır sayısı {veri_kesitleri.get(i).shape[0]}")
                if (i=="x_train"): 
                    self.textEdit_9.setText(f"{i}: satır sayısı {veri_kesitleri.get(i).shape[0]}")
                if (i=="x_test"): 
                    self.textEdit_10.setText(f"{i}: satır sayısı {veri_kesitleri.get(i).shape[0]}")
                if (i=="y_train"): 
                    self.textEdit_11.setText(f"{i}: satır sayısı {veri_kesitleri.get(i).shape[0]}")
                if (i=="y_test"): 
                    self.textEdit_12.setText(f"{i}: satır sayısı {veri_kesitleri.get(i).shape[0]}") 
                  
#k-fold ile verisetini test train olarak ayırma        
         if(self.radioButton_2.isChecked()):
            self.comboBox_4.setEnabled(True)
            n_splits=self.comboBox_2.currentText()
            indexal=self.comboBox_4.currentText()
            sayac=0
            cv = KFold(n_splits=int(n_splits), random_state=1, shuffle=True)
            for self.train, self.test in cv.split(self.dataset):
                sayac+=1
                if(sayac==int(indexal)):
                    self.X_train, self.X_test = self.X[self.train], self.X[self.test]
                    self.y_train, self.y_test = self.y[self.train], self.y[self.test]
                    self.textEdit.setText(str(self.X_train))
                    self.textEdit_2.setText(str(self.y_train))
                    self.textEdit_3.setText(str(self.X_test))
                    self.textEdit_4.setText(str(self.y_test))
                print("X train",self.X_train)
                print("X test",self.X_test)
                print("y train",self.y_train)
                print("y test",self.y_test)
                self.textEdit_9.setText(str(self.X_train.shape[0]))
                self.textEdit_10.setText(str(self.X_test.shape[0]))
                self.textEdit_11.setText(str(self.y_train.shape[0]))
                self.textEdit_12.setText(str(self.y_test.shape[0]))
         if (self.radioButton.isChecked()==False and self.radioButton_2.isChecked()==False):
            self.hata = "Lütfen bir seçim yapınız!"
            self.error()
    
    def ucuncusayfayagec(self):
        self.tabWidget.setTabEnabled(2,True)
        self.tabWidget.setCurrentIndex(2)
    
    def birincisayfayadon(self):
        self.tabWidget.setCurrentIndex(0)
        
    def dengesiz(self):
# degesiz mi dengeli mi
        soncolumn=self.dataset.columns[-1]
        print("Son column",soncolumn)
        say=self.dataset[soncolumn].value_counts().values
        print("Dengeli mi? Dengesiz mi?",say)
#NearMiss-3 : Her bir azınlık sınıfı örneğine minimum mesafe ile çoğunluk sınıfı örnekleri        
        undersample = NearMiss(version=1, n_neighbors=3)#büyüyk veriseti için undersampling
        self.X, self.y = undersample.fit_resample(self.X, self.y)
        counter = Counter(self.y)
        print(counter)
        
    def normalizet(self):
        mms=MinMaxScaler()
        self.X_train=mms.fit_transform(self.X_train)#eğitim setine normalizasyon uygulama
        self.X_test=mms.transform(self.X_test)#test setine normalizasyon uygulama
        print("\nX_train_normali:\n",self.X_train,"\nX_test_normali:\n",self.X_test)
   
    def standardizet(self):
        stdsc=StandardScaler()
        self.X_train=stdsc.fit_transform(self.X_train)
        self.X_test_std=stdsc.transform(self.X_test)
        print("\nX_train_std:\n", self.X_train,"\nX_test_std:\n",self.X_test)
   
#    def oznitelik(self):
##oznitelik çok mu
#        ozniteliksayisi= self.verilerim.shape[1]        
#        print("Öznitelik sayısı:",ozniteliksayisi)
#        if(ozniteliksayisi>10):
##öznitelik seçimi            
#            self.X_new = SelectKBest(chi2, k=2).fit_transform(self.X, self.y)
#            print(self.X_new.shape) 
##PCA uygulanırsa        
#            pca = PCA(n_components = 2)
#            self.X_new = pca.fit_transform(self.X)
#            #X_test2 = pca.transform(self.X_test)
#            print( self.X_new.shape) 
   
    def ozniteliksecimi(self):
        #öznitelik seçimi            
        self.X = SelectKBest(chi2, k=2).fit_transform(self.X, self.y)
        print(self.X.shape) 
        
    def oznitelikdonusumu(self):
        pca = PCA(n_components = 2)
        self.X = pca.fit_transform(self.X)
        #self.X_test2 = pca.transform(self.X_test)
        print( self.X.shape) 
        
    
     
    def islemsec(self):
        if(self.comboBox_5.currentIndex()==0):
            print("Seçim yapınız")
        if(self.comboBox_5.currentIndex()==1):
            self.dengesiz()
            self.apply()
            
        if(self.comboBox_5.currentIndex()==2):
            self.normalizet()
        if(self.comboBox_5.currentIndex()==3):
            self.standardizet()
        if(self.comboBox_5.currentIndex()==4):
            self.ozniteliksecimi()
            self.apply()
        if(self.comboBox_5.currentIndex()==5):
            self.oznitelikdonusumu()
            self.apply()    
        self.dinfo()
   
    def toplu(self):
        self.apply()
#toplu öğrenme bagging(Torbalama)
        if(self.comboBox_6.currentIndex()==0):
            self.model=BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            sonuc= self.model.score(self.X_test,self.y_test)
            print("Bagging classifier score",sonuc)
            self.textEdit_13.setText("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))
            self.rocegrisi()
            self.cmatrix()
            self.basari()
            self.modeilbasaridegerlendirmesi()
            self.basarili = "Bagging Classifier Sonuçlandı."
            self.success()
            self.dinfo()
            
#toplu öğrenme boosting(Artırma)
        if(self.comboBox_6.currentIndex()==1):
            self.model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            result=self.model.score(self.X_test,self.y_test)
            print("Boosting classifier score",result)
            self.textEdit_13.setText("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))
            self.rocegrisi()
            self.cmatrix()
            self.basari()
            self.modeilbasaridegerlendirmesi()
            self.basarili = "Gradient Boosting Classifier Sonuçlandı."
            self.success()
   
    def dinfo(self):   
        self.pushButton_5.setEnabled(True)
        self.textEdit_7.setText("")
        self.textEdit_8.setText("")
        self.textEdit_13.setText("")
        self.textEdit_14.setText(" ")
        self.textEdit_15.setText(" ")
        self.textEdit_18.setText("")
        self.textEdit_19.setText("")
        self.textEdit_20.setText("")
        if (self.comboBox_3.currentIndex()==0):
#model insası(SVC)             
             self.model=svc(probability=True)
             self.model.fit(self.X_train,self.y_train)
             self.y_pred = self.model.predict(self.X_test)

        if(self.comboBox_3.currentIndex()==1):   
#model insası(logistic regression)      
             self.model = LogisticRegression(random_state=0)
             self.model.fit(self.X_train,self.y_train)
             self.y_pred = self.model.predict(self.X_test)        
             
#model insası(KNN)                   
        if (self.comboBox_3.currentIndex()==2):
             self.model = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
             self.model.fit(self.X_train,self.y_train)
             self.y_pred = self.model.predict(self.X_test)
         
#model insası(Naive Bayes)  
        if(self.comboBox_3.currentIndex()==3):   
            self.model= GaussianNB()
            self.model.fit(self.X_train,self.y_train)
            self.y_pred = self.model.predict(self.X_test)
           
#model insası(Decision Tree Classification)  
        if(self.comboBox_3.currentIndex()==4):   
            self.model= DecisionTreeClassifier(criterion = 'entropy')
            self.model.fit(self.X_train,self.y_train)
            self.y_pred = self.model.predict(self.X_test)

        self.textEdit_13.setText("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))                               
        self.rocegrisi()
        self.cmatrix()
        self.basari()
        self.modeilbasaridegerlendirmesi()
        self.basarili = "Sınıflandırma İşlemi Yapıldı."
        self.success()
        
    def rocegrisi(self):
        plt.clf()
        self.label_33.setText("ROC Eğrisi Yükleniyor..")
        self.pred_prob = self.model.predict_proba(self.X_test)
        fpr, tpr, thresh = roc_curve(self.y_test, self.pred_prob[:,1], pos_label=1)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr,color='cornflowerblue')
        plt.plot([0,1], [0,1], linestyle='--', color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.annotate('Random Guess',(.5,.48),color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('ROC EĞRİSİ')
        plt.savefig("./rocsiniflandirma.png")
        self.pixmap = QPixmap("./rocsiniflandirma.png")
        self.label_33.setPixmap(self.pixmap)
        plt.show()
        
    def cmatrix(self):
        plt.clf()
        self.label_8.setText("Confusion Matrix Grafiği Yükleniyor..")
#Sınıflandırma için karışıklık matrislerinin gösterimi
        self.cm = confusion_matrix(self.y_test,self.y_pred)
        TN=self.cm[0][0]
        FP=self.cm[0][1]
        FN=self.cm[1][0]
        TP=self.cm[1][1]
        sensitivity=round(float(TP)/(TP+FN)*100,2)
        specificity=round(float(TN)/(FP+TN)*100,2)
        pre=round(float(TP)/(TP+FP)*100,2)
        self.textEdit_7.setText(#"[ TN="+str(TN)+"  FP="+str(FP)+"\n  FN="+str(FN)+"  TP="+str(TP)+" ]"
                                  "Sensitivity: "+str(sensitivity)+"\n\nSpecificity: "+str(specificity)
                                  +"\n\nPre: "+str(pre))
#modelin performansının nasıl değerlendirileceğini gösterir      
        print(classification_report(self.y_test, self.y_pred))
        plot_confusion_matrix(self.model,self.X_test, self.y_test,cmap=plt.cm.Blues)        
        plt.title('Confusion Matrix')
        plt.savefig("./Sınıflandırmatrix.png")
        self.pixmap = QPixmap("./Sınıflandırmatrix.png")
        self.label_8.setPixmap(self.pixmap)

    def basari(self):
        self.basarii=accuracy_score(self.y_test, self.y_pred)
        self.textEdit_8.setText("%.3f" % self.basarii)#virgülden sonrası 3 tane olsun diye(.3f)               
    
    def modeilbasaridegerlendirmesi(self):
        R2=r2_score(self.y_test, self.y_pred)
        print(" R²:",R2)
        self.textEdit_18.setText("%.3f" % R2)
        MAE=mean_absolute_error(self.y_test, self.y_pred )
        print("Mean Absolute Error(MAE)",MAE)
        self.textEdit_19.setText("%.3f" % MAE)
        MSE=mean_squared_error(self.y_test, self.y_pred)
        print("Mean Squared Error(MSE)",MSE)
        self.textEdit_20.setText("%.3f" % MSE)
    
    def ikincisayfayadon(self):
        self.tabWidget.setCurrentIndex(1)
        
    def dorduncusayfayagec(self):
        self.apply()#sınıflandırma işlemleri yapay sinir ağı işlemlerini etkilememsi için
        self.tabWidget.setTabEnabled(3,True)
        self.tabWidget.setCurrentIndex(3)
   
    def ucuncusayfayadon(self):
        self.tabWidget.setCurrentIndex(2)
        
    def searchler(self):
        self.gridSearch()
        self.randSearch()
   
    def gridSearch(self):
         
        self.model=svc(probability=True, random_state=0)
        param_grid={"C":np.arange(2,10,20),
                    "gamma":np.arange(0.1,1,0.2)}
        grid_search=GridSearchCV(self.model,param_grid, n_jobs=4, cv=5)
        grid_search.fit(self.X,self.y)
        grid_search.best_score_
        grid_search.best_params_
        self.textEdit_14.setText("En iyi başarı: "+str(grid_search.best_score_) +"\nEn iyi parametre: "
                                 +str(grid_search.best_params_) )             
       
    def randSearch(self):      
       
        rand_list={"C":stats.uniform(2,10),
                   "gamma":stats.uniform(0.1,1)}
        rand_search=RandomizedSearchCV(self.model,param_distributions=rand_list,n_iter=20,n_jobs=4, cv=5)
        rand_search.fit(self.X,self.y)
        rand_search.best_score_
        rand_search.best_params_
        self.textEdit_15.setText("En iyi başarı: "+str(rand_search.best_score_) 
                                 +"\nEn iyi parametre: "+str(rand_search.best_params_) ) 
# Hata Mesajı
    def error(self):
        msg = QMessageBox()
        msg.setWindowTitle("Uyarı")
        msg.setText(self.hata)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()  
   
    def success(self):
        msg = QMessageBox()
        msg.setWindowTitle("Başarılı")
        msg.setText(self.basarili)
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_() 
   
    def testibaslat(self):
        self.cnn()
        
    def cnn(self):
            self.epoch=self.lineEdit.text()
            print(self.epoch)
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import  Dense
            self.cnnmodel=Sequential()
            self.cnnmodel.add(Dense(12, kernel_initializer="uniform", activation='relu'))
            self.cnnmodel.add(Dense(8, kernel_initializer='uniform', activation='relu'))
            self.cnnmodel.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
            self.cnnmodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
            self.model=self.cnnmodel.fit(self.X_train, self.y_train,
                      epochs=int(self.epoch),
                      batch_size=1,
                      validation_data=(self.X_test, self.y_test)
                     )
            self.y_pred=self.cnnmodel.predict(self.X_test)
            scores=self.cnnmodel.evaluate(self.X_test,self.y_test)
            print("Başarı:%{:.2f}".format(scores[1]*100))
            self.textEdit_21.setText("%{:.2f}".format(scores[1]*100))
            self.basarili = "Model Eğitim İşlemi Başarılı."
            self.success()
            
    def grafikgoster(self):   
            plt.clf()#temizlemek için
            if(self.radioButton_3.isChecked()):          
# Plot training & validation accuracy values        
                plt.figure(figsize=(14,4))
                plt.subplot(1, 2, 1)
                plt.plot(self.model.history['accuracy'])
                plt.plot(self.model.history['val_accuracy'])
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.savefig("./basari.png")
                self.pixmap = QPixmap("./basari.png")
                self.label_29.setPixmap(self.pixmap)
            
            if(self.radioButton_4.isChecked()):              
# Plot training & validation loss values
                plt.figure(figsize=(14,4))
                plt.subplot(1, 2, 1)
                plt.plot(self.model.history['loss'])
                plt.plot(self.model.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.savefig("./loss.png")
                self.pixmap = QPixmap("./loss.png")
                self.label_29.setPixmap(self.pixmap)
                
            if(self.radioButton_5.isChecked()):
                splt.plot_confusion_matrix(self.y_test, self.y_pred.round(), normalize=False)
                plt.savefig("./matrit.png")
                self.pixmap = QPixmap("./matrit.png")
                self.label_29.setPixmap(self.pixmap)
                
            if(self.radioButton_6.isChecked()):#roc eğrisi
                self.pred_prob = self.cnnmodel.predict_proba(self.X_test)    
                fpr, tpr,thresh = roc_curve(self.y_test, self.pred_prob)
                plt.figure(figsize=(8,5))
                plt.plot(fpr, tpr,color='cornflowerblue')
                plt.plot([0, 1], [0, 1], 'k--',color='red')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.annotate('Random Guess',(.5,.48),color='red')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC EĞRİSİ')
                plt.legend(loc="lower right")
                plt.savefig("./roc.png")
                self.pixmap = QPixmap("./roc.png")
                self.label_29.setPixmap(self.pixmap)
                      
            if (self.radioButton_3.isChecked()==False and self.radioButton_4.isChecked()==False and 
                self.radioButton_5.isChecked()==False and self.radioButton_6.isChecked()==False):
                self.hata = "Lütfen bir seçim yapınız!"
                self.error()   
    