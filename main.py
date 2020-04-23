# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np

from pandasmodel import PandasModel
from Creditcard import project
task=project()


import pickle 
from sklearn.externals import joblib


class Ui_CreditcardFraud(object):
    def setupUi(self, CreditcardFraud):
        CreditcardFraud.setObjectName("CreditcardFraud")
        CreditcardFraud.resize(546, 509)
        self.centralwidget = QtWidgets.QWidget(CreditcardFraud)
        self.centralwidget.setObjectName("centralwidget")
        
        
        self.load_data = QtWidgets.QPushButton(self.centralwidget)
        self.load_data.setGeometry(QtCore.QRect(20, 90, 151, 28))
        self.load_data.setObjectName("load_data")
        self.load_data.clicked.connect(self.loadFile)
        
        self.show_result = QtWidgets.QPushButton(self.centralwidget)
        self.show_result.setGeometry(QtCore.QRect(220, 140, 93, 28))
        self.show_result.setObjectName("show_result")
        self.show_result.clicked.connect(self.showresult)
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 30, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(True)
        font.setUnderline(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(125, 460, 400, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(20)
        self.label1.setFont(font)
        self.label1.setObjectName("label1")
        
        
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(192, 90, 331, 22))
        self.lineEdit.setObjectName("lineEdit")
 
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(20, 190, 511, 261))
        self.tableView.setObjectName("tableView")
        
        CreditcardFraud.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(CreditcardFraud)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 546, 26))
        self.menubar.setObjectName("menubar")
        CreditcardFraud.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(CreditcardFraud)
        self.statusbar.setObjectName("statusbar")
        CreditcardFraud.setStatusBar(self.statusbar)

        self.retranslateUi(CreditcardFraud)
        QtCore.QMetaObject.connectSlotsByName(CreditcardFraud)

    def loadFile(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName()
        self.lineEdit.setText(fileName)
        self.df = pd.read_csv(fileName)
       
        
    # Here we have two different ways to show result after pickle the model the first we load the model directly in same file and use it 
    # and the second way to call a function from CreditCard.py class file which do the whole process and call it with the result after give the data    
    
    
    def showresult(self):
        mj=pickle.load(open("final_model.sav","rb"))
        x_test=self.df.to_numpy()
        y_pred=mj.predict(x_test)
        y_pred=pd.DataFrame(y_pred)
        self.df["class"]=y_pred
        self.model = PandasModel(self.df)
        self.tableView.setModel(self.model)
        
        
    #def showresult(self):
        #task=project(self.df)
        #data=task.get_label()
        #self.df["class"]=data
        #self.model = PandasModel(self.df)
        #self.tableView.setModel(self.model)
        
        
        
        
    def retranslateUi(self, CreditcardFraud):
        _translate = QtCore.QCoreApplication.translate
        CreditcardFraud.setWindowTitle(_translate("CreditcardFraud", "Creditcard Fraud Detection"))
        self.load_data.setText(_translate("CreditcardFraud", "Load a CSV File"))
        self.show_result.setText(_translate("CreditcardFraud", "Show Result"))
        self.label.setText(_translate("CreditcardFraud", "CreditCard Fraud Detection"))
        self.label1.setText(_translate("CreditcardFraud", "0 for Normal case , 1 for Fraud case"))
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CreditcardFraud = QtWidgets.QMainWindow()
    ui = Ui_CreditcardFraud()
    ui.setupUi(CreditcardFraud)
    CreditcardFraud.show()
    sys.exit(app.exec_())
