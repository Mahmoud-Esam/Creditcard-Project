import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', DeprecationWarning)
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

import pickle
from sklearn.externals import joblib


class project:
    
       
    def read_data(self):
        self.data=pd.read_csv('creditcard.csv')
    
    
    def before_cleaning(self):
       not_clean=pd.read_csv('creditcard.csv')
       not_cleaned=not_clean.head()
       return not_cleaned
        
        
    def show_data(self):
        print(self.data.head())
        return self.data.head()
        
        
    def shape_data(self):
        print(self.data.shape)
        
        
    def show_info(self):
        print(self.data.info())
        
        
    def show_describe(self):
        print(self.data.describe())
        
        
    def check_null(self):
         print('The Numbers of NULL values in each column is :','\n\n',self.data.isnull().sum(),'\n\n')
        
        
    def check_fraud_normal(self):
        df=self.data.copy()
        fraud= df[df["Class"]==1]
        normal= df[df["Class"]==0]
        print("The number of Fraud cases : {}".format(len(fraud)))
        print("The number of Normal cases : {}".format(len(normal)))
        
        
    def apply_smote_tomek(self):
        df=self.data.copy()
        x=df.drop(["Class"],axis=1).values
        y=df["Class"].values
        smk=SMOTETomek(random_state=39)
        x_sm,y_sm=smk.fit_sample(x,y)
        print(x_sm.shape)
        print(y_sm.shape)
        self.y=y
        self.x=x
        self.x_sm=x_sm
        self.y_sm=y_sm
        #print(self.data.columns)
    
    def counter(self):
        print("the orignal data is : {}".format(Counter(self.y)))
        print("the New data is {}".format(Counter(self.y_sm)))
        
        
    def concat_data(self):
        nd=pd.concat([pd.DataFrame(self.x_sm), pd.DataFrame(self.y_sm)], axis=1)
        nd.columns=self.data.columns
        print(nd.shape)
        self.data=nd
        #print(self.data.shape)
        #print(self.data.columns)
  
    def split_train_test(self):
        X=self.data.drop("Class",axis=1).values
        Y=self.data["Class"].values
        #print(x.shape)
        #print(y.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.25, random_state=2)
        
       
    def models(self,model_name):
        if model_name=='DecisionTree':
            self.model=DecisionTreeClassifier()
      
        elif(model_name=='RandomForest'):
            self.model=RandomForestClassifier()

            self.model=self.model.fit(self.X_train,self.y_train)
        
        
    def evaluate_models(self):
        
        model_prediction=self.model.predict(self.X_test)
        print('Accuracy Score for Model You have choosed : ',
             round(accuracy_score(self.y_test,model_prediction),3)*100,'%')
        
        accuracy  =  round(accuracy_score(self.y_test,model_prediction),3)*100
        print('Confusion Matrix for Model You have choosed: ','....','\n',
             confusion_matrix(self.y_test,model_prediction))
        cm=confusion_matrix(self.y_test,model_prediction)
        
        print('Classification  Report for Model You have choosed: ','....','\n'
             ,classification_report(self.y_test,model_prediction))
        f1=f1_score(self.y_test,model_prediction)
        
        return accuracy, cm, f1    
        
            
    def __init__(self, data = pd.DataFrame(), parent=None): 
        self._data = data
        
        
    def get_label(self):
        mj=pickle.load(open("final_model.sav","rb"))
        x_test=self._data.to_numpy()
        pred=mj.predict(x_test)
        pred=pd.DataFrame(pred)
        return pred
        
   

       
        
if __name__=='__main__':
    test=project()
    test.read_data()
    test.show_data()
    test.shape_data()
    test.show_info()
    test.show_describe()
    test.check_null()
    test.check_fraud_normal()
    test.apply_smote_tomek()
    test.counter()
    test.concat_data()
    test.split_train_test()
    test.models("RandomForest")
    test.evaluate_models()
  
    
    
    
    
    
    