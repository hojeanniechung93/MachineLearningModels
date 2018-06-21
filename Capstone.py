# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
ApiCalls = pd.read_table("../Desktop/Data/ApiCalls.tsv")
ApiNames=pd.read_table("../Desktop/Data/ApiNames.tsv")
AppInteractivity = pd.read_table("../Desktop/Data/AppInteractivity.tsv")
DeviceCensus = pd.read_table("../Desktop/Data/DeviceCensus.tsv")
StoreSearch=pd.read_table("../Desktop/Data/StoreSearch.tsv")

#whittle the dataframes for concatnation
AppInter_adj = pd.DataFrame(AppInteractivity[["DateId","nDeviceId","appSessionGuid","AppName","InFocusDurationMS","EngagementDurationMS"]])
SS_adj = pd.DataFrame(StoreSearch[["Category","appName"]])
SS_adj.columns=["Category","AppName"]

#Concat
SourceData = pd.concat([ApiCalls,AppInter_adj,ApiNames],axis=0,ignore_index = True, sort=False)

#Merge
FinalData = pd.merge(SS_adj,SourceData, on=["AppName"])


#Based on previous data, the device will use an x type application tomorrow
#Using Decision Trees
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing

#Create a MultiColumLabelEncoder
class MultiColumnLabelEncoder(preprocessing.LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe):
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = preprocessing.LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = preprocessing.LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return self

    def fit_transform(self, dframe):
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            self.all_labels_ = np.ndarray(shape=self.columns.shape,
                                          dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = preprocessing.LabelEncoder()
                # fit and transform labels in the column
                dframe.loc[:, column] =\
                    le.fit_transform(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
                self.all_labels_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = preprocessing.LabelEncoder()
                dframe.loc[:, column] = le.fit_transform(
                        dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return dframe

    def transform(self, dframe):
        ##Transform labels to normalized encoding.
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[
                    idx].transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .transform(dframe.loc[:, column].values)
        return dframe.loc[:, self.columns].values

    def inverse_transform(self, dframe):
        ###Transform labels back to original encoding.
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        return dframe
    
#Pick out Mixed type data frames
FeaturePicker = pd.DataFrame(FinalData[["Category","DateId","nDeviceId"]])        
# get `object` columns
df_object_columns = FeaturePicker.iloc[:, :].select_dtypes(include=['object']).columns

#instantiate MultiColumnLabelEncoder
mcle = MultiColumnLabelEncoder(columns=df_object_columns)

#fit to FeaturePicker
mcle.fit(FeaturePicker)

#transform the FeaturePicker
mcle.transform(FeaturePicker)
print(FeaturePicker.head())

#inverse data
#mcle.inverse_transform(FinalData)
#X=Feature
#Y=Label

y=FeaturePicker.Category.to_frame()
X_train, X_test, y_train, y_test=train_test_split(FeaturePicker.iloc[:,1:],y,test_size=0.33,random_state=42)


#Sklearn will generate a decision tree for your dataset using an optimized version of the CART algorithm when you run the following code
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

#metrics
import sklearn.metrics as met
y_pred = dtree.predict(X_test)
print(met.classification_report(y_test,y_pred))
#Decode the arrays back into string

#Graph the Tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
