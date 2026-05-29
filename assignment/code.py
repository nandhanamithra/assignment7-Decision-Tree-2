import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class5\assignment2\data.csv") #Load the dataset

x = data[["tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"]]
y = data["fail"]

#Train the Decision Tree model
model=DecisionTreeClassifier(criterion="entropy")
model.fit(x,y)

#Predict whether the following new machine condition will result in failure or not
sample = [[4, 5, 3, 6, 1, 45, 5, 1]]
prediction = model.predict(sample)

#Predict: fail = 1 or 0
if prediction[0] == 1:
    print("Machine will Fail")
else:
    print("Machine will not Fail")
    
#plot decision tree
plt.figure(figsize=(8,10))
tree.plot_tree(
    model,
    feature_names=[
        "tempMode",
        "AQ",
        "USS",
        "CS",
        "VOC",
        "RP",
        "IP",
        "Temperature"
    ],
    class_names=["No Failure", "Failure"],
)
plt.title("decision tree")
plt.show()
