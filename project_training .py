import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("https://raw.githubusercontent.com/20AhmedRamadan04/Data2/main/Data2.csv")

print("data is: \n",data)
print("-_"*20)

print("data of dimention : \n",data.shape)
print("-_"*20)

print("Names of column : \n",data.head(0))
print("-_"*20)

print("type of each columns :\n",data.dtypes)
print("-_"*20)

model = preprocessing.MinMaxScaler()
scaledData = model.fit_transform(data.values)
scaledData = pd.DataFrame(scaledData , columns=data.columns)

corre = scaledData.corr()
print(f" correlation data is: \n {corre}")

heat_data = sns.heatmap(corre)
plt.show()

sns.catplot(x='Rate', hue='Apr', data=scaledData, kind='strip')
plt.show()

plt.scatter(x='Rate' , y='Apr' , data=scaledData)
plt.show()


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
print('x Values = \n',x)
print("_-"*20)

print('y value = \n',y)
print("_-"*20)

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size = 0.3)
print("x_train = \n", x_train)
print("_-"*20)
print("x_test = \n", x_test)
print("_-"*20)
print("y_train = \n", y_train)
print("_-"*20)
print("y_test = \n", y_test)
print("_-"*20)


my_model = LinearRegression()
my_model.fit(x_train, y_train)

a = my_model.intercept_
b = my_model.coef_

print(f" interception = {a}")
print("_-"*20)
print(f" coef_ = {b}")
print("_-"*20)

my_predicted  = 20
y_per = (a+my_predicted) + b
print(f" y_per = {y_per}")
print("_-"*20)

accuracy = my_model.score(x_test, y_test)
print(f" accuracy is : {accuracy}")
print("_-"*20)

poly_regs= PolynomialFeatures(degree= 2)
x_poly= poly_regs.fit_transform(x_train)
lin_reg =LinearRegression()
lin_reg.fit(x_poly, y_train)
print('a0 = ',lin_reg.intercept_)
print("_-"*20)

print('0, a1, a2 = ',lin_reg.coef_)
