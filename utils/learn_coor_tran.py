from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

x=[[21],[35],[46],[56],[74],[89],[105],[132],[143],[148],[174],[187]]

y=[[130,50],[120,50],[115,52],[110,55],[100,60],[95,59],[90,58],[80,58],[78,56],[76,55],[65,53],[60,52]]

assert len(x)==len(y)

poly=preprocessing.PolynomialFeatures(degree=2)
x_p=poly.fit_transform(x)
model=LinearRegression()
model.fit(x_p,y)
# print(model.predict(x_p))
print("得分",model.score(x_p,y))
from sklearn.externals import joblib
joblib.dump(model,'model.m')

m=joblib.load('model.m')
print(m.predict([x_p[0]]))