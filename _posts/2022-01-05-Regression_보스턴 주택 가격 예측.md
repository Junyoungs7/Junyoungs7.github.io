## 기본 라이브러리


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# sklearn에서 데이터셋 로딩

from sklearn import datasets

housing = datasets.load_boston()
housing.keys()
```




    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])




```python
data = pd.DataFrame(housing['data'], columns=housing['feature_names'])
target = pd.DataFrame(housing['target'], columns=['Target'])

print(data.shape)
print(target.shape)
```

    (506, 13)
    (506, 1)
    


```python
# 데이터프레임 결합
df = pd.concat([data, target], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    float64
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    float64
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  Target   506 non-null    float64
    dtypes: float64(14)
    memory usage: 55.5 KB
    


```python
# 결측값 확인
df.isnull().sum()
```




    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    Target     0
    dtype: int64




```python
# 상관 관계 분석
df_corr = df.corr()

plt.figure(figsize=(10,10))
sns.set(font_scale=0.8)
sns.heatmap(df_corr, annot=True, cbar=False);
plt.show()
```


    
![png](output_7_0.png)
    



```python
# 변수 간의 상관 관계 분석 - Target 변수와 상관 관계가 높은 순서대로 정리
corr_order = df.corr().loc[:'LSTAT','Target'].abs().sort_values(ascending=False)
corr_order
```




    LSTAT      0.737663
    RM         0.695360
    PTRATIO    0.507787
    INDUS      0.483725
    TAX        0.468536
    NOX        0.427321
    CRIM       0.388305
    RAD        0.381626
    AGE        0.376955
    ZN         0.360445
    B          0.333461
    DIS        0.249929
    CHAS       0.175260
    Name: Target, dtype: float64




```python
# 시각화로 분석할 피처 선택 추출
plot_cols = ['Target', 'LSTAT','RM','PTRATIO','INDUS']
plot_df = df.loc[:,plot_cols]
plot_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>LSTAT</th>
      <th>RM</th>
      <th>PTRATIO</th>
      <th>INDUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.0</td>
      <td>4.98</td>
      <td>6.575</td>
      <td>15.3</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.6</td>
      <td>9.14</td>
      <td>6.421</td>
      <td>17.8</td>
      <td>7.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.7</td>
      <td>4.03</td>
      <td>7.185</td>
      <td>17.8</td>
      <td>7.07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.4</td>
      <td>2.94</td>
      <td>6.998</td>
      <td>18.7</td>
      <td>2.18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.2</td>
      <td>5.33</td>
      <td>7.147</td>
      <td>18.7</td>
      <td>2.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
# regplot으로 선형회귀선 표시
plt.figure(figsize=(10,10))
for idx, col in enumerate(plot_cols[1:]):
    ax1 = plt.subplot(2,2,idx+1)
    sns.regplot(x=col, y=plot_cols[0], data=plot_df, ax=ax1)
plt.show()
```


    
![png](output_10_0.png)
    



```python
# target 데이터 분포
sns.displot(x='Target', kind='hist', data=df)
plt.show()
```


    
![png](output_11_0.png)
    



```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df_scaled = df.iloc[:,:-1]
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

df.iloc[:,:-1] = df_scaled[:,:]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.18</td>
      <td>0.067815</td>
      <td>0.0</td>
      <td>0.314815</td>
      <td>0.577505</td>
      <td>0.641607</td>
      <td>0.269203</td>
      <td>0.000000</td>
      <td>0.208015</td>
      <td>0.287234</td>
      <td>1.000000</td>
      <td>0.089680</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000236</td>
      <td>0.00</td>
      <td>0.242302</td>
      <td>0.0</td>
      <td>0.172840</td>
      <td>0.547998</td>
      <td>0.782698</td>
      <td>0.348962</td>
      <td>0.043478</td>
      <td>0.104962</td>
      <td>0.553191</td>
      <td>1.000000</td>
      <td>0.204470</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000236</td>
      <td>0.00</td>
      <td>0.242302</td>
      <td>0.0</td>
      <td>0.172840</td>
      <td>0.694386</td>
      <td>0.599382</td>
      <td>0.348962</td>
      <td>0.043478</td>
      <td>0.104962</td>
      <td>0.553191</td>
      <td>0.989737</td>
      <td>0.063466</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000293</td>
      <td>0.00</td>
      <td>0.063050</td>
      <td>0.0</td>
      <td>0.150206</td>
      <td>0.658555</td>
      <td>0.441813</td>
      <td>0.448545</td>
      <td>0.086957</td>
      <td>0.066794</td>
      <td>0.648936</td>
      <td>0.994276</td>
      <td>0.033389</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000705</td>
      <td>0.00</td>
      <td>0.063050</td>
      <td>0.0</td>
      <td>0.150206</td>
      <td>0.687105</td>
      <td>0.528321</td>
      <td>0.448545</td>
      <td>0.086957</td>
      <td>0.066794</td>
      <td>0.648936</td>
      <td>1.000000</td>
      <td>0.099338</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 학습 - 테스트 데이터셋 분할
from sklearn.model_selection import train_test_split
x_data = df.loc[:,['LSTAT','RM']]
y_data = df.loc[:,'Target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=12)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

    (404, 2) (404,)
    (102, 2) (102,)
    


```python
# 선형 회귀 모형
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

print("회귀계수(기울기):",np.round(lr.coef_, 1))
print("상수항(절편):",np.round(lr.intercept_, 1))
```

    회귀계수(기울기): [-23.2  25.4]
    상수항(절편): 16.3
    

LSTAT이 클수록 Target값은 작아진다. 반면 RM이 클수록 Target값은 커진다.


```python
# 예측
y_test_pred = lr.predict(x_test)

# 예측값, 실제값의 분포
plt.figure(figsize=(10,5))
plt.scatter(x_test['LSTAT'], y_test, label='y_test')
plt.scatter(x_test['LSTAT'], y_test_pred, c='r', label='y_pred')
plt.legend(loc='best')
plt.show()
```


    
![png](output_16_0.png)
    



```python
# 평가
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(x_train)

train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:30.8042
    Test MSE:29.5065
    


```python
# cross_val_score 함수
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
mse_scores = -1*cross_val_score(lr, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

print("개별 Fold의 MSE:", np.round(mse_scores,4))
print("평균 MSE:%.4f"% np.mean(mse_scores))
```

    개별 Fold의 MSE: [31.465  34.668  28.9147 29.3535 34.6627]
    평균 MSE:31.8128
    

## 과대적합 회피


```python
# 2차 다항식 변환
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
x_train_poly = pf.fit_transform(x_train)
print("원본 학습 데이터셋: ",x_train.shape)
print("2차 다항식 변환 데이터셋:",x_train_poly.shape)
```

    원본 학습 데이터셋:  (404, 2)
    2차 다항식 변환 데이터셋: (404, 6)
    


```python
# 2차 다항식 변환 데이터셋으로 선형 회귀 모형 학습
lr = LinearRegression()
lr.fit(x_train_poly, y_train)

y_train_pred = lr.predict(x_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)

x_test_poly = pf.fit_transform(x_test)
y_test_pred = lr.predict(x_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:21.5463
    Test MSE:16.7954
    


```python
# 15차 다항식 변환 데이터셋으로 선형 회귀 모형 학습
pf = PolynomialFeatures(degree=15)
x_train_poly = pf.fit_transform(x_train)

lr = LinearRegression()
lr.fit(x_train_poly, y_train)

y_train_pred = lr.predict(x_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)

x_test_poly = pf.fit_transform(x_test)
y_test_pred = lr.predict(x_test_poly)
test_mes = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mes)
```

    Train MSE:11.2567
    Test MSE:88160763423636.5469
    
15차 다항식으로 변환해 선형 회귀 모형 학습 진행, train mse는 11로 감소했지만 새로운 test 데이터로 학습한 test mse는 급격하게 증가 : 이로 봤을때 overfitting 발생으로 신규 데이터에 대한 예측력 상실

```python
# 다항식 차수에 따른 모델 적합도 변화
plt.figure(figsize=(15,5))
for n, deg in enumerate([1,2,15]):
    ax1 = plt.subplot(1,3,n+1)
    # plt.axis('off')
    # degree별 다항 회귀 모형 적용
    pf = PolynomialFeatures(degree = deg)
    x_train_poly = pf.fit_transform(x_train.loc[:,['LSTAT']])
    x_test_poly = pf.fit_transform(x_test.loc[:,['LSTAT']])
    lr = LinearRegression()
    lr.fit(x_train_poly, y_train)
    y_test_pred = lr.predict(x_test_poly)
    # 실제값 분포
    plt.scatter(x_test.loc[:,['LSTAT']], y_test, label='Targets')
    # 예측값 분포
    plt.scatter(x_test.loc[:,['LSTAT']], y_test_pred, label='Predictions')
    # 제목 표시
    plt.title("Degree %d"% deg)
    # 범례 표시
    plt.legend()
plt.show()
```


    
![png](output_24_0.png)
    


모델의 복잡도를 높이면 예측율을 높일 수 있지만 너무 높이면 과대적합이 생김.

## 규제


```python
# Ridge(L2 규제)
from sklearn.linear_model import Ridge
rdg = Ridge(alpha=2.5)
rdg.fit(x_train_poly, y_train)

y_train_pred = rdg.predict(x_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)
y_test_pred = rdg.predict(x_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:35.9484
    Test MSE:42.0011
    


```python
# Lasso(L1 규제)
from sklearn.linear_model import Lasso
las = Lasso(alpha=0.05)
las.fit(x_train_poly, y_train)

y_train_pred = las.predict(x_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)
y_test_pred = las.predict(x_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:32.3204
    Test MSE:37.7103
    


```python
# ElasticNet(L2/L1 규제)
from sklearn.linear_model import ElasticNet
ela = ElasticNet(alpha=0.01, l1_ratio=0.7)
ela.fit(x_train_poly, y_train)

y_train_pred = ela.predict(x_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)
y_test_pred = ela.predict(x_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:33.7551
    Test MSE:39.4968
    


```python
# 의사결정나무
from sklearn.tree import DecisionTreeRegressor
dtr= DecisionTreeRegressor(max_depth=3, random_state=12)
dtr.fit(x_train, y_train)

y_train_pred = dtr.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)
y_test_pred = dtr.predict(x_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:18.8029
    Test MSE:17.9065
    


```python
# 랜덤 포레스트 - 하나의 트리를 사용하는 의사결정나무에 비하여, 여러 개의 트리 모델이 예측한 값을 종합
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3, random_state=12)
rfr.fit(x_train, y_train)

y_train_pred = rfr.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)
y_test_pred = rfr.predict(x_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:16.0201
    Test MSE:17.7751
    


```python
# XGBoost
from xgboost import XGBRegressor
xgbr = XGBRegressor(objective='reg:squarederror',max_depth=3, random_state=12)
xgbr.fit(x_train, y_train)

y_train_pred = xgbr.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f"% train_mse)
y_test_pred = xgbr.predict(x_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f"% test_mse)
```

    Train MSE:3.5176
    Test MSE:20.2060
    

데이터의 개수가 작기 때문에 XGBoost와 같이 복잡도가 높은 알고리즘이 쉽게 과대적합될 위험성이 있다


```python

```
