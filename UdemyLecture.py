
# coding: utf-8

# # データの読み込み

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv('housing.csv')


# In[5]:


df.head(3)


# In[6]:


len(df)


# In[7]:


df.describe()


# # 分布の確認

# In[8]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


sns.distplot(df['x6'], bins=10)


# In[10]:


#相関係数を確認
df.corr()


# In[11]:


# 相関係数を目視で確認できるように
sns.pairplot(df)


# # 入力変数と出力変数の切り分け

# In[12]:


df.head()


# In[13]:


X = df.iloc[:,:-1]


# In[14]:


y = df.iloc[:,-1]


# # モデル構築と検証（sklearnを利用）

# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[16]:


# モデルの宣言
model = LinearRegression()


# In[19]:


# モデルの学習
model.fit(X,y)


# In[20]:


# モデルの検証（決定係数の計算）
model.score(X,y)


# # 訓練データ（train）と検証データ（test）

# In[21]:


# 訓練データと検証データの分割
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=1) # randomstateは乱数のシード


# In[22]:


model.fit(x_train,y_train)


# In[23]:


# 検証（検証データ）
model.score(x_test, y_test)


# In[24]:


# 検証（訓練データ）
model.score(x_train, y_train)


# # 予測値の計算

# In[25]:


x = X.iloc[1,:]
x


# In[26]:


y_pred = model.predict([x])
y_pred


# # モデルの保存

# In[27]:


from sklearn.externals import joblib


# In[28]:


# モデルの保存
joblib.dump(model, 'model.pkl')


# # モデルの読み込み

# In[29]:


model_new = joblib.load('model.pkl')


# In[30]:


model_new.fit(x_train, y_train)


# In[31]:


model_new.score(x_train, y_train)


# In[32]:


model_new.predict([x])


# # パラメータの確認

# In[33]:


# パラメータ確認
model.coef_


# In[34]:


np.set_printoptions(precision=3,suppress=True) # 指数関数での表示を禁止


# In[35]:


model.coef_


# In[36]:


df.head(3)


# # 外れ値除去（３σ法）

# In[44]:


col = 'x6'


# In[45]:


# 平均
mean = df.mean()
# mean


# In[46]:


mean[col]


# In[50]:


# 標準偏差(standard deviation)
sigma = df.std()
# sigma


# In[51]:


sigma[col]


# In[53]:


low = mean[col] -3 * sigma[col]


# In[54]:


high = mean[col] + 3 * sigma[col]


# In[59]:


df2 = df[(df[col] > low) & (df[col] < high)]


# In[60]:


len(df)


# In[61]:


len(df2)


# In[62]:


# 分布の確認
sns.distplot(df[col]) # オリジナル


# In[63]:


sns.distplot(df2[col])


# # 外れ値除去を全列に適用

# In[64]:


cols = df.columns
cols


# In[79]:


import sys
_df = df
for col in cols:
    # 3σ法の上下限値を設定
    low = mean[col] - 3 * sigma[col]
    high = mean[col] + 3 * sigma[col]
    # 条件で絞り込み
    _df = _df[(_df[col] > low) & (_df[col] < high)]


# In[80]:


len(df)


# In[81]:


len(_df)


# # 外れ値とスケーリングを考慮した実装

# In[84]:


X = _df.iloc[:,:-1]
y = _df.iloc[:,-1]


# In[85]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# In[86]:


model = LinearRegression()


# In[87]:


# 学習
model.fit(x_train,y_train)


# In[88]:


# 検証（訓練データ）
model.score(x_train,y_train)


# In[89]:


# 検証（検証データ）
model.score(x_test, y_test)


# # スケーリング

# In[90]:


from sklearn.preprocessing import StandardScaler


# In[91]:


# scalerの宣言
scaler = StandardScaler()


# In[92]:


# scalerの学習　平均と標準偏差を計算
scaler.fit(x_train)


# In[93]:


# scaling
x_train2 = scaler.transform(x_train)
x_test2 = scaler.transform(x_test)


# In[94]:


x_train


# In[97]:


# モデルの宣言
model = LinearRegression()


# In[98]:


model.fit(x_train2,y_train)


# In[99]:


model.score(x_train2, y_train)


# In[103]:


model.score(x_test2, y_test)

