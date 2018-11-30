
# coding: utf-8

# # データの読み込み

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv('housing.csv')


# In[6]:


df.head(3)


# In[7]:


len(df)


# In[8]:


df.describe()


# # 分布の確認

# In[11]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


sns.distplot(df['x6'], bins=10)


# In[17]:


#相関係数を確認
df.corr()


# In[18]:


# 相関係数を目視で確認できるように
sns.pairplot(df)


# # 入力変数と出力変数の切り分け

# In[19]:


df.head()


# In[25]:


X = df.iloc[:,:-1]


# In[26]:


y = df.iloc[:,-1]


# # モデル構築と検証（sklearnを利用）

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[28]:


# モデルの宣言
model = LinearRegression()


# In[30]:


# モデルの学習
model.fit(X,y)


# In[33]:


# モデルの検証（決定係数の計算）
model.score(X,y)


# # 訓練データ（train）と検証データ（test）

# In[41]:


# 訓練データと検証データの分割
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=1) # randomstateは乱数のシード


# In[42]:


model.fit(x_train,y_train)


# In[43]:


# 検証（検証データ）
model.score(x_test, y_test)


# In[44]:


# 検証（訓練データ）
model.score(x_train, y_train)


# # 予測値の計算

# In[50]:


x = X.iloc[1,:]
x


# In[51]:


y_pred = model.predict([x])
y_pred


# # モデルの保存

# In[52]:


from sklearn.externals import joblib


# In[53]:


# モデルの保存
joblib.dump(model, 'model.pkl')


# # モデルの読み込み

# In[54]:


model_new = joblib.load('model.pkl')


# In[56]:


model_new.fit(x_train, y_train)


# In[57]:


model_new.score(x_train, y_train)


# In[58]:


model_new.predict([x])


# # パラメータの確認

# In[61]:


# パラメータ確認
model.coef_


# In[60]:


np.set_printoptions(precision=3,suppress=True) # 指数関数での表示を禁止


# In[62]:


model.coef_


# In[63]:


df.head()

