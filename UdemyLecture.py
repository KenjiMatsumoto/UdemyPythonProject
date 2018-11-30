
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


# In[41]:


df2 = df.drop(['y'], axis=1)


# In[42]:


df2

