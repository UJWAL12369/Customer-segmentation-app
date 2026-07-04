import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\srish\Downloads\ml\customer_segmentation.csv")
print(df)

print(df.columns)

print(df.shape)

df=df.dropna()

print(df.shape)

print(df.describe())


print(df['Education'].unique())

df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'],dayfirst=True)

df['Age']=2025-df['Year_Birth']
print(df['Age'])

df['Total_children']=df['Kidhome']+df['Teenhome']


tot_spend=['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']

df['total_spend']=df[tot_spend].sum(axis=1)

print(df['total_spend'])


df['customer_since']=(pd.Timestamp('today')-df['Dt_Customer']).dt.days

print(df['customer_since'])

# EDA

# sns.histplot(df['Age'],bins=30,kde=True)
# plt.title('age distribution')
# plt.show()


# sns.histplot(df['Income'],bins=30,kde=True)
# plt.title('income distribution')
# plt.show()


# sns.histplot(df['total_spend'],bins=30,kde=True)
# plt.title('spending distribution')
# plt.show()



# sns.boxplot(x='Education',y='Income',data=df)
# plt.xticks(rotation=45)

# plt.title("income by education level")
# plt.show()


# sns.boxplot(x='Marital_Status',y='total_spend',data=df)
# plt.xticks(rotation=45)

# plt.title("spending by marital status")
# plt.show()


# corr=df[['Income','Age','Recency','total_spend','NumWebPurchases','NumStorePurchases']].corr()
# print(corr)


# sns.heatmap(corr,annot=True,cmap='coolwarm')
# plt.title("correlation matrix")
# plt.show()


pivot_income=df.pivot_table(values='Income',index='Education',columns='Marital_Status',aggfunc='mean')
print(pivot_income)

# sns.heatmap(pivot_income,annot=True,cmap='coolwarm')
# plt.title("Avg income by education and marital status")
# plt.show()


grp1=df.groupby('Education')['total_spend'].mean().sort_values(ascending=False)
print(grp1)

# grp1.plot(kind='bar',color='skyblue')
# plt.title('avg spending by education')
# plt.ylabel('avg total spending')
# plt.xticks(rotation=45)
# plt.show()


df['accepted any']=df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)

df['accepted any']=df['accepted any'].apply(lambda x:1 if x>0 else 0)
print(df['accepted any'].unique())


grp2=df.groupby('Marital_Status')['accepted any'].mean().sort_values(ascending=False)
print(grp2)

# grp2.plot(kind='bar',color='orange')
# plt.title('campaign acceptance rate by marital status')
# plt.ylabel('acceptance rate')
# plt.xticks(rotation=45)
# plt.show()


bins=[18,30,40,50,60,70,90]

labels=['18-29','30-39','40-49','50-59','60-69','70+']

df['age grp']=pd.cut(df['Age'],bins=bins,labels=labels)

grp3=df.groupby('age grp')['Income'].mean()
print(grp3)
# grp3.plot(kind='barh',color='green')
# plt.title('avg income by each grp')
# plt.xlabel('avg income')

plt.show()

features=['Age','Income','total_spend',"NumWebPurchases",'NumStorePurchases','NumWebVisitsMonth','Recency']

X=df[features].copy()


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
print(x_scaled)


# elbow method
from sklearn.cluster import k_means

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# wcss = []

# for i in range(2,10):

#     kmeans = KMeans(
#         n_clusters=i,
#         random_state=42,
#         n_init=10
#     )

#     kmeans.fit(x_scaled)

#     wcss.append(kmeans.inertia_)

# plt.figure(figsize=(8,5))

# plt.plot(
#     range(2,10),
#     wcss,
#     marker='o'
# )

# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")

# plt.grid(True)

# plt.show()


kmeans=KMeans(n_clusters=6)
df['clusters']=kmeans.fit_predict(x_scaled)

print(df['clusters'])


cluster_summary=df.groupby('clusters')[features].mean()
print(cluster_summary)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_data=pca.fit_transform(x_scaled)
df['pca1'],df['pca2']=pca_data[:,0],pca_data[:,1]


print(df['pca1'])


sns.scatterplot(x='pca1',y='pca2',hue='clusters',data=df,palette='Set1')
plt.title("customer segmentation(pca)")
plt.show()


cluster_names = {
    0:"Premium Customers",
    1:"Budget Customers",
    2:"Regular Customers",
    3:"Inactive Customers",
    4:"Average Customers",
    5:"VIP Customers"
}

df["Segment"] = df["clusters"].map(cluster_names)
sns.countplot(x="Segment",data=df)

plt.xticks(rotation=20)

plt.show()

# just an example
# cluster 0 high income and high spending-premiium custmer
# clsuter 2 high web purchases low store purchases

# we can define this to amke our model more advanced


# saving model

import joblib
joblib.dump(kmeans,"kmeans_model.pkl")
joblib.dump(scaler,'scaler.pkl')
