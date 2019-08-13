import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
from statsmodels.graphics.factorplots import interaction_plot

sns.set(style="whitegrid")

df = pd.read_excel('Dados_TCC.xlsx', sheet_name='Página1')

df = df.drop(['StdOrder','RunOrder','CenterPt','Blocks'],axis=1)

# fatores
# classe do aço, direção de corte, equipamento de teste, frequência

df['Direção de laminação'] = df['Direção de laminação'].astype('category')
df['Equipamento de teste'] = df['Equipamento de teste'].astype('category')
df['Frequência'] = df['Frequência'].astype('category')
df['Classe do aço'] = df['Classe do aço'].astype('category')

print(df.head())
print(df.info())
print(df.describe())

# boxplot
# sns.catplot(x='Equipamento de teste',y='Perdas Totais 0,5T',data=df,kind='box')
# sns.catplot(x='Equipamento de teste',y='Perdas Totais 1,0T',data=df,kind='box')

# sns.catplot(x='Equipamento de teste',y='Perdas Totais 1,5T',data=df,hue='Direção de laminação',kind='box')
# sns.catplot(x='Equipamento de teste',y='Perdas Totais 1,5T',data=df,hue='Frequência',kind='box')
# sns.catplot(x='Equipamento de teste',y='Perdas Totais 1,5T',data=df,hue='Classe do aço',kind='box')

# sns.catplot(x='Equipamento de teste',y='Permeabilidade 0,5T',data=df,kind='box')
# sns.catplot(x='Equipamento de teste',y='Permeabilidade 1,0T',data=df,kind='box')
# sns.catplot(x='Equipamento de teste',y='Permeabilidade 1,5T',data=df,kind='box')

# df_coded = df.copy()
# df_coded = pd.get_dummies(df_coded,columns=['Direção de laminação','Equipamento de teste','Frequência','Classe do aço'],prefix=['direcao','equipamento','freq','classe'])
# print(df_coded.head())
# print(df_coded.info())

# modelo usando lm e dividindo dataset em train e test (resultado: dataset muito pequeno pra isso)

# y = df_coded['Perdas Totais 1,5T']
# X = df_coded.drop(['Perdas Totais 0,5T','Perdas Totais 1,0T','Permeabilidade 0,5T','Permeabilidade 1,0T','Permeabilidade 1,5T'],axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# lm = LinearRegression()
# lm.fit(X_train,y_train)

# coefficients
# coefficients = pd.DataFrame(lm.coef_,X.columns)
# coefficients.columns = ['Coeffecient']
# print(coefficients)

# predictions = lm.predict(X_test)

# fig = plt.figure()
# plt.scatter(y_test,predictions)
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')

# evaluating model
# print('MAE:', metrics.mean_absolute_error(y_test, predictions))
# print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# residuals
# fig2 = plt.figure()
# sns.distplot((y_test-predictions),bins=50)

# modelo usando statsmodel

A = df['Direção de laminação']
B = df['Equipamento de teste']
C = df['Frequência']
D = df['Classe do aço']

perdas1 = df['Perdas Totais 0,5T']
perdas2 = df['Perdas Totais 1,0T']
perdas3 = df['Perdas Totais 1,5T']
permeabilidade1 = df['Permeabilidade 0,5T']
permeabilidade2 = df['Permeabilidade 1,0T']
permeabilidade3 = df['Permeabilidade 1,5T']

# Estimando primeiro modelo pra Perdas totais a 0,5 T
formula1 = 'perdas1 ~ A + B + C + D + A:B + A:C + A:D + B:C + B:D + C:D'
modelo1 = ols(formula1, df).fit()
print(modelo1.summary())

# Gráfico dos resíduos do modelo
fig1 = plt.figure(1)
plt.subplot(121)
stats.probplot(modelo1.resid, plot=plt)
plt.title('Variável original')

# Teste de normalidade
test_ad1 = sms.normal_ad(modelo1.resid)
print("AD: {}, p valor: {}".format(test_ad1[0],test_ad1[1]))

# transformacao box cox
boxcox1 = stats.boxcox(perdas1)
lambda1 = boxcox1[1]
lambda1 = round(boxcox1[1])
perdas1_bc = stats.boxcox(perdas1,lmbda=lambda1)
formula1_bc = 'perdas1_bc ~ A + B + C + D + A:B + A:C + A:D + B:C + B:D + C:D'
modelo1_bc = ols(formula1_bc, df).fit()
print(modelo1_bc.summary())

# resíduos e teste AD pra variável transformada
res1_bc = modelo1_bc.resid
test1_bc = sms.normal_ad(res1_bc)
print("AD: {}, p valor: {}".format(test1_bc[0],test1_bc[1]))

plt.subplot(122)
stats.probplot(res1_bc, plot=plt)
plt.title('Variável transformada')

# anova
anova = sm.stats.anova_lm(modelo1_bc)
pvalues = anova['PR(>F)']
print(anova)
print(pvalues)

# significant factores (p < 5%)
sig_factor = pvalues.loc[pvalues < 0.05]
print(sig_factor)

# grafico residuos x fitted value
perdas1_predict = modelo1_bc.predict()
fig2 = plt.figure(2)
plt.title('Gráfico dos resíduos')
plt.scatter(perdas1_predict,res1_bc)
plt.axhline(color='red')
plt.xlabel("Valor previsto pelo modelo")
plt.ylabel("Resíduos")

# coeficientes do modelo
n_coef1 = ["A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD"]
coef1 = modelo1_bc.params
coef1 = coef1.drop(labels=['Intercept'])
print(coef1)

coef1_values = coef1.values
coef1_names = coef1.index
coef1_s = coef1.sort_values()

# bar plot dos efeitos e interações
fig3 = plt.figure(3)
plt.title('Efeitos dos fatores e interações')
sns.barplot(x=coef1_values,y=n_coef1,data=df)
#plt.barh(n_coef1, coef1_values)
#plt.axvline(color='black')

# interaction plots

# fig, ax = plt.subplots(figsize=(6, 6))
# plt.title('Interação direção de corte com frequência')
# fig = interaction_plot(x=A.astype('category'), trace=C.astype('category'), response=perdas1_bc, colors=['red', 'blue'], markers=['D', '^'], ms=10, ax=ax)

# fig, ax = plt.subplots(figsize=(6, 6))
# plt.title('Interação direção de corte com classe do aço')
# fig = interaction_plot(x=A.astype('category'), trace=D.astype('category'), response=perdas1_bc, colors=['red', 'blue'], markers=['D', '^'], ms=10, ax=ax)

# fig, ax = plt.subplots(figsize=(6, 6))
# plt.title('Interação frequência com classe do aço')
# fig = interaction_plot(x=C.astype('category'), trace=D.astype('category'), response=perdas1_bc, colors=['red', 'blue'], markers=['D', '^'], ms=10, ax=ax)

fig4 = plt.figure(4,figsize=(16, 5))
plt.subplot(131)
sns.boxplot(x=B,y=perdas1,hue=A)
plt.subplot(132)
sns.boxplot(x=B,y=perdas1,hue=C)
plt.subplot(133)
sns.boxplot(x=B,y=perdas1,hue=D)

# bar plots
# fig4 = plt.figure(4,figsize=(10, 8))
# plt.subplot(221)
# sns.barplot(x=A,y=perdas1,data=df)
# plt.subplot(222)
# sns.barplot(x=B,y=perdas1,data=df)
# plt.subplot(223)
# sns.barplot(x=C,y=perdas1,data=df)
# plt.subplot(224)
# sns.barplot(x=D,y=perdas1,data=df)
# fig4.suptitle('Perdas totais 0,5T')

# fig5 = plt.figure(5,figsize=(10, 8))
# plt.subplot(221)
# sns.boxplot(x=A,y=perdas1,data=df)
# plt.subplot(222)
# sns.boxplot(x=B,y=perdas1,data=df)
# plt.subplot(223)
# sns.boxplot(x=C,y=perdas1,data=df)
# plt.subplot(224)
# sns.boxplot(x=D,y=perdas1,data=df)
# fig5.suptitle('Perdas totais 0,5T')

# fig5 = plt.figure(5,figsize=(10, 8))
# plt.subplot(221)
# sns.barplot(x=A,y=perdas2,data=df)
# plt.subplot(222)
# sns.barplot(x=B,y=perdas2,data=df)
# plt.subplot(223)
# sns.barplot(x=C,y=perdas2,data=df)
# plt.subplot(224)
# sns.barplot(x=D,y=perdas2,data=df)
# fig5.suptitle('Perdas totais 1,0T')

# fig6 = plt.figure(6,figsize=(10, 8))
# plt.subplot(221)
# sns.barplot(x=A,y=perdas3,data=df)
# plt.subplot(222)
# sns.barplot(x=B,y=perdas3,data=df)
# plt.subplot(223)
# sns.barplot(x=C,y=perdas3,data=df)
# plt.subplot(224)
# sns.barplot(x=D,y=perdas3,data=df)
# fig6.suptitle('Perdas totais 1,5T')

# fig7 = plt.figure(7,figsize=(10, 8))
# plt.subplot(221)
# sns.barplot(x=A,y=permeabilidade1,data=df)
# plt.subplot(222)
# sns.barplot(x=B,y=permeabilidade1,data=df)
# plt.subplot(223)
# sns.barplot(x=C,y=permeabilidade1,data=df)
# plt.subplot(224)
# sns.barplot(x=D,y=permeabilidade1,data=df)
# fig7.suptitle('Permeabilidade 0,5T')

# fig8 = plt.figure(8,figsize=(10, 8))
# plt.subplot(221)
# sns.barplot(x=A,y=permeabilidade2,data=df)
# plt.subplot(222)
# sns.barplot(x=B,y=permeabilidade2,data=df)
# plt.subplot(223)
# sns.barplot(x=C,y=permeabilidade2,data=df)
# plt.subplot(224)
# sns.barplot(x=D,y=permeabilidade2,data=df)
# fig8.suptitle('Permeabilidade 1,0T')

# fig9 = plt.figure(9,figsize=(10, 8))
# plt.subplot(221)
# sns.barplot(x=A,y=permeabilidade3,data=df)
# plt.subplot(222)
# sns.barplot(x=B,y=permeabilidade3,data=df)
# plt.subplot(223)
# sns.barplot(x=C,y=permeabilidade3,data=df)
# plt.subplot(224)
# sns.barplot(x=D,y=permeabilidade3,data=df)
# fig6.suptitle('Permeabilidade 1,5T')



plt.show()