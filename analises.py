import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
from statsmodels.graphics.factorplots import interaction_plot

df = pd.read_excel('Dados_TCC.xlsx', sheet_name='Página1')
#print(df.describe())

df = df.drop(columns="StdOrder")
df = df.drop(columns="RunOrder")
df = df.drop(columns="CenterPt")
df = df.drop(columns="Blocks")
#print(df.head())

# 4 fatores, 2 níveis
n_factors = 4
n_levels = 2

# fatores
# classe do aço, direção de corte, equipamento de teste, frequência

factors = [df.iloc[:,i].astype('category') for i in range(0,n_factors)]

A = df.iloc[:,0].astype('category')
B = df.iloc[:,1].astype('category')
C = df.iloc[:,2].astype('category')
D = df.iloc[:,3].astype('category')

# variáveis resposta
n_output = 6

outputs = [df.iloc[:,n_factors+i] for i in range(0,n_output)]

perdas1 = df['Perdas Totais 0,5T']
perdas2 = df['Perdas Totais 1,0T']
perdas3 = df['Perdas Totais 1,5T']
permeabilidade1 = df['Permeabilidade 0,5T']
permeabilidade2 = df['Permeabilidade 1,0T']
permeabilidade3 = df['Permeabilidade 1,5T']

outputs_bc = [df.iloc[:,n_factors+i] for i in range(0,n_output)]

for i in range(0,n_output):
	formula = 'outputs[i] ~ A + B + C + D + A:B + A:C + A:D + B:C + B:D + C:D'
	modelo = ols(formula, df).fit()
	print(modelo.summary())
	res = modelo.resid
	test = sms.normal_ad(res)
	pvalue = test[1]
	
	if (pvalue < 0.05):
		# nao aderencia a normalidade, transformacao box cox
		boxcox = stats.boxcox(outputs[i])
		lambda_bc = round(boxcox[1])
		outputs_bc[i] = stats.boxcox(outputs[i],lmbda=lambda_bc)
	else:
		outputs_bc[i] = np.zeros([len(outputs[i])])


plt.show()