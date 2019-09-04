import time
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
def sacarOutliers(data,cantAmbientes,barrio):
	dfFiltrado = (data.ix[(data.place_name == barrio) & (data.rooms == cantAmbientes)])	
	media=dfFiltrado.loc[:,"price"].mean()
	desvioStandar=dfFiltrado.loc[:,"price"].std()
	dfFiltrado2=(dfFiltrado.ix[(dfFiltrado.price > (media-(3*desvioStandar))) & (dfFiltrado.price < (media+(3*desvioStandar)))])
	return (dfFiltrado2)

def estimarNSE(data,rooms):
	dic={}
	dicOrdenado={}
	df = (data.ix[(data.state_name == 'Capital Federal') & (data.rooms == 3)])
	dfNombres=df.groupby('place_name').groups
	for x in dfNombres:
		dfBarrio=sacarOutliers(data,rooms,x)
		i=dfBarrio.loc[:,"price"].mean()		
		dic[x]=i
	lista=sorted((value,key) for (key,value) in dic.items())
	contador=1
	for x in lista:
		dicOrdenado[x[1]]=contador
		contador=contador+1
	return dicOrdenado


data = read_csv('properati-AR-2018-02-01-properties-sell.csv')



#1
df = (data.ix[(data.state_name == 'Capital Federal') & (data.rooms >= 1) & (data.rooms <= 8)])
#2

print (df)
df['place_name'].replace('', np.nan, inplace=True)
df.dropna(subset=['place_name'], inplace=True)
df['rooms'].replace('', np.nan, inplace=True)
df.dropna(subset=['rooms'], inplace=True)
df['surface_total_in_m2'].replace('', np.nan, inplace=True)
df.dropna(subset=['surface_total_in_m2'], inplace=True)
df['price'].replace('', np.nan, inplace=True)
df.dropna(subset=['price'], inplace=True)


#3a
dfNombres=df.groupby('place_name').groups
dic=estimarNSE(df,3)

frames=[]
for nombreBarrio in dfNombres:
    #print(nombreBarrio)    
    for rooms in range(1,9):
        if(nombreBarrio != "Catalinas" and nombreBarrio!="Villa Riachuelo"):
            dfFiltrado=sacarOutliers(df,rooms,nombreBarrio)
            dfFiltrado['NS'] = dic[nombreBarrio]
            frames.append(dfFiltrado)
result = pd.concat(frames)

result[['surface_total_in_m2','rooms', 'price','NS']].to_csv(r'dataset.csv')
