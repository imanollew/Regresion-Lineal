import time
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
def sacarOutliers(data,cantAmbientes,barrio):
	dfFiltrado = (data.ix[(data.place_name == barrio) & (data.rooms == cantAmbientes)])	
	media=dfFiltrado.loc[:,"price"].mean()
	#print(media)
	desvioStandar=dfFiltrado.loc[:,"price"].std()
	#outlier:  media(u) +- 3* desvio standard (o)
	dfFiltrado2=(dfFiltrado.ix[(dfFiltrado.price > (media-(3*desvioStandar))) & (dfFiltrado.price < (media+(3*desvioStandar)))])
	#media=dfFiltrado2.loc[:,"price"].mean()
	#return (media)
	return (dfFiltrado2)

def estimarNSE(data,rooms):
	dic={}
	dicOrdenado={}
	df = (data.ix[(data.state_name == 'Capital Federal') & (data.rooms == 3)])
	#dfAgrupado= df.groupby('place_name') #df
	dfNombres=df.groupby('place_name').groups
	
	#print (data)
	#for x in df:
		#print(x)

	#xx=dfAgrupado.get_group('Almagro') >>printea todos los de almagro
	for x in dfNombres:
		#dfBarrio=(df.ix[(data.place_name == x)])  #version 1
		dfBarrio=sacarOutliers(data,rooms,x)
		i=dfBarrio.loc[:,"price"].mean()		
		dic[x]=i
	#print ("dic: ",dic)
	lista=sorted((value,key) for (key,value) in dic.items())
	contador=1
	for x in lista:
		dicOrdenado[x[1]]=contador
		contador=contador+1
	#print(dicOrdenado)
	return dicOrdenado


data = read_csv('properati-AR-2018-02-01-properties-sell.csv')



#1
df = (data.ix[(data.state_name == 'Capital Federal') & (data.rooms >= 1) & (data.rooms <= 8)])
#2

print (df)
#time.sleep(5)
df['place_name'].replace('', np.nan, inplace=True)
df.dropna(subset=['place_name'], inplace=True)
#print (df)
#time.sleep(5)
df['rooms'].replace('', np.nan, inplace=True)
df.dropna(subset=['rooms'], inplace=True)
#print (df)
#time.sleep(5)
df['surface_total_in_m2'].replace('', np.nan, inplace=True)
df.dropna(subset=['surface_total_in_m2'], inplace=True)
#print (df)
#time.sleep(5)
df['price'].replace('', np.nan, inplace=True)
df.dropna(subset=['price'], inplace=True)
#print (df)


#3a
dfNombres=df.groupby('place_name').groups
dic=estimarNSE(df,3)
#print(dic)
#print(dfNombres)
#dfAgrupado= df.groupby('place_name')
#cat=dfAgrupado.get_group('Catalinas') 
#print(cat)
#print (cat[['surface_total_in_m2','rooms', 'price',]])
#print(dic.keys())
#dfFiltrado1 = pd.DataFrame()
#print(dfFiltrado)
frames=[]
for nombreBarrio in dfNombres:
    #print(nombreBarrio)    
    for rooms in range(1,9):
        if(nombreBarrio != "Catalinas" and nombreBarrio!="Villa Riachuelo"):
            dfFiltrado=sacarOutliers(df,rooms,nombreBarrio)
            dfFiltrado['NS'] = dic[nombreBarrio]
            #print(dfFiltrado[['surface_total_in_m2']])            
            #dfFiltrado1.loc[i] = dfFiltrado
            frames.append(dfFiltrado)
            #[dic[nombreBarrio],dfFiltrado[['surface_total_in_m2']],3,4]
            #print(dfFiltrado)            
            #print ("Barrio:", nombreBarrio,"NS: ",dic[nombreBarrio],dfFiltrado[['surface_total_in_m2','rooms', 'price',]])
            #dfFiltrado[['surface_total_in_m2','rooms', 'price']]
result = pd.concat(frames)
#print(result[['surface_total_in_m2','rooms', 'price','NS']])
'''	
1.  Considerar el conjunto de propiedades de la CABA que tengan entre 1 y 8 ambientes
2.  Conservar las propiedades que tengan todas las columnas definidas: barrio, cantidad de ambientes (CA), metros cua-
drados (MC) y el valor de la propiedad (VP)
3.  Para cada par (barrio, cantidad de ambientes) del conjunto anterior:
a Filtrar los outliers (o sea: ejecutar la funci ́on del ejercicio anterior)
b Para cada propiedad imprimir por pantalla la siguiente informaci ́on: NS,MC,CA(rooms),VP(price) (notar que para determinar
NS se requiere el diccionario del punto anterior)
Finalmente hay que generar el archivo “dataset.csv” redireccionando el output a un archivo, por ejemplo:
g e n e r a r
d a t a s e t . py
>
d a t a s e t . c s v'''

result[['surface_total_in_m2','rooms', 'price','NS']].to_csv(r'dataset.csv')
