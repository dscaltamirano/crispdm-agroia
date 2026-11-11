# CRISP-DM — Fase 4: Modelado mediante regresión.

**Proyecto:** AgroIA — Producción de piña en México (2003–2025)

**Objetivo:** Predecir el **precio medio rural de la piña** a partir de las características de su producción. Compararemos tres modelos de regresión:

- Lineal.
- Logarítmica.
- Polinómica de grado 2.

Primero usaremos **solo variables numéricas**. Después incorporaremos **estado, municipio y modalidad** mediante *One-Hot Encoding* y repetiremos la evaluación.

---

## 1. Regresión con valores numéricos

### 1.1 Leer datos y seleccionar variables
Importamos las librerías y leemos el archivo CSV:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("dataset/pinna_2003_2025.csv")
df.info()
```

Seleccionamos las variables a utilizar:
```python
selected_variables = df[["Anio", "Sembrada", "Cosechada", "Siniestrada", "Volumenproduccion", "Rendimiento", "Preciomediorural"]]
```
Eliminamos campos vacíos:
```python
selected_variables=selected_variables.dropna()
```

Seleccionamos variables de entrada (X) y variable a predecir (y)
```python
X = selected_variables[["Anio", "Sembrada", "Cosechada", "Siniestrada", "Volumenproduccion", "Rendimiento"]]
y = selected_variables["Preciomediorural"]
```

Dividimos los datos en datos para entrenamiento y datos para prueba:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 1.2 Regresión lineal
La clase `LinearRegression` de Scikit-learn permite realizar un ajuste lineal. Su método `fit()` calcula los coeficientes del hiperplano de ajuste a partir de los datos de entrenamiento, mientras que `predict()` utiliza el modelo ajustado para estimar los valores de salida correspondientes a las variables de entrada que recibe como argumento (`X_test`, en este caso).
```python
lineal = LinearRegression()
lineal.fit(X_train, y_train)
y_predicted_lin = lineal.predict(X_test)
print("Lineal:", r2_score(y_test, y_predicted_lin))
```
La función `r2_score()` calcula el coeficiente de determinación \(R^2\), que permite evaluar el ajuste del modelo. Un valor de \(R^2=1\) indica que las predicciones coinciden perfectamente con los valores reales de los datos evaluados. Sin embargo, esto no garantiza que el modelo sea capaz de realizar buenas predicciones con datos nuevos, ya que podría presentar sobreajuste.


### 1.3 Regresión polinomial de orden 2
La clase `PolynomialFeatures` de Scikit-learn permite generar términos polinómicos a partir de las variables de entrada. Al crear el objeto, especificamos los parámetros `degree=2`, para generar términos hasta segundo grado, e `include_bias=False`, para no incluir una columna constante de unos, ya que `LinearRegression()` incorpora el término independiente por defecto.

```python
poli = PolynomialFeatures(degree=2, include_bias=False)
```
Posteriormente, por medio de la función `fit_transform` se generan los términos polinómicos de los datos de entrenamiento y prueba:

```python
X_train_poli = poli.fit_transform(X_train)
X_test_poli = poli.fit_transform(X_test)
```
Por ejemplo, si tenemos dos variables de entrada:

\[
X=[x_1,x_2]
\]

esta instrucción nos generará cinco características:

\[
X_{\text{poli}}=[x_1,x_2,x_1^2,x_1x_2,x_2^2]
\]

Con estas nuevas características, el modelo ajusta la ecuación:

\[
\hat{y}=b_0+b_1x_1+b_2x_2+b_3x_1^2+b_4x_1x_2+b_5x_2^2
\]

Para hacer el ajuste, se ejecuta el siguiente código:
```python
polinomica = LinearRegression()
polinomica.fit(X_train_poli, y_train)
y_predicted_poli = polinomica.predict(X_test_poli)
print("Polinómica:", r2_score(y_test, y_predicted_poli))
```

### 1.4 Regresión logarítmica

La regresión logarítmica permite modelar relaciones no lineales mediante el logaritmo de las variables de entrada.

Por ejemplo, para dos variables \(x_1\) y \(x_2\), el modelo tiene la forma:

\[
\hat{y}=b_0+b_1\ln(1+x_1)+b_2\ln(1+x_2)
\]

Utilizamos la función `np.log1p()` de NumPy para calcular el logaritmo natural de cada valor más uno:

```python
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)
```

Por ejemplo, si tenemos dos variables de entrada:

\[
X=[x_1,x_2]
\]

La transformación genera:

\[
X_{\text{log}}=[\ln(1+x_1),\ln(1+x_2)]
\]

Se suma `1` para poder trabajar con valores iguales a cero, ya que `ln(0)` no está definido.

A diferencia de `PolynomialFeatures`, esta transformación no genera nuevas columnas: simplemente transforma los valores originales. Se requiere que las entradas sean mayores que \(-1\).

Posteriormente, utilizamos `LinearRegression()` para ajustar el modelo con las variables transformadas:

```python
logaritmica = LinearRegression()
logaritmica.fit(X_train_log, y_train)
y_predicted_log = logaritmica.predict(X_test_log)
print("Logarítmica:", r2_score(y_test, y_predicted_log))
```

### 1.5 Limitaciones de los modelos

No se utilizaron variables categóricas:
- **Información geográfica:** no estamos utilizando el estado ni el municipio donde se produjo la piña. Los precios pueden variar entre regiones, incluso cuando las condiciones de producción son similares.
- **Modalidad de producción:** no consideramos si el cultivo es de riego o temporal, lo que puede influir en las condiciones productivas.

Además, existen variables no consideradas en el dataset como: la demanda, los costos de transporte y las condiciones del mercado los cuales pueden influir en el precio.


**¿Podemos mejorar los modelos incorporando información geográfica y la modalidad de producción?**

En la siguiente sección utilizaremos **One-Hot Encoding** para incorporar estas variables categóricas y volveremos a evaluar los modelos mediante \(R^2\).


## 2. Regresión con valores numéricos y categóricos

En la sección anterior utilizamos únicamente variables numéricas para estimar el precio medio rural de la piña.

Ahora incorporaremos tres variables categóricas: **estado, municipio y modalidad de producción**, y compararemos los resultados de los modelos.

### 2.1 Preparación de datos

Leemos el archivo CSV:

```python
df = pd.read_csv("dataset/pinna_2003_2025.csv")
```

Agrupamos los registros por estado, municipio, modalidad y año. Sumamos las variables de producción y calculamos los promedios de rendimiento y precio:

```python
columnas = ["Idestado", "Idmunicipio", "Idmodalidad", "Anio"]

df_sumas = df.groupby(columnas, as_index=False)[
    ["Sembrada", "Cosechada", "Siniestrada", "Volumenproduccion"]
].sum()

df_promedios = df.groupby(columnas, as_index=False)[
    ["Rendimiento", "Preciomediorural"]
].mean()

df = pd.merge(df_sumas, df_promedios, on=columnas)
df = df.dropna()
```

Seleccionamos las variables de entrada y la variable que queremos predecir:

```python
X = df.drop(columns=["Preciomediorural"])
y = df["Preciomediorural"]
```

**One-Hot Encoding**

Los identificadores de estado, municipio y modalidad representan categorías, no cantidades numéricas.

Por ejemplo, si tenemos tres municipios:

| Municipio | Loma Bonita | Tuxtepec | Isla |
|---|---|---|---|
| Loma Bonita | 1 | 0 | 0 |
| Tuxtepec | 0 | 1 | 0 |
| Isla | 0 | 0 | 1 |

One-Hot Encoding genera una columna por categoría, indicando su presencia con `1` y su ausencia con `0`.

En Pandas podemos realizar esta transformación mediante `pd.get_dummies()`:

```python
X = pd.get_dummies(
    X,
    columns=["Idestado", "Idmunicipio", "Idmodalidad"],
    dtype=int
)
```

Las variables numéricas originales permanecen sin cambios.

Finalmente, dividimos los datos en entrenamiento y prueba:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

### 2.2 Regresión lineal

La regresión lineal puede utilizar directamente las variables numéricas y las nuevas columnas binarias generadas por One-Hot Encoding.

```python
lineal = LinearRegression()
lineal.fit(X_train, y_train)
y_predicted_lin = lineal.predict(X_test)
print("Lineal:", r2_score(y_test, y_predicted_lin))
```

### 2.3 Regresión polinomial de orden 2

Para la regresión polinomial generaremos términos de segundo grado únicamente con las variables numéricas originales. Las columnas de One-Hot Encoding permanecerán sin modificar.

Seleccionamos las variables numéricas:

```python
numericas = ["Anio", "Sembrada", "Cosechada", "Siniestrada",
             "Volumenproduccion", "Rendimiento"]
```

Configuramos el polinomio de segundo grado:

```python
poli = PolynomialFeatures(degree=2, include_bias=False)
```

Generamos los términos polinómicos y los combinamos con las variables categóricas:

```python
X_train_poli = np.column_stack((
    poli.fit_transform(X_train[numericas]),
    X_train.drop(columns=numericas)
))

X_test_poli = np.column_stack((
    poli.transform(X_test[numericas]),
    X_test.drop(columns=numericas)
))
```

Por ejemplo, si tenemos dos variables numéricas \(x_1\) y \(x_2\), y una variable categórica \(c_1\):

\[
X=[x_1,x_2,c_1]
\]

La transformación genera:

\[
X_{\text{poli}}=[x_1,x_2,x_1^2,x_1x_2,x_2^2,c_1]
\]

Observamos que \(c_1\) permanece sin modificar.

Finalmente, entrenamos y evaluamos el modelo:

```python
polinomica = LinearRegression()
polinomica.fit(X_train_poli, y_train)
y_predicted_poli = polinomica.predict(X_test_poli)
print("Polinómica:", r2_score(y_test, y_predicted_poli))
```

### 2.4 Regresión logarítmica

En este modelo aplicaremos el logaritmo únicamente a las variables numéricas originales, conservando sin cambios las columnas de One-Hot Encoding.

Primero creamos copias de los datos:

```python
X_train_log = X_train.copy()
X_test_log = X_test.copy()
```

Aplicamos la transformación logarítmica:

```python
X_train_log[numericas] = np.log1p(X_train[numericas])
X_test_log[numericas] = np.log1p(X_test[numericas])
```

Por ejemplo, para dos variables numéricas \(x_1\) y \(x_2\), y una variable categórica \(c_1\):

\[
X=[x_1,x_2,c_1]
\]

La transformación produce:

\[
X_{\text{log}}=[\ln(1+x_1),\ln(1+x_2),c_1]
\]

Observamos que únicamente se transforman las variables numéricas, mientras que la variable categórica conserva su valor original.

Finalmente, entrenamos y evaluamos el modelo:

```python
logaritmica = LinearRegression()
logaritmica.fit(X_train_log, y_train)
y_predicted_log = logaritmica.predict(X_test_log)
print("Logarítmica:", r2_score(y_test, y_predicted_log))
```

### 2.5 Comparación de resultados

Comparamos los valores de \(R^2\) obtenidos por los tres modelos.

- **Regresión lineal:** utiliza directamente las variables numéricas y categóricas.
- **Regresión polinomial:** incorpora términos cuadráticos e interacciones entre las variables numéricas.
- **Regresión logarítmica:** aplica una transformación logarítmica a las variables numéricas.

Con estos resultados podemos analizar si incorporar información del estado, municipio y modalidad mejora la capacidad predictiva de los modelos.



