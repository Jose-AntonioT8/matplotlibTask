"""Página Streamlit con los cinco gráficos del cuaderno de viviendas."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from graficos import (
    ORDEN_PROXIMIDAD,
    figura_barras,
    figura_dispersion,
    figura_histograma,
    figura_lineas,
    figura_mapa_calor,
)

st.set_page_config(
    page_title="Viviendas en California",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def cargar_viviendas() -> pd.DataFrame:
    ruta = Path(__file__).resolve().parent / "housing.csv"
    return pd.read_csv(ruta)


def dibujar(fig) -> None:
    st.pyplot(fig, clear_figure=False)
    plt.close(fig)


viviendas = cargar_viviendas()

st.title("Viviendas en California")
st.markdown(
    "Los cinco gráficos del cuaderno, con la librería que elijas. "
    "Los filtros de la barra lateral recalculan barras, histograma, dispersión, "
    "línea y mapa de calor sobre el mismo subconjunto de distritos."
)

with st.sidebar:
    st.header("Vista")
    opcion_libreria = st.radio("Librería", ["Matplotlib", "Seaborn"], index=0)
    st.header("Filtros")
    proximidades = st.multiselect(
        "Proximidad al océano",
        ORDEN_PROXIMIDAD,
        default=ORDEN_PROXIMIDAD,
    )
    ingreso = st.slider("Ingreso medio", min_value=0.0, max_value=15.0, value=(0.0, 15.0), step=0.1)
    st.caption(
        "El ingreso viene escalado en el dataset: 3 son unos 30 000 $ "
        "y los valores más altos están cortados en 15."
    )

libreria = opcion_libreria.lower()
# El dataset guarda el tope de ingreso como 15.0001. Si el control está en 15,
# esos distritos censurados siguen dentro de la selección.
ingreso_min, ingreso_max = ingreso
if ingreso_max >= 15:
    ingreso_max = float(viviendas["median_income"].max())
datos = viviendas[
    viviendas["ocean_proximity"].isin(proximidades)
    & viviendas["median_income"].between(ingreso_min, ingreso_max)
].copy()

if datos.empty:
    st.warning("Ningún distrito cumple esos filtros. Amplía la proximidad o el rango de ingreso.")
    st.stop()

if len(datos) < 30:
    st.warning("Hay menos de 30 distritos en la selección. Los gráficos quedan poco fiables.")

precio_mediano = float(datos["median_house_value"].median())
ingreso_mediano = float(datos["median_income"].median())
correlacion_ingreso = datos[["median_income", "median_house_value"]].corr().iloc[0, 1]
texto_correlacion = (
    "—" if pd.isna(correlacion_ingreso) else f"{correlacion_ingreso:.2f}".replace(".", ",")
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Distritos", f"{len(datos):,}".replace(",", "."))
m2.metric("Precio mediano", f"{precio_mediano / 1000:.0f} mil $")
m3.metric("Ingreso mediano", f"{ingreso_mediano:.2f}".replace(".", ","))
m4.metric("Correlación ingreso–precio", texto_correlacion)

st.caption(f"Librería activa: {opcion_libreria}. Misma selección de datos en los cinco gráficos.")

with st.expander("Por qué estos cinco gráficos"):
    st.markdown(
        """
- **Barras.** Comparan el precio medio entre categorías. El eje empieza en 0
  para no exagerar las diferencias, y la etiqueta `n` muestra el tamaño de cada grupo.
- **Histograma.** Resume la forma del ingreso: cola a la derecha y corte en 15.
- **Dispersión.** Cruza dos variables numéricas. El color añade la proximidad al océano.
- **Líneas.** Los tramos de ingreso tienen orden, así que la línea enseña la tendencia.
- **Mapa de calor.** Junta las correlaciones numéricas. La paleta azul–rojo es divergente
  y evita el par rojo–verde.
        """
    )

columna_barras, columna_histograma = st.columns(2)
with columna_barras:
    st.subheader("Comparar")
    dibujar(figura_barras(datos, libreria))
    st.caption("Precio medio del distrito según la proximidad al océano. ISLAND tiene muy pocos casos.")
with columna_histograma:
    st.subheader("Distribución")
    dibujar(figura_histograma(datos, libreria))
    st.caption("Casi todos los distritos se concentran en ingresos bajos o medios. El tope en 15 deja una barra final.")

st.subheader("Relación entre dos variables")
dibujar(figura_dispersion(datos, libreria))
st.caption("A mayor ingreso, mayor precio. También se ve el tope de 500 000 $ en el precio de la vivienda.")

st.subheader("Tendencia")
dibujar(figura_lineas(datos, libreria))
st.caption("La media del precio sube en cada tramo de ingreso, sin retrocesos.")

st.subheader("Varias variables a la vez")
dibujar(figura_mapa_calor(datos, libreria))
st.caption("El ingreso es la variable numérica que más acompaña al precio. Habitaciones, dormitorios y hogares van muy juntos.")
