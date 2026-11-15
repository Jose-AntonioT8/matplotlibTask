"""Cinco gráficos del dataset de viviendas, con Matplotlib y con Seaborn.

Cada función devuelve una figura del mismo gráfico en la librería pedida
(`matplotlib` o `seaborn`), para que el cuaderno y la página de Streamlit
enseñen la misma vista.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ORDEN_PROXIMIDAD = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# El ingreso del dataset está escalado: 3 equivale a unos 30 000 $ y el máximo
# está cortado en 15. Los cortes cubren ese rango, incluido el valor 15.0001.
TRAMOS_INGRESO = [0, 2, 3, 4, 5, 6, 8, 15.1]
ETIQUETAS_TRAMO = ["0-2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-15"]

COLUMNAS_NUMERICAS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
]

ETIQUETAS = {
    "longitude": "Longitud",
    "latitude": "Latitud",
    "housing_median_age": "Edad media",
    "total_rooms": "Habitaciones",
    "total_bedrooms": "Dormitorios",
    "population": "Población",
    "households": "Hogares",
    "median_income": "Ingreso",
    "median_house_value": "Precio",
}

COLOR_SERIE = "#4C78A8"


def comprobar_libreria(libreria: str) -> str:
    nombre = libreria.strip().lower()
    if nombre not in {"matplotlib", "seaborn"}:
        raise ValueError("La librería debe ser 'matplotlib' o 'seaborn'.")
    return nombre


def aplicar_estilo(libreria: str) -> None:
    """Matplotlib se deja en su estilo base; Seaborn aplica su tema estadístico."""
    if libreria == "seaborn":
        sns.set_theme(style="whitegrid", palette="colorblind")
        return
    sns.reset_orig()
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
        }
    )


def formatear_precio(eje) -> None:
    eje.set_major_formatter(mticker.FuncFormatter(lambda valor, _pos: f"{valor / 1000:.0f} mil $"))


def precio_medio_por_proximidad(df: pd.DataFrame) -> pd.DataFrame:
    conteo = df["ocean_proximity"].value_counts()
    presentes = [categoria for categoria in ORDEN_PROXIMIDAD if categoria in conteo.index]
    resumen = (
        df.groupby("ocean_proximity", as_index=False)["median_house_value"]
        .mean()
        .rename(columns={"median_house_value": "precio_medio"})
    )
    resumen["ocean_proximity"] = pd.Categorical(
        resumen["ocean_proximity"], categories=presentes, ordered=True
    )
    resumen = resumen.sort_values("ocean_proximity")
    resumen["n"] = resumen["ocean_proximity"].astype(str).map(conteo).astype(int)
    # El recuento va en la etiqueta: ISLAND solo tiene 5 distritos y su media
    # no debe leerse como si pesara lo mismo que INLAND.
    resumen["etiqueta"] = resumen.apply(
        lambda fila: f"{fila['ocean_proximity']}  (n={fila['n']})", axis=1
    )
    return resumen


def precio_medio_por_tramo(df: pd.DataFrame) -> pd.DataFrame:
    trabajo = df.copy()
    trabajo["tramo_ingreso"] = pd.cut(
        trabajo["median_income"],
        bins=TRAMOS_INGRESO,
        labels=ETIQUETAS_TRAMO,
        include_lowest=True,
    )
    resumen = (
        trabajo.groupby("tramo_ingreso", observed=True)["median_house_value"]
        .mean()
        .reset_index()
        .rename(columns={"median_house_value": "precio_medio"})
    )
    resumen["tramo_ingreso"] = resumen["tramo_ingreso"].astype(str)
    return resumen


def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame | None:
    columnas = [columna for columna in COLUMNAS_NUMERICAS if df[columna].nunique(dropna=True) > 1]
    if len(columnas) < 2:
        return None
    return df[columnas].corr()


def _titulo(ax, texto: str) -> None:
    ax.set_title(texto, fontsize=13, pad=12)


def _sin_marcos(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figura_barras(df: pd.DataFrame, libreria: str) -> plt.Figure:
    """Comparación: precio medio de cada proximidad al océano."""
    libreria = comprobar_libreria(libreria)
    aplicar_estilo(libreria)
    resumen = precio_medio_por_proximidad(df)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    etiquetas_valor = [f"{valor / 1000:.0f} mil $" for valor in resumen["precio_medio"]]
    # Un solo color: la categoría ya está en el eje. Otro color por barra
    # no añadiría información.
    if libreria == "matplotlib":
        barras = ax.barh(resumen["etiqueta"], resumen["precio_medio"], color=COLOR_SERIE)
        ax.bar_label(barras, labels=etiquetas_valor, padding=4, fontsize=9)
        # barh coloca la primera categoría abajo; Seaborn la coloca arriba.
        ax.invert_yaxis()
        _sin_marcos(ax)
    else:
        sns.barplot(
            data=resumen,
            x="precio_medio",
            y="etiqueta",
            order=resumen["etiqueta"].tolist(),
            color=COLOR_SERIE,
            ax=ax,
        )
        ax.bar_label(ax.containers[0], labels=etiquetas_valor, padding=4, fontsize=9)

    _titulo(ax, "Precio medio según la proximidad al océano")
    ax.set_xlabel("Precio medio de la vivienda")
    ax.set_ylabel("")
    ax.set_xlim(0, resumen["precio_medio"].max() * 1.22)
    formatear_precio(ax.xaxis)
    ax.grid(True, axis="x", alpha=0.35)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig


def figura_histograma(df: pd.DataFrame, libreria: str) -> plt.Figure:
    """Distribución: forma del ingreso medio, con los mismos cortes en ambas librerías."""
    libreria = comprobar_libreria(libreria)
    aplicar_estilo(libreria)
    # Bordes fijos para que Matplotlib y Seaborn partan el eje igual.
    cortes = np.linspace(0, 16, 33)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    if libreria == "matplotlib":
        ax.hist(
            df["median_income"],
            bins=cortes,
            color=COLOR_SERIE,
            edgecolor="white",
            linewidth=0.6,
        )
        _sin_marcos(ax)
    else:
        sns.histplot(
            data=df,
            x="median_income",
            bins=cortes,
            color=COLOR_SERIE,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
        )

    _titulo(ax, "Distribución del ingreso medio")
    ax.set_xlabel("Ingreso medio (unidades del dataset, tope en 15)")
    ax.set_ylabel("Número de distritos")
    ax.set_xlim(0, 16)
    fig.tight_layout()
    return fig


def figura_dispersion(df: pd.DataFrame, libreria: str) -> plt.Figure:
    """Relación: ingreso medio frente al precio, con la proximidad en el color."""
    libreria = comprobar_libreria(libreria)
    aplicar_estilo(libreria)
    presentes = [
        categoria for categoria in ORDEN_PROXIMIDAD if categoria in set(df["ocean_proximity"])
    ]
    paleta = dict(zip(presentes, sns.color_palette("colorblind", n_colors=len(presentes))))
    fig, ax = plt.subplots(figsize=(10.6, 5.6))

    if libreria == "matplotlib":
        for categoria in presentes:
            grupo = df[df["ocean_proximity"] == categoria]
            ax.scatter(
                grupo["median_income"],
                grupo["median_house_value"],
                s=12,
                alpha=0.28,
                linewidths=0,
                label=categoria,
                color=paleta[categoria],
            )
        _sin_marcos(ax)
    else:
        sns.scatterplot(
            data=df,
            x="median_income",
            y="median_house_value",
            hue="ocean_proximity",
            hue_order=presentes,
            palette=paleta,
            s=12,
            alpha=0.28,
            linewidth=0,
            ax=ax,
        )

    # Fuera del área de puntos para no tapar el tope de 500 000 $.
    ax.legend(
        title="Proximidad al océano",
        markerscale=2.5,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )

    _titulo(ax, "Ingreso medio y precio de la vivienda")
    ax.set_xlabel("Ingreso medio (unidades del dataset)")
    ax.set_ylabel("Precio mediano de la vivienda")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 550_000)
    formatear_precio(ax.yaxis)
    fig.tight_layout()
    return fig


def figura_lineas(df: pd.DataFrame, libreria: str) -> plt.Figure:
    """Tendencia: el precio medio sube al pasar a tramos de ingreso más altos."""
    libreria = comprobar_libreria(libreria)
    aplicar_estilo(libreria)
    resumen = precio_medio_por_tramo(df)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    if libreria == "matplotlib":
        ax.plot(
            resumen["tramo_ingreso"],
            resumen["precio_medio"],
            marker="o",
            linewidth=2.2,
            color=COLOR_SERIE,
        )
        _sin_marcos(ax)
    else:
        sns.lineplot(
            data=resumen,
            x="tramo_ingreso",
            y="precio_medio",
            marker="o",
            linewidth=2.2,
            color=COLOR_SERIE,
            ax=ax,
        )

    for tramo, valor in zip(resumen["tramo_ingreso"], resumen["precio_medio"]):
        ax.annotate(
            f"{valor / 1000:.0f} mil $",
            (tramo, valor),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1F2933",
        )

    _titulo(ax, "Precio medio por tramo de ingreso")
    ax.set_xlabel("Tramo de ingreso medio")
    ax.set_ylabel("Precio medio de la vivienda")
    if not resumen.empty:
        ax.set_ylim(0, resumen["precio_medio"].max() * 1.22)
    formatear_precio(ax.yaxis)
    fig.tight_layout()
    return fig


def figura_mapa_calor(df: pd.DataFrame, libreria: str) -> plt.Figure:
    """Relación entre muchas variables: matriz de correlación."""
    libreria = comprobar_libreria(libreria)
    aplicar_estilo(libreria)
    matriz = matriz_correlacion(df)
    fig, ax = plt.subplots(figsize=(9, 7.2))

    if matriz is None:
        ax.text(
            0.5,
            0.5,
            "No hay variación suficiente para calcular correlaciones.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        _titulo(ax, "Correlación entre variables numéricas")
        fig.tight_layout()
        return fig

    etiquetas = [ETIQUETAS[columna] for columna in matriz.columns]
    anotaciones = matriz.map(lambda valor: f"{valor:.2f}".replace(".", ","))

    if libreria == "matplotlib":
        imagen = ax.imshow(matriz.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(etiquetas)), labels=etiquetas, rotation=40, ha="right")
        ax.set_yticks(range(len(etiquetas)), labels=etiquetas)
        for fila in range(matriz.shape[0]):
            for columna in range(matriz.shape[1]):
                valor = matriz.iloc[fila, columna]
                ax.text(
                    columna,
                    fila,
                    anotaciones.iloc[fila, columna],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(valor) >= 0.55 else "#1F2933",
                )
        ax.set_aspect("equal")
        fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04).set_label("Correlación")
    else:
        sns.heatmap(
            matriz,
            annot=anotaciones,
            fmt="",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Correlación"},
            xticklabels=etiquetas,
            yticklabels=etiquetas,
            annot_kws={"fontsize": 8},
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        for texto in ax.texts:
            try:
                valor = float(texto.get_text().replace(",", "."))
            except ValueError:
                continue
            texto.set_color("white" if abs(valor) >= 0.55 else "#1F2933")

    ax.grid(False)
    _titulo(ax, "Correlación entre variables numéricas")
    fig.tight_layout()
    return fig
