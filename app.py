from threading import RLock

import contextily as ctx
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    configure_page,
    plot_ambientes_paisaje,
    plot_degradacion_acumulada_dimension,
    plot_predio_ambientes,
    plot_valores_conservacion,
    process_uploaded_file,
)


def main() -> None:
    configure_page()
    col_0_1, col_0_2, col_0_3 = st.columns([0.1, 0.8, 0.1])
    with col_0_2:
        st.markdown("# NOMBRE DE LA HERRAMIENTA")
        st.markdown("## 1. Entrada de datos")

    if "predio" not in st.session_state:
        st.session_state.predio = None

    col_1_1, col_1_2, col_1_3, col_1_4 = st.columns([0.1, 0.4, 0.4, 0.1])

    predio = st.session_state.predio

    with col_1_2:
        st.markdown("### 1.1  🎛️ Parámetros para el análisis")
        with st.form(key="analysis_params"):
            col_dist, col_perc = st.columns(2)
            with col_dist:
                region = st.number_input(
                    "Región (metros)",
                    min_value=None,
                    value=10_000,
                    max_value=None,
                    step=1_000,
                )  # región sin valor máximo
                distancia_paisaje = st.number_input(
                    "Distancia Paisaje del Predio (metros)",
                    min_value=100,
                    value=5_000,
                    max_value=None,
                    step=1_000,
                )  # todo: cambiar a distancia paisaje
                intervalo = st.number_input(
                    "Intervalo (metros) (=>100, =<distancia paisaje)",
                    min_value=100,
                    value=500,
                    max_value=None,
                    step=1,
                )
                replicas = st.number_input(
                    "Réplicas", min_value=1, value=100, max_value=500, step=10
                )
                percentil_inferior = st.number_input(
                    "Percentil Inferior", min_value=None, value=0.33, max_value=None
                )
                percentil_superior = st.number_input(
                    "Percentil Superior", min_value=None, value=0.66, max_value=None
                )
                st.form_submit_button(label="Aplicar parámetros")
            if distancia_paisaje > region:
                st.error("La distancia paisaje debe ser menor que la región.")
            if intervalo > distancia_paisaje:
                st.error("El intervalo debe ser menor o igual a la distancia paisaje.")
            if percentil_inferior >= percentil_superior:
                st.error(
                    "El percentil inferior debe ser menor que el percentil superior."
                )
    with col_1_3:
        st.markdown("### 1.2 📍Predio por analizar")
        uploaded_file = st.file_uploader(
            "Subir el predio en formato KML o KMZ para visualizarlo en el mapa y realizar análisis.",
            type=["kml", "kmz"],
        )
        if uploaded_file is not None:
            predio = process_uploaded_file(uploaded_file)
            if predio is not None:
                st.session_state.predio = predio
                col_a, col_b, _ = st.columns([0.2, 0.6, 0.2])
                with col_b:
                    # st.pyplot(fig)
                    _lock = RLock()
                    with _lock:
                        # Convert to Web Mercator projection for adding basemap
                        predio_webmerc = predio.to_crs(epsg=3857)
                        # Create figure and axis
                        fig, ax = plt.subplots()
                        ax.set_title("Mapa del Predio")
                        predio_webmerc.plot(
                            ax=ax, color="lightblue", edgecolor="black", alpha=0.5
                        )
                        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
                        st.pyplot(fig)

    col_2_1, col_2_2, col_2_3 = st.columns([0.1, 0.8, 0.10])
    if predio is not None:
        with col_2_2:
            st.markdown("## 2. Predio en el paisaje")
            with st.spinner("Mapeando el predio en el paisaje...", show_time=True):
                plot_predio_ambientes(
                    predio, region=region, distancia_paisaje_metros=distancia_paisaje
                )

            st.markdown("## 3. Degradación acumulada por Dimensión")
            with st.spinner(
                "Calculando Degradación Acumulada por Dimensión...", show_time=True
            ):
                plot_degradacion_acumulada_dimension(
                    predio,
                    region=region,
                    distancia_paisaje_metros=distancia_paisaje,
                    intervalo_buffer_metros=intervalo,
                    replicas=replicas,
                    percentil_inferior=percentil_inferior,
                    percentil_superior=percentil_superior,  # Manteniendo el valor original, se puede
                )
            st.success("Listo!")

            st.markdown("## 4. Valores de Conservación")
            with st.spinner("Calculando Valores de Conservación...", show_time=True):
                plot_valores_conservacion(
                    predio,
                    region=region,
                    distancia_paisaje_metros=distancia_paisaje,
                    intervalo_buffer_metros=intervalo,
                )

            st.markdown("## 5. Ambientes del paisaje")
            with st.spinner("Calculando Ambientes del Paisaje...", show_time=True):
                plot_ambientes_paisaje(
                    predio,
                    region=region,
                    distancia_paisaje_metros=distancia_paisaje,
                    intervalo_buffer_metros=intervalo,
                )


if __name__ == "__main__":
    main()
