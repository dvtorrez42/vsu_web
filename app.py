import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from threading import RLock
import geopandas as gpd

import contextily as ctx


from utils import (
    configure_page,
    configure_overview,
    plot_degradacion_acumulada_dimension,
    plot_valores_conservacion,
    plot_ambientes_paisaje,
    process_uploaded_file,
    plot_predio_ambientes,
    configure_form
)


def main() -> None:
    
    configure_page()

    if "form_values" not in st.session_state:
        st.session_state.form_values = {
            "r21": None,
            "r22": None,
            "r23": None,
            "r24": None,    
            "r25": None,
            "r31": None,
            "r32": None,
            "r33": None,
            "r34": None,
            "r35": None,
            "r36": None,
            "r37": None,
            "r38": None,
            "r39": None,
        }

    if "predio" not in st.session_state:
        st.session_state.predio = None


                
    col1, col2 = st.columns([2, 1])

    predio = st.session_state.predio
    form_values = st.session_state.form_values

    with col1:
        st.markdown("## Parámetros para el análisis")
        with st.form(key="analysis_params"):
            col_dist, col_perc = st.columns(2)
            with col_dist:
                region = st.number_input("Región (metros)", min_value=None, value=10_000, max_value=None, step=1_000)  # región sin valor máximo
                distancia_paisaje = st.number_input("Distancia Paisaje del Predio (metros)", min_value=100, value=5_000, max_value=None, step=1_000)  #todo: cambiar a distancia paisaje
                intervalo = st.number_input("Intervalo (metros) (=>100, =<distancia paisaje)", min_value=100, value=500, max_value=None, step=1)
                replicas = st.number_input("Réplicas", min_value=1, value=250, max_value=500, step=10)
                percentil_inferior = st.number_input("Percentil Inferior", min_value=None, value=0.33, max_value=None)
                percentil_superior = st.number_input("Percentil Superior", min_value=None, value=0.66, max_value=None)
                st.form_submit_button(label="Aplicar parámetros")
            if distancia_paisaje > region:
                st.error("La distancia paisaje debe ser menor que la región.")
            if intervalo > distancia_paisaje:
                st.error("El intervalo debe ser menor o igual a la distancia paisaje.")
            if percentil_inferior >= percentil_superior:
                st.error("El percentil inferior debe ser menor que el percentil superior.")
                
        st.markdown("# Entrada de datos")
        st.markdown("## 1. Primera parte")
        st.markdown("Subir el predio en formato KML o KMZ para visualizarlo en el mapa y realizar análisis.")
        uploaded_file = st.file_uploader("Subir archivo KML", type=["kml", "kmz"])
        if uploaded_file is not None:
            predio = process_uploaded_file(uploaded_file)
            if predio is not None:
                st.session_state.predio = predio
                col_a, col_b, _ = st.columns([.2, .6, .2])
                with col_b:
                    #st.pyplot(fig)
                    _lock = RLock()
                    with _lock:
                        # Convert to Web Mercator projection for adding basemap
                        predio_webmerc = predio.to_crs(epsg=3857)
                        # Create figure and axis
                        fig, ax = plt.subplots()
                        ax.set_title("Mapa del Predio")
                        predio_webmerc.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5)
                        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
                        st.pyplot(fig)
                st.success("Archivo procesado correctamente.")
        
        if predio is not None:
            st.markdown("# Predio en el paisaje")
            with st.spinner("Mapeando el predio en el paisaje...", show_time=True): 
                plot_predio_ambientes(predio)
            st.success("Listo!")
            
            st.markdown("# Degradación acumulada por Dimensión")
            with st.spinner("Calculando Degradación Acumulada por Dimensión...", show_time=True):
                plot_degradacion_acumulada_dimension(
                    predio, 
                    region=region,
                    distancia_paisaje_metros=distancia_paisaje, 
                    intervalo_buffer_metros=intervalo,
                    replicas=replicas,
                    percentil_inferior=percentil_inferior,
                    percentil_superior=percentil_superior  # Manteniendo el valor original, se puede
                    )
            st.success("Listo!")
            
            st.markdown("# Valores de Conservación")
            with st.spinner("Calculando Valores de Conservación...", show_time=True):
                plot_valores_conservacion(
                    predio, 
                    region=region,
                    distancia_paisaje_metros=distancia_paisaje, 
                    intervalo_buffer_metros=intervalo,
                )
            st.success("Listo!")    
                
            st.markdown("# Ambientes del paisaje")
            with st.spinner("Calculando Ambientes del Paisaje...", show_time=True):
                plot_ambientes_paisaje(
                    predio, 
                    distancia_paisaje_metros=distancia_paisaje, 
                    intervalo_buffer_metros=intervalo,
                )
            st.success("Listo!")
            # configure_overview()
        elif predio is not None:
            st.write("### No todos los valores son distintos de None en form_values")
            
    with col2:
        st.markdown("# Formulario (opcional)")
        # Mostrar el formulario si predio está definido y no es None
        if predio is not None:  #todo: cambiar por if todos los graficosya se mostraron
            def on_form_submit(updated_values):
                st.session_state.form_values.update(updated_values)
            configure_form(form_values)
        

if __name__ == "__main__":
    main()