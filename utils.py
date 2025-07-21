import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import geopandas as gpd
from threading import RLock
from matplotlib_scalebar.scalebar import ScaleBar

DIMENSIONES_DICT = {
    'Conectividad':{
        'path': "./dimensiones/DIM_fragmentacion_TEM_celdas.shp",
        'label': 'Conectividad', #Título del mapa, también nombre de archivo
        'color_map': 'Reds',
        'color_plot': 'red',
        'pc_regiones_inferior': 0.150,
        'pc_regiones_superior': 0.205
    },
    'Representacion':{
        'path': "./dimensiones/DIM_representacion_MB_celdas.shp",
        'label': 'Representación de Ambientes Naturales',
        'color_map': 'Greys',
        'color_plot': 'black',
        'pc_regiones_inferior': 0.071,
        'pc_regiones_superior': 0.113
    },
    'H20':{
        'path': "./dimensiones/DIM_filtradoH2O_MB_poligonos.shp",
        'label': 'Mantenimiento de Calidad de Agua',
        'color_map': 'Blues',
        'color_plot': 'blue',
        'pc_regiones_inferior': 0.060,
        'pc_regiones_superior': 0.084
    }
}

AMBIENTES_DICT = {
    'Bosque':{
        'path': "./ambientes/DIM_bosque_MB.shp",
        'label': 'Bosque', 
        'color_plot': 'darkgreen',
    },
    'Pastizal':{
        'path': "./ambientes/DIM_pastizal_MB.shp",
        'label': 'Pastizal',
        'color_plot': 'brown',
    },
    'Humedal':{
        'path': "./ambientes/DIM_humedal_MB.shp",
        'label': 'Humedal',
        'color_plot': 'skyblue',
    }
}

def plot_degradacion_acumulada_dimension(
        predio, 
        region=10_000,
        distancia_paisaje_metros=5_000, 
        intervalo_buffer_metros=500,
        replicas=250,
        percentil_inferior=0.33,
        percentil_superior=0.66,
    ):
    ################### NECESIDADES DEL PAISAJE - VALORES ACUMULATIVOS OPTIMIZADO ###############################

    # Lista con dimensiones
    dimensiones_list = [value['path'] for value in DIMENSIONES_DICT.values()]


    # Lista con buffers a usar
    #!Posible parámetro (distancia máxima y n particiones y/o cada cuánto)
    #!PARAMETRO: "Distancia máxima": (lo que se llama paisaje), default: 5000
    #!PARAMETRO: "Región": Default 10000
    #!PARAMETRO: "Replicas": Default 250
    #!PARAMETRO: percentil_inferior = 0.33
    #!PARAMETRO: percentil_superior = 0.66
    # Create buffer distances programmatically
    # Maximum buffer distance

    distancias_buffer = list(range(intervalo_buffer_metros, distancia_paisaje_metros + intervalo_buffer_metros, intervalo_buffer_metros))  # Creates [500, 1000, 1500, ..., 5000]
    #st.write("Distancias de buffer:", distancias_buffer)
    max_buffer = max(distancias_buffer)

    predio_original = predio.to_crs(32721) 
    valores_indicador_dimensiones = []

    file_name = "Predio NN"  #todo: CAMBIAR  #  os.path.splitext(os.path.basename(predios_path))[0] # para usar despues como nombre del predio en figuras

    # crear BUFFER de la region alrededor del predio
    extension = max_buffer + intervalo_buffer_metros
    region = region  #! variable


    predio_region = predio_original.copy()
    predio_region["geometry"] = predio_original.buffer(region)
    promedios_tabla = []

    dim_1 = gpd.read_file(dimensiones_list[0])
    dim_1_region = dim_1.overlay(predio_region, how='intersection')
    dim_2 = gpd.read_file(dimensiones_list[1])
    dim_2_region = dim_2.overlay(predio_region, how='intersection')
    dim_3 = gpd.read_file(dimensiones_list[2])
    dim_3_region = dim_3.overlay(predio_region, how='intersection')

    dimensiones_list_region = [dim_1_region, dim_2_region, dim_3_region]

    #### Mover el predio aleatoriamente
    replicas = replicas  #! hacer variable
    max_metros = region - extension # para que el buffer de 5000 metros nunca se salga de la región de 10000
    desplazamientos_x = []
    desplazamientos_y = []
    np.random.seed(42)

    for i in range(replicas):
        desplazamientos_x.append(np.random.uniform(-max_metros, max_metros, size= 1))
        desplazamientos_y.append(np.random.uniform(-max_metros, max_metros, size= 1))

    for i in range(replicas):
        predio = predio_original.copy()
        if i == 0:
            desplazado_x = 0
            desplazado_y = 0
        else:
            desplazado_x = desplazamientos_x[i]
            desplazado_y = desplazamientos_y[i]
        predio["geometry"] = predio.translate(desplazado_x, desplazado_y)     # Mover el predio aleatoriamente

        # crear BUFFER de la extension del predio (buffer un poco mayor al buffer más grande, para generar intersecciones más livianas y mapear)
        predio_extension = predio.copy()
        predio_extension["geometry"] = predio.buffer(extension) #

        # este loopea por todas las dimensiones
        for dimension_region in dimensiones_list_region:
            dimension_extension = dimension_region.overlay(predio_extension, how='intersection')

            # Intersección de la DIMENSION con la REGION
            #dimension_region = dimension_total.overlay(predio_region, how='intersection')
            dimension_region["area_Ha"] = dimension_region.area/10000
            dimension_region["indicador_parche"] = dimension_region["area_Ha"] * dimension_region["INDICADOR"]
            region_indicador = dimension_region["indicador_parche"].sum()
            region_promedio = round(region_indicador / (dimension_region.area.sum()/10000),4)
            promedios_tabla.append(region_promedio)

            valores_indicador = []
            contador_vueltas = 0

            # fig, ax = plt.subplots(figsize=(4, 4))
            # dimension.plot(column="INDICADOR", cmap='OrRd', legend= True, ax=ax)
            # predio.plot(ax=ax)
            # plt.title(predios_path[46:55])


            # este loop calcula el indicador para cada buffer de una dimensión
            for buf in distancias_buffer:
                #  Crear BUFFER del predio
                predio_buf = predio.copy()
                predio_buf["geometry"] = predio.buffer(buf)
                #  Intersección de la DIMENSION con el BUFFER
                dimension_buf = dimension_extension.overlay(predio_buf, how='intersection')
                # paisaje_buf = dimension_buf.overlay(predio, how="difference")   #  DIMENSION cortada por BUFFER agujereada con el predio = le llamamos "paisaje"
                paisaje_buf = dimension_buf
                #  Agregamos columna de área en hectáreas
                paisaje_buf["area_Ha"] = paisaje_buf.area/10000
                #  Calculo del indicador para ese paisaje
                paisaje_buf["indicador_parche"] = paisaje_buf["area_Ha"] * paisaje_buf["INDICADOR"]
                #
                paisaje_indicador = paisaje_buf["indicador_parche"].sum()
                #
                out = round((((paisaje_indicador) / (dimension_buf.area.sum()/10000)) - region_promedio) , 4)

                out = 0 if out > 0 else out # quedarse solo con los valores negativos (0 si es positivo)
                try:
                    out_anterior = valores_indicador[contador_vueltas-1]
                except IndexError:
                    out_anterior = 0
                accu = round(-out + out_anterior, 4)
                valores_indicador.append(accu)
                contador_vueltas = contador_vueltas + 1

            valores_indicador_dimensiones.append(valores_indicador)
    # Create a list to store the data for the DataFrame
    table_data = []

    # Number of dimensions
    num_dimensions = len(dimensiones_list)

    # # Iterate through the dimensions, replicas, and buffer distances
    # for replica in range(replicas):
    #     for dim_index in range(num_dimensions):
    #         dimension_name = dimensiones_list[dim_index].split('/')[-1].split('.')[0]
    #         for buf_index, buffer_value in enumerate(valores_indicador_dimensiones[replica * num_dimensions + dim_index]):
    #             table_data.append([dimension_name, distancias_buffer[buf_index], buffer_value])
    
    # Iterate through the dimensions, replicas, and buffer distances
    for replica in range(replicas):
        for i, dim in enumerate(DIMENSIONES_DICT.values()):
            for buf_index, buffer_value in enumerate(valores_indicador_dimensiones[replica * num_dimensions + i]):
                table_data.append([dim['label'], distancias_buffer[buf_index], buffer_value])

    # Create the DataFrame
    df = pd.DataFrame(table_data, columns=['Dimension', 'Buffer_Distance', 'Indicator_Value'])



    #############   calculate percentiles of the three variables in df at max buffer distance

    # Filter data for max buffer distance
    df_max_buffer = df[df['Buffer_Distance'] == max_buffer]

    # Calculate percentiles for each dimension
    percentil_inferior = percentil_inferior  
    percentil_superior = percentil_superior  
    percentiles = df_max_buffer.groupby('Dimension')['Indicator_Value'].quantile([percentil_inferior, percentil_superior]).unstack()
    
    st.write("Percentiles at maximum Buffer Distance:")
    st.write(percentiles)

    # Access the first value of each dimension in df_max_buffer
    valores_accu_predio_maxbuff = df_max_buffer.groupby('Dimension')['Indicator_Value'].first()

    st.markdown("## Prioridad por Dimensión")
    # Iterate through dimensions and compare values with percentiles
    for dimension in percentiles.index:
        value = valores_accu_predio_maxbuff[dimension]
        p_inf = percentiles.loc[dimension, percentil_inferior]
        p_sup = percentiles.loc[dimension, percentil_superior]

        #! Esto es lo que va a la tablita de prioridad por dimensión
        
        if value > p_sup:
            st.markdown(f"### {dimension}: Prioridad Alta")
            st.write("Percentil superior:", p_sup)
            st.write("Percentil inferior:", p_inf)
            st.write("Valor predio:", value)
            
        elif value < p_inf:
            st.markdown(f"### {dimension}: Prioridad Baja")
            st.write("Percentil superior:", p_sup)
            st.write("Percentil inferior:", p_inf)
            st.write("Valor predio:", value)
        else:
            st.markdown(f"### {dimension}: Prioridad Media")
            st.write("Percentil superior:", p_sup)
            st.write("Percentil inferior:", p_inf)
            st.write("Valor predio:", value)




    ############## PLOTEO DE ACUMULADOS ####################

    # Define colors, markers, and dimension names manually
    colors = ['blue', 'black', 'brown']
    # markers = ['o', 's', 'D']
    # dimension_names = {
    #     'DIM_representacion_MB_celdas': 'Representación de ambientes naturales',
    #     'DIM_fragmentacion_TEM_celdas': 'Conectividad',
    #     'DIM_filtradoH2O_MB_poligonos': 'Mantenimiento de calidad de H2O'
    # }

    ubicacion_lineas = [5000,5030,5060]

    fig = plt.figure(figsize=(8, 4))

    for i, dimension in enumerate(percentiles.index):
        # Filter data for the current dimension and the first series (10 records)
        df_dim = df[(df['Dimension'] == dimension)].head(10)

        # Plot indicator_value vs buffer_distance
        plt.plot(df_dim['Buffer_Distance'], df_dim['Indicator_Value'],
                label=dimension, color=colors[i], marker=None)

    # Get actual y-axis limits
    y_min, y_max = plt.ylim()


    # Adjust y-axis limits to include all percentiles
    #y_min = min(y_min, percentiles.min().min()) - 0.
    #y_max = 0.03
    y_max = max(y_max, percentiles.max().max()) + 0.03
    y_min = 0

    x_max = max(ubicacion_lineas) + max(ubicacion_lineas) * 0.01
    x_min = 0

    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)

    for i, dimension in enumerate(percentiles.index):
        # Get the actual percentile values
        min_percentile = percentiles.loc[dimension, percentil_inferior]
        max_percentile = percentiles.loc[dimension, percentil_superior]

        # Normalize ymin and ymax
        normalized_ymin = (min_percentile - y_min) / (y_max - y_min)
        normalized_ymax = (max_percentile - y_min) / (y_max - y_min)

        # Add vertical lines at actual percentile positions
        plt.axvline(x= ubicacion_lineas[i], ymin=normalized_ymin, ymax=normalized_ymax,
                    color=colors[i], linestyle='--', linewidth = 2, alpha = 0.7, label=f"Percentiles {percentil_inferior}-{percentil_superior} {dimension}")

    plt.xlabel('Distancia al predio (m)')
    plt.ylabel('Degradación acumulada')
    plt.title(f'Degradación acumulada por dimensión: {file_name}')
    plt.legend(fontsize=8)
    plt.grid(False)
    # plt.show()
    st.pyplot(fig)

def plot_valores_conservacion(
        predio, 
        region=10_000,
        distancia_paisaje_metros=5_000, 
        intervalo_buffer_metros=500,
    ):
    # Lista con dimensiones
    dimensiones_list = [value['path'] for value in DIMENSIONES_DICT.values()]

    # Lista con buffers a usar
    distancias_buffer = list(range(intervalo_buffer_metros, distancia_paisaje_metros + intervalo_buffer_metros, intervalo_buffer_metros))  # Creates [500, 1000, 1500, ..., 5000]
    max_buffer = max(distancias_buffer)
    extension = max_buffer + intervalo_buffer_metros
    # este loopea por todos los predios

    predio_original = predio.to_crs(32721) 
    valores_indicador_dimensiones = []

    predio_extension = predio_original.copy()
    predio_extension["geometry"] = predio_original.buffer(extension)

    # crear BUFFER de la region alrededor del predio
    predio_region = predio_original.copy()
    predio_region["geometry"] = predio_original.buffer(region)
    promedios_tabla = []

    # este loopea por todas las dimensiones
    for dimension_path in dimensiones_list:
        dimension_total = gpd.read_file(dimension_path)
        dimension = dimension_total.overlay(predio_extension, how='intersection')

        # Intersección de la DIMENSION con la REGION
        dimension_region = dimension_total.overlay(predio_region, how='intersection')
        dimension_region["area_Ha"] = dimension_region.area/10000
        dimension_region["indicador_parche"] = dimension_region["area_Ha"] * dimension_region["INDICADOR"]
        region_indicador = dimension_region["indicador_parche"].sum()
        region_promedio = round(region_indicador / (dimension_region.area.sum()/10000),4)
        promedios_tabla.append(region_promedio)

        valores_indicador = []


        # este loop calcula el indicador para cada buffer de una dimensión
        for buf in distancias_buffer:

            predio_buf = predio_original.copy() #  Crear BUFFER del predio
            predio_buf["geometry"] = predio_original.buffer(buf)

            dimension_buf = dimension.overlay(predio_buf, how='intersection') #  Intersección de la DIMENSION con el BUFFER

            # paisaje_buf = dimension_buf.overlay(predio, how="difference") #  DIMENSION cortada por BUFFER agujereada con el predio = le llamamos "paisaje"
            paisaje_buf = dimension_buf

            paisaje_buf["area_Ha"] = paisaje_buf.area/10000 #  Agregamos columna de área en hectáreas

            paisaje_buf["indicador_parche"] = paisaje_buf["area_Ha"] * paisaje_buf["INDICADOR"] #  Calculo del indicador para ese paisaje
            #
            paisaje_indicador = paisaje_buf["indicador_parche"].sum()
            #
            #out = round((((paisaje_indicador) / (dimension_buf.area.sum()/10000)) - region_promedio) , 4)
            out = round((paisaje_indicador)/(dimension_buf.area.sum()/10000), 4)
            #out = round((paisaje_indicador), 4)
            valores_indicador.append(out)
        valores_indicador_dimensiones.append(valores_indicador)


    # Create separate figures for each dimension
    for i, dim in enumerate(DIMENSIONES_DICT.values()):
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(distancias_buffer, valores_indicador_dimensiones[i], 
                label=dim['label'], color=dim['color_plot'], linewidth=2)
        
        # Add reference lines if available
                # Add regional average line if available
        if i < len(promedios_tabla):
            ax.axhline(y=promedios_tabla[i], color=dim['color_plot'], 
                      linestyle='-', linewidth=0.8, alpha=0.9,
                      label=f"Promedio regional")
            
        if 'pc_regiones_superior' in dim:
            ax.axhline(y=dim['pc_regiones_superior'], color=dim['color_plot'], 
                      linestyle='dashed', linewidth=0.5, alpha=0.7,
                      label=f"Percentil superior")
            
        if 'pc_regiones_inferior' in dim:
            ax.axhline(y=dim['pc_regiones_inferior'], color=dim['color_plot'], 
                      linestyle='dotted', linewidth=0.5, alpha=0.7, 
                      label=f"Percentil inferior")
        

        
        
        ax.legend(fontsize=8)
        ax.set_title(f'Valor de Conservación: {dim["label"]}')
        ax.set_xlabel('Distancia al predio (m)')
        ax.set_ylabel('Valor del indicador')
        ax.grid(False)
        
        # Display the figure in Streamlit
        st.pyplot(fig)

def plot_ambientes_paisaje(
        predio, 
        #region=10_000,
        distancia_paisaje_metros=5_000, 
        intervalo_buffer_metros=500,
    ):
    # Lista con ambientes
    dimensiones_list = [value['path'] for value in AMBIENTES_DICT.values()]

    # Lista con buffers a usar
    distancias_buffer = list(range(intervalo_buffer_metros, distancia_paisaje_metros + intervalo_buffer_metros, intervalo_buffer_metros))  # Creates [500, 1000, 1500, ..., 5000]

    predio_original = predio.to_crs(32721)
    valores_indicador_dimensiones = []

    predio_extension = predio_original.copy()
    predio_extension["geometry"] = predio_original.buffer(5500)

    #crear BUFFER de la region alrededor del predio
    predio_region = predio_original.copy()
    predio_region["geometry"] = predio_original.buffer(10000)
    promedios_tabla = []

    # este loopea por todas las dimensiones
    for dimension_path in dimensiones_list:
        dimension_total = gpd.read_file(dimension_path)
        dimension = dimension_total.overlay(predio_extension, how='intersection')

        # Intersección de la DIMENSION con la REGION
        dimension_region = dimension_total.overlay(predio_region, how='intersection')
        dimension_region["area_Ha"] = dimension_region.area/10000
        dimension_region["indicador_parche"] = dimension_region["area_Ha"] * dimension_region["INDICADOR"]
        region_indicador = dimension_region["indicador_parche"].sum()
        region_promedio = round(region_indicador / (dimension_region.area.sum()/10000),4)
        promedios_tabla.append(region_promedio)

        valores_indicador = []

        # este loop calcula el indicador para cada buffer de una dimensión
        for buf in distancias_buffer:

            predio_buf = predio_original.copy() #  Crear BUFFER del predio
            predio_buf["geometry"] = predio_original.buffer(buf)

            dimension_buf = dimension.overlay(predio_buf, how='intersection') #  Intersección de la DIMENSION con el BUFFER

            # paisaje_buf = dimension_buf.overlay(predio, how="difference") #  DIMENSION cortada por BUFFER agujereada con el predio = le llamamos "paisaje"
            paisaje_buf = dimension_buf

            paisaje_buf["area_Ha"] = paisaje_buf.area/10000 #  Agregamos columna de área en hectáreas

            paisaje_buf["indicador_parche"] = paisaje_buf["area_Ha"] * paisaje_buf["INDICADOR"] #  Calculo del indicador para ese paisaje
            #
            paisaje_indicador = paisaje_buf["indicador_parche"].sum()
            #
            #out = round((((paisaje_indicador) / (dimension_buf.area.sum()/10000)) - region_promedio) , 4)
            out = round((paisaje_indicador)/(dimension_buf.area.sum()/10000), 4)
            #out = round((paisaje_indicador), 4)
            valores_indicador.append(out)
        valores_indicador_dimensiones.append(valores_indicador)
    
        # Create separate figures for each dimension
    for i, dim in enumerate(AMBIENTES_DICT.values()):
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(distancias_buffer, valores_indicador_dimensiones[i], 
                label=dim['label'], color=dim['color_plot'], linewidth=2)
        
        # Add reference lines if available
                # Add regional average line if available
        if i < len(promedios_tabla):
            ax.axhline(y=promedios_tabla[i], color=dim['color_plot'], 
                      linestyle='-', linewidth=0.8, alpha=0.9,
                      label="Promedio regional")
            
        if 'pc_regiones_superior' in dim:
            ax.axhline(y=dim['pc_regiones_superior'], color=dim['color_plot'], 
                      linestyle='dashed', linewidth=0.5, alpha=0.7,
                      label="Percentil superior")
            
        if 'pc_regiones_inferior' in dim:
            ax.axhline(y=dim['pc_regiones_inferior'], color=dim['color_plot'], 
                      linestyle='dotted', linewidth=0.5, alpha=0.7, 
                      label="Percentil inferior")
        

        
        
        ax.legend(fontsize=8)
        ax.set_title(f'Valor de Conservación: {dim["label"]}')
        ax.set_xlabel('Distancia al predio (m)')
        ax.set_ylabel('Valor del indicador')
        ax.grid(False)
        
        # Display the figure in Streamlit
        st.pyplot(fig)
    
def load_ambientes():
    fragmentacion = gpd.read_file("./dimensiones/DIM_fragmentacion_TEM_celdas.shp")
    representacion = gpd.read_file("./dimensiones/DIM_representacion_MB_celdas.shp")
    filtrado = gpd.read_file("./dimensiones/DIM_filtradoH2O_MB_poligonos.shp")
    
    return fragmentacion, representacion, filtrado

def configure_page() -> None:
    st.set_page_config(page_title="Streamlit App", page_icon=":guardsman:", layout="wide")  # TODO: Cambiar nombre de la pestaña

def plot_predio_ambientes(predio):
    if predio is not None:
        predio = predio.to_crs(32721)  # Convertir a UTM zona 21S
        predio_extension = predio.copy()
        predio_extension["geometry"] = predio.buffer(15000)

        # crear BUFFER de la region alrededor del predio
        predio_region = predio.copy()
        predio_region["geometry"] = predio.buffer(10000)

        # Lista con buffers a usar
        distancias_buffer = [0, 5000, 10000]
            
        frag, rep, filt = load_ambientes()
        frag = frag.overlay(predio_extension, how='intersection')
        rep = rep.overlay(predio_extension, how='intersection')
        filt = filt.overlay(predio_extension, how='intersection')
        
        _lock = RLock()
        with _lock:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
            fig.set_facecolor('ghostwhite')
            
            # Ambiente 1
            ax1.set_title('Fragmentación')
            ax1.add_artist(ScaleBar(1))
            
            # Plot first map with custom legend
            cax1 = fig.add_axes([0.29, 0.1, 0.01, 0.15])  # [x, y, width, height]
            frag_plot = frag.plot(
                column='INDICADOR',
                ax=ax1,
                cmap='Greens',
                vmin=0,
                vmax=1,
                legend=False
            )
            fig.colorbar(frag_plot.collections[0], cax=cax1)
            
            for buf in distancias_buffer:
                #  Crear BUFFER del predio
                predio_buf = predio.copy()
                predio_buf["geometry"] = predio.buffer(buf)

                # Plot buffer
                buf_color = 'red' if buf==0 else 'gray'
                predio_buf.boundary.plot(ax=ax1, color=buf_color)

            ax1.set_xlim(predio_buf.total_bounds[0]-500, predio_buf.total_bounds[2]+500)
            ax1.set_ylim(predio_buf.total_bounds[1]-500, predio_buf.total_bounds[3]+500)
            
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Ambiente 2
            ax2.set_title('Representación')
            ax2.add_artist(ScaleBar(1))
            
            # Plot second map with custom legend
            cax2 = fig.add_axes([0.615, 0.1, 0.01, 0.15]) # [x, y, width, height]
            rep_plot = rep.plot(
                column='INDICADOR',
                ax=ax2,
                cmap='Greys',
                vmin=0,
                vmax=1,
                legend=False
            )
            fig.colorbar(rep_plot.collections[0], cax=cax2)
            
            for buf in distancias_buffer:
            #  Crear BUFFER del predio
                predio_buf = predio.copy()
                predio_buf["geometry"] = predio_buf.buffer(buf)

                # Plot buffer
                buf_color = 'red' if buf==0 else 'gray'
                predio_buf.boundary.plot(ax=ax2, color=buf_color)

            ax2.set_xlim(predio_buf.total_bounds[0]-500, predio_buf.total_bounds[2]+500)
            ax2.set_ylim(predio_buf.total_bounds[1]-500, predio_buf.total_bounds[3]+500)
            
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Ambiente 3
            ax3.set_title('Filtrado')
            ax3.add_artist(ScaleBar(1))
            
            # Plot second map with custom legend
            cax3 = fig.add_axes([0.937, 0.1, 0.01, 0.15]) # [x, y, width, height]
            filt_plot = filt.plot(
                column='INDICADOR',
                ax=ax3,
                cmap='Blues',
                vmin=0,
                vmax=1,
                legend=False
            )
            fig.colorbar(filt_plot.collections[0], cax=cax3)
            
            for buf in distancias_buffer:
            #  Crear BUFFER del predio
                predio_buf = predio.copy()
                predio_buf["geometry"] = predio_buf.buffer(buf)

                # Plot buffer
                buf_color = 'red' if buf==0 else 'gray'
                predio_buf.boundary.plot(ax=ax3, color=buf_color)

            ax3.set_xlim(predio_buf.total_bounds[0]-500, predio_buf.total_bounds[2]+500)
            ax3.set_ylim(predio_buf.total_bounds[1]-500, predio_buf.total_bounds[3]+500)
            
            ax3.set_xticks([])
            ax3.set_yticks([])
            
            # Adjust spacing
            plt.tight_layout(rect=[0, 0, 0.98, 1])
            
            st.pyplot(fig)
            
def plot_predio_ambientes_largo(predio):
    if predio is not None:
        _lock = RLock()
        with _lock:

            predio = predio.to_crs(32721)  # Convertir a UTM zona 21S
            predio_extension = predio.copy()
            predio_extension["geometry"] = predio.buffer(15000)

            # crear BUFFER de la region alrededor del predio
            predio_region = predio.copy()
            predio_region["geometry"] = predio.buffer(10000)

            # Lista con buffers a usar
            distancias_buffer = [0, 5000, 10000]

            # este loopea por todas las dimensiones
            for dim in DIMENSIONES_DICT.values():
                dimension = gpd.read_file(dim['path'])
                dimension = dimension.overlay(predio_extension, how='intersection')

                fig, ax = plt.subplots()


                # Remove axis ticks and labels
                ax.set_xticks([])
                ax.set_yticks([])

                # Optionally remove axis spines for a cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                # este loop calcula el indicador para cada buffer de una dimensión
                for buf in distancias_buffer:
                    #  Crear BUFFER del predio
                    predio_buf = predio.copy()
                    predio_buf["geometry"] = predio.buffer(buf)


                    # Plot buffer
                    buf_color = 'black' if buf==0 else 'gray'
                    predio_buf.boundary.plot(ax=ax, color=buf_color)

                ax.set_xlim(predio_buf.total_bounds[0]-500, predio_buf.total_bounds[2]+500)
                ax.set_ylim(predio_buf.total_bounds[1]-500, predio_buf.total_bounds[3]+500)
                
                st.pyplot(fig)


def configure_overview() -> None:
    st.markdown("# Indicador de Paisaje")  #TODO: Cambiar nombre del título
    st.markdown(
        "Indicador etc."
    )
    st.markdown(
        "pin pun pan"
    )
    
        
    with st.container():
        st.markdown("""
            > « Lorem ipsum dolor sit amet, consectetur adipisci elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. » 
        """
        )
  
    with st.container():
        df = pd.DataFrame(
        {
            "Command": ["***Aumentar Cobertura Vegetal***", "***Mejorar Integridad de Cobertura***"],
            "Type": ["Generar o mejorar zona de amortiguación no ribereña", "Mejorar zona buffer ribereña"],
        }
    )
    st.table(df)

    # # Create a download button for fig1
    # buffer = BytesIO()
    # fig1.savefig(buffer, format="png")
    # buffer.seek(0)
    
    # st.download_button(
    #     label="Descargar Mapa 1",
    #     data=buffer,
    #     file_name="environment_map.png",
    #     mime="image/png",
    # )

def configure_form(form_values):
    with st.form("Datos del campo", clear_on_submit=False):
                    
        st.markdown("### 2. Condicionantes según características generales del predio")
        form_values["r21"] = st.radio(
            "2.1 ¿Cuerpo de agua (semi)permanente dentro del predio o cercano?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r22"] = st.radio(
            "2.2 Proporción de ambientes naturales continuos (tamaño efectivo de la malla)",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r23"] = st.radio(
            "2.3 ¿Contiene humedal (o debería)?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r24"] = st.radio(
            "2.4 ¿Contiene bosque (o debería)?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r25"] = st.radio(
            "2.5 ¿Contiene pastizal (o debería)?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        
        st.markdown("### 3. Condicionantes según diagnóstico ambiental")
        form_values["r31"] = st.radio(
            "3.1 Proporción de zona riparia con cobertura vegetal natural",
            ("Baja", "Media", "Alta", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=4
        )
        form_values["r32"] = st.radio(
            "3.2 Estructura de la vegetación en zona buffer",
            ("Baja", "Media", "Alta", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=4
        )
        form_values["r33"] = st.radio(
            "3.3 ¿Erosión observada o riesgo de erosión alto?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r34"] = st.radio(
            "3.4 ¿El ganado accede a cuerpos de agua?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r35"] = st.radio(
            "3.5 ¿Uso de agroquímicos conalto riesgo de contaminación?",
            ("Sí", "No", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=3
        )
        form_values["r36"] = st.radio(
            "3.6 Continuidad de parches de ecosistemas naturales",
            ("Baja", "Media", "Alta", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=4
        )
        form_values["r37"] = st.radio(
            "3.7 Integridad ecológica de parches de ecosistemas naturales",
            ("Baja", "Media", "Alta", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=4
        )
        form_values["r38"] = st.radio(
            "3.8 Proporción de ecosistemas naturales en el predio",
            ("Baja", "Media", "Alta", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=4
        )
        form_values["r39"] = st.radio(
            "3.9 Estado de conservación de parches",
            ("Baja", "Media", "Alta", "Sin Dato", "Ninguna"),
            horizontal=True,
            index=4
        )
        
        st.form_submit_button("Guardar respuestas")
    
def configure_sidebar():
    with st.sidebar:
        st.markdown("# Entrada de datos")
        st.markdown("## 1. Primera parte")
        uploaded_file = st.file_uploader("Subir archivo KML", type=["kml", "kmz"])
        if uploaded_file is not None:
            predio = process_uploaded_file(uploaded_file)
            if predio is not None:
                _lock = RLock()
                with _lock:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title("Mapa del Predio")
                    predio.plot(ax=ax, color='lightblue', edgecolor='black')
                    ax.set_xlabel("Longitud")
                    ax.set_ylabel("Latitud")
                    st.pyplot(fig)
                st.success("Archivo procesado correctamente.")

# def process_uploaded_file(uploaded_file):
#     if uploaded_file is not None:
#         string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         df = pd.read_csv(string_data)
#         return df
#     else:
#         return None
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        #string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        predio = gpd.read_file(uploaded_file)
        return predio
    else:
        return None


