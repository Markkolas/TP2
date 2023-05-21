import argparse
from API import *

def parse_opt(known=False):
    # Crea un objeto ArgumentParser
    parser = argparse.ArgumentParser()
    
# Agrega los argumentos que deseas procesar
    parser.add_argument('-help', action='store_true', help='Muestra la ayuda')
    parser.add_argument('-modelName', type=str, default=None, help='Selección de los pesos del modelo')
    parser.add_argument('-muestreo', type=int, default=5, help='Cada cuantos frames se pasan por el modelo')
    parser.add_argument('-cloud', action='store_true', help='Utiliza la opción en la nube')
    parser.add_argument('-tracking', action='store_true', help='Activa el seguimiento')
    parser.add_argument('-autonomo', action='store_true', help='Modo autónomo')
    parser.add_argument('-webcam', action='store_true', help='Utiliza la webcam en vez de usar el coche')

    # Parsea los argumentos de la línea de comandos
    args = parser.parse_known_args()[0] if known else parser.parse_args()

    return args

def print_help():
    modelos = send_help()
    print("\nDescripción del código:")
    print("\tEste código realiza acciones específicas basadas en los datos recibidos a través de un socket.\n")
    print("Argumentos disponibles:")
    print("\t-help: Muestra esta ayuda")
    print("\t-modelName: Selección de los pesos del modelo.")
    print("\t-muestreo: Cada cuantos frames se pasan por el modelo")
    print("\t-cloud: Utiliza la opción en la nube")
    print("\t-tracking: Activa el seguimiento")
    print("\t-autonomo: Modo autónomo")
    print("\t-webcam: Utiliza la webcam en vez de usar el coche")
    
    print("\nLos modelos disponibles en la nube son:")
    for modelo in modelos:
        print("\t",modelo)
    print("\nAdemás de los modelos de COCO: yolov5s yolov5m yolov5l yolov5x yolov5n yolov5n6 yolov5s6 yolov5m6 yolov5l6 yolov5x6")
    
    model_selection_options = []