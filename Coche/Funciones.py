import requests as r
import json
import numpy as np
import cv2
import os
import subprocess
import threading
import torch
import ultralytics

from pydantic import BaseModel
from typing import List, Optional

def box_color():
    # Colores para dibujar las cajas
    colors = [
        (31, 119, 180),   (255, 127, 14),    (44, 160, 44),    (214, 39, 40),
        (148, 103, 189),  (140, 86, 75),     (227, 119, 194),  (127, 127, 127),
        (188, 189, 34),   (23, 190, 207),    (174, 199, 232),  (255, 187, 120),
        (152, 223, 138),  (255, 152, 150),   (197, 176, 213),  (196, 156, 148),
        (247, 182, 210),  (199, 199, 199),   (219, 219, 141),  (158, 218, 229),
        (199, 199, 199),  (255, 187, 120),   (196, 156, 148),  (23, 190, 207),
        (158, 218, 229),  (174, 199, 232),   (197, 176, 213),  (255, 152, 150),
        (227, 119, 194),  (127, 127, 127),   (140, 86, 75),    (148, 103, 189),
        (214, 39, 40),    (44, 160, 44),     (255, 127, 14),   (31, 119, 180),
        (255, 187, 120),  (152, 223, 138),   (219, 219, 141),  (199, 199, 199),
        (196, 156, 148),  (247, 182, 210),   (158, 218, 229),  (23, 190, 207),
        (174, 199, 232),  (197, 176, 213),   (255, 152, 150),  (227, 119, 194),
        (127, 127, 127),  (140, 86, 75),     (148, 103, 189),  (214, 39, 40),
        (44, 160, 44),    (255, 127, 14),    (31, 119, 180),   (255, 187, 120),
        (152, 223, 138),  (219, 219, 141),   (199, 199, 199),  (196, 156, 148),
        (247, 182, 210),  (158, 218, 229),   (23, 190, 207),   (174, 199, 232),
        (197, 176, 213),  (255, 152, 150),   (227, 119, 194),  (127, 127, 127),
        (140, 86, 75),    (148, 103, 189),   (214, 39, 40),    (44, 160, 44),
        (255, 127, 14),   (31, 119, 180),    (255, 187, 120),  (152, 223, 138),
        (219, 219, 141),  (199, 199, 199),   (196, 156, 148),  (247, 182, 210),
        (158, 218, 229),  (23, 190, 207),    (174, 199, 232),  (197, 176, 213),
        (255, 152, 150),  (227, 119, 194),   (127, 127, 127),  (140, 86, 75),
        (148, 103, 189),  (214, 39, 40),     (44, 160, 44),    (255, 127, 14),
        (31, 119, 180),   (255, 187, 120),   (152, 223, 138),  (219, 219, 141),
        (199, 199, 199),  (196, 156, 148),   (247, 182, 210),  (158, 218, 229),
        (23, 190, 207),   (174, 199, 232),   (197, 176, 213),  (255, 152, 150),
        (227, 119, 194),  (127, 127, 127),   (140, 86, 75),    (148, 103, 189),
        (214, 39, 40),    (44, 160, 44),     (255, 127, 14),   (31, 119, 180)
    ]
    
    return colors

def plot_one_box(x, im, color=(128, 128, 128), label=None, confidence=None, tracker_id=None, line_thickness=2):
    # Función cogida del repositorio del repositorio de Ultralytics
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        if confidence:
            label = f'{label}: {confidence:.2f}'
        if tracker_id:
            label = f'{label} | ID: {tracker_id}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def send_help():
    # Enviamos la solicitud GET al punto final /custom_models
    res = r.get("http://143.47.38.177/custom_models")
    
    if res.ok and res.status_code == 200:
        # print("La solicitud se ha completado correctamente:", response.status_code)
        
        # Obtenemos los datos JSON de la respuesta
        json_data = res.json()
        data = json.loads(json_data)
        
        return data
    else:
        print("Error al enviar la solicitud:", res.status_code)
        
def t_proc():
    # Enviamos la solicitud GET al punto final /custom_models
    res = r.get("http://143.47.38.177/t_proc")
    
    if res.ok and res.status_code == 200:
       # Obtenemos los datos JSON de la respuesta
        data = res.json()
        media = data["media"]
        return media
    else:
        print("Error al enviar la solicitud:", res.status_code)

def send_request(file_list = [], 
                    model_name = 'yolov5s',
                    img_size = 640,
                    tracking = False):
    
    # Se podrían subir mas de una imagen a la vez, nosotros no hacemos uso de esta opción ya que enviamos una imagen en cada petición
    files = [('file_list', file) for file in file_list]

    # Se pasan los demas argumentos en esta variable
    other_form_data = {'model_name': model_name,
                    'img_size': img_size,
                    'tracking': tracking}

    # Se manda la petición y se reccogen los resultados
    res = r.post("http://143.47.38.177/detect", 
                    data = other_form_data, 
                    files = files)
    
    
    if res.ok and res.status_code == 200:
        # print("Todo ha ido bien: ", res.status_code)
        
        # Devolvemos el json que ha devuelto la respuesta
        json_data = res.json()
        data = json.loads(json_data)

        return data
    else:
        print("Error al enviar la solicitud: ", res.status_code)
        
def dibujar_caja(imagen, datos):
    # Dibuja todas las cajas en una imagen
    
    colors = box_color()

    if datos:
        for img_data in datos:
            for detection in img_data:
                # Obtener el color de la detección
                color=colors[int(detection['class'])]
                # Dibujar la caja y el label en la imagen
                if 'tracker_id' in detection:
                    plot_one_box(detection['bbox'], imagen, color=color, label=detection['class_name'], confidence=detection['confidence'], tracker_id=detection['tracker_id'])
                else:
                    plot_one_box(detection['bbox'], imagen, color=color, label=detection['class_name'], confidence=detection['confidence'])
                    
            return imagen
    else:
        return imagen
    
def local(imagen, model_name, img_size=None):
    
    print(type(imagen[0]))
    model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                            'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']
    model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

    # Obtener la lista de nombres de archivos en la carpeta
    file_names = os.listdir('modelos/')
    # Extensiones de modelos válidas
    valid_extensions = ['.torchscript', '.onnx', '_openvino_model', '.engine ', '.mlmodel', '_saved_model', '.pt', '.tflite', '_edgetpu.tflite', '_paddle_model ']
    
    # Filtrar solo los archivos con extensiones de modelos válidas
    model_names = [file_name for file_name in file_names if os.path.splitext(file_name)[1] in valid_extensions]
    # Crear el diccionario con los nombres de los modelos
    custom_model_dict = {model_name: None for model_name in model_names}

    if model_name in custom_model_dict:
        custom_model = True
    else:
        custom_model = False
        
    if model_name in model_dict:
        yolo_model = True
    else:
        yolo_model = False
    
    print(model_dict)
    print(custom_model_dict)
 
    if custom_model:
        if custom_model_dict[model_name] is None:
            custom_path = 'modelos/'+model_name
            print(custom_path)
            for clave in custom_model_dict.keys():
                print(clave)
            custom_model_dict[model_name] = torch.hub.load('ultralytics/yolov5', 'custom',  path=custom_path)
        results = custom_model_dict[model_name](imagen, size = img_size) 
        json_results = results_to_json(results,custom_model_dict[model_name])
    elif yolo_model:
        if model_dict[model_name] is None:
            model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        results = model_dict[model_name](imagen, size = img_size) 
        json_results = results_to_json(results,model_dict[model_name])
    else:
        print("El modelo elegido no esta disponible")
        exit()
        
    encoded_json_results = str(json_results).replace("'",r'"')
    return encoded_json_results

def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
            ]

def add_text_to_image(image, timer, datos, FPS):
    # Crear una imagen en blanco con el mismo tamaño que la imagen original
    overlay = np.zeros_like(image)

    # Definir las propiedades del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_thickness = 1
    line_type = cv2.LINE_AA

    # Obtener las dimensiones del texto
    (text_width, text_height), _ = cv2.getTextSize("Texto de ejemplo", font, font_scale, text_thickness)

    # Calcular la posición del rectángulo de fondo y del texto
    rect_x = image.shape[1] - text_width - 20
    rect_y = 30
    rect_width = text_width + 20
    rect_height = text_height*3 + 10
    text_x = rect_x + 10
    text_y = rect_y + text_height + 10
    
    # Dibujar el rectángulo de fondo en la capa de superposición
    cv2.rectangle(overlay, (rect_x, 0), (rect_x + rect_width, rect_y + rect_height), (100, 100, 100), -1)
    
    cv2.rectangle(overlay, (rect_x, 0), (rect_x + rect_width, rect_y + rect_height), (50, 50, 50), thickness=3, lineType=cv2.LINE_AA)
    
    # Agregar el texto a la capa de superposición
   
    if len(datos) > 1:
        cv2.putText(overlay, "FPS: " + f'{FPS:.2f}', (text_x, text_y - text_height - 10), font, font_scale, (1, 1, 1), text_thickness, line_type)
        cv2.putText(overlay, "RTT: " + f'{timer*1000:.2f}' + "ms", (text_x, text_y), font, font_scale, (1, 1, 1), text_thickness, line_type)
        cv2.putText(overlay, "T. proc: " + datos[1] + "ms", (text_x, text_y + text_height + 10), font, font_scale, (1, 1, 1), text_thickness, line_type)
    else:
        cv2.putText(overlay, "FPS: " + f'{FPS:.2f}', (text_x, text_y - text_height), font, font_scale, (1, 1, 1), text_thickness, line_type)
        cv2.putText(overlay, "T. proc: " + f'{timer*1000:.2f}' + "ms", (text_x, text_y + text_height), font, font_scale, (1, 1, 1), text_thickness, line_type)
    
    # Fusionar la capa de superposición con la imagen original
    image_with_text = cv2.add(image, overlay)

    return image_with_text