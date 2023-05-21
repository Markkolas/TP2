import torch
import cv2
import os
from sort import Sort
from API import *

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
    
if __name__ == '__main__':
    # Cargar el modelo YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.float()
    model.eval()

    # Crear la instancia de SORT
    mt_tracker = Sort()

    # Configuración de la webcam
    webcam = cv2.VideoCapture(0)  # Usar el ID 0 para la primera cámara

    while True:
        # Leer el cuadro de la webcam
        ret, frame = webcam.read()

        # Obtener las detecciones
        preds = model(frame)
        detections = preds.pred[0].numpy()

        # Actualizar SORT
        track_bbs_ids = mt_tracker.update(detections)

        # Mostrar los resultados y agregar el ID actualizado a json_results
        json_results = results_to_json(preds, model)
        
        if len(track_bbs_ids) > 0:
            for j in range(len(track_bbs_ids.tolist())):
                coords = track_bbs_ids.tolist()[j]
                
                # Agregar el ID actualizado a json_results
                json_results[0][j]['tracker_id'] = int(coords[4])
        
        imagen = dibujar_caja(frame, json_results)

        # Mostrar la imagen con los resultados
        cv2.imshow('Webcam', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos de la webcam y cerrar la ventana
    webcam.release()
    cv2.destroyAllWindows()


