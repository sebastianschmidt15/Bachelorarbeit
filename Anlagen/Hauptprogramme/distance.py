import pyrealsense2 as rs
import numpy.core.multiarray
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Methode zum Umwandlen der Bildkoordinaten in 3D-Koordinaten
def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    # Auslesen der intrinsischen Parameter
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.ppx
    _intrinsics.ppy = cameraInfo.ppy
    _intrinsics.fx = cameraInfo.fx
    _intrinsics.fy = cameraInfo.fy
    _intrinsics.model = cameraInfo.model
    _intrinsics.coeffs = cameraInfo.coeffs

    # Berechnen der Koordinaten und Speichern in Array
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)

    # result[0]: right, result[1]: down, result[2]: forward
    return result[0], -result[1], -result[2]

# Transformationsmatrix definieren
# Falls sich die Position der Kamera verändert hat, muss die Matrix angepasst werden
# Durch Eingeben der Standardmatrix [[1 0 0 0]
#                                    [0 1 0 0]
#                                    [0 0 1 0]]
# werden die Kamerakoordinaten beibehalten und nur die Transformation auf die Bechermitte und -höhe durchgeführt
TRANSFORMATION = np.array([ [1,   0,   0,   0],
                            [0,   1,   0,   0],
                            [0,   0,   1,   0]])

# Thresholds für Objekerkennung definieren
CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Pfade zu Dateien definieren
config_path = "yolov4/yolov4.cfg"
weights_path = "yolov4/yolov4.weights"

# Laden der Labels für die Klassen (hier nur Becher)
labels = open("yolov4/obj.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Laden des neuronalen Netzwerks
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Starten des Videostreams der Tiefenkamera
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 1280x720 Pixel, 16 Bit Tiefenwerte, 30 FPS
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 1280x720 Pixel, 8 Bit BGR-Farbwerte, 30 FPS
profile = pipe.start(cfg)

# Einstellen des 'High Accuracy' Modus
depth_sensor = profile.get_device().first_depth_sensor()
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
for i in range(int(preset_range.max)):
    visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
    print('%02d: %s'%(i,visulpreset))
    if visulpreset == "High Accuracy":
        depth_sensor.set_option(rs.option.visual_preset, i)
# Erhöhen der Laserpower des Emitters
depth_sensor.set_option(rs.option.laser_power, 210)
# Veringern der kleinsten Tiefeneinheit für genauere Ergebnisse
depth_sensor.set_option(rs.option.depth_units, 0.0005)

# Die ersten Frames überspringen, damit die Belichtung sich einstellen kann
for x in range(10):
    pipe.wait_for_frames()

# Schleife für das Auslesen und Erkennen der Bilder
while(1):

    # Aktuelle frames werden gespeichert
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Überprüfen, ob sowohl Tiefen- als auch Farbbild vorhanden sind
    if color_frame and depth_frame:

        # Die beiden Frames der Kamera zueinander ausrichten
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Neuen Farbframe holen und für Weiterverarbeitung als numpy-Array speichern
        color_frame = frameset.get_color_frame()
        color = np.asanyarray(color_frame.get_data())

        # Tiefenwerte mit Hilfe des colorizers farbig darstellen und als numpy-Array speichern
        colorizer = rs.colorizer()
        aligned_depth_frame = frameset.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # Schneiden des Farbbilds, um aus 16:9 ein 1:1 Format zu machen
        # Ränder enthalten keine wichtigen Informationen und können deshalb problemlos abgeschnitten werden
        color_cropped = color[0:720, 280:1000]
        crop_img = cv2.resize(color_cropped, (640, 640))

        # Definieren der Klassennamen und Erstellen eines Blobs für das neuronale Netz
        classNames = ("cup")
        h, w = crop_img.shape[:2]
        blob = cv2.dnn.blobFromImage(crop_img, 1/255.0, (640, 640), swapRB=True, crop=False)
        
        # Setzen des Eingangs
        net.setInput(blob)

        # Abholen der Schichten
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Starten der Erkennung und Messung der Dauer
        start = time.perf_counter()
        layer_outputs = net.forward(ln)
        time_took = time.perf_counter() - start
        print(f"Time took: {time_took:.2f}s")

        # Festlegen der Schriftgröße etc.
        font_scale = 1
        thickness = 2
        boxes, confidences, class_ids = [], [], []

        # Durchlaufen der verschiedenen outputs
        for output in layer_outputs:

            # Durchlaufen jedes erkannten Objekts
            for detection in output:

                # ClassId und confidence für das erkannte Objekt auslesen
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Prüfen, ob confidence über dem angegebenen Threshold liegt
                if confidence > CONFIDENCE:

                    # Koordinaten des erkannten Objekts auslesen
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Obere linke Ecke berechnen
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Listen mit bounding boxes, confidenecs und classIds updaten
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS (non maximum suppression) durchführen um doppelte Erkennungen auszuschließen
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        # Setzen der Fläche und des unteren Rands auf 0
        # Wird zum Prüfen des vorderen Bechers benötigt
        area = 0.0
        lowest = 0

        # Rüfen, ob eine Erkennung vorhanden ist
        if len(idxs) > 0:

            # Alle mit der NMS nicht entfernten Objekte durchlafen
            for i in idxs.flatten():

                # Position, Höhe und Breite auslesen
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # Zeichnen der bounding box
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(crop_img, (x, y), (x + w, y + h), color=color, thickness=thickness)
                text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

                # Textgröße bestimmen, transparente Box um den Text zeichnen und Text einfügen
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = crop_img.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                crop_img = cv2.addWeighted(overlay, 0.6, crop_img, 0.4, 0)
                cv2.putText(crop_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

                # Prüfen, ob der Becher am weitesten unten im Bild zu sehen ist
                if (y+h) > lowest:
                    lowest = y + h

                    # Prüfen, ob die bounding box des Bechers die größte Fläche besitzt
                    # Besitzt der Becher die größte Fläche und ist am weitesten unten zu sehen, steht er höchstwahrscheinlich vorne
                    # Zusätzlich wird geprüft, ob die Fläche über 85000. Liegt die Fläche darüber, ist es sehr wahrscheinlich eine Falscherkennung
                    if (w*h) > area and (w*h)<85000:
                        # Speichern der Werte und Index
                        area = w*h
                        lowest = y + h
                        nearest = i

            # Auslesen der Position, Höhe und Breite des vorderen Bechers
            x, y = boxes[nearest][0], boxes[nearest][1]
            w, h = boxes[nearest][2], boxes[nearest][3]

            # Bestimmen der Mitte des Bechers
            x_mitte = x + w/2
            y_mitte = y + h/2

            # Die Pixel des 640x640 Bilds zurück auf die Originalgröße rechnen, damit die Pixel mit dem Tiefenbild übereinstimmen
            x_depth = int( int(x_mitte) * 1.125 + 280 )
            y_depth = int( int(y_mitte) * 1.125 )

            # Berechnen der Distanz zur Kamera und Abholen der intrinsichen Parameter
            dist = aligned_depth_frame.get_distance(x_depth, y_depth)
            camera_info = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            # Berechnen der Kamerakoordinaten und Umrechnen in Weltkoordinaten mit Hilfe der vorher bestimmten Transformationsmatrix
            x_w, y_w, z_w = convert_depth_to_phys_coord_using_realsense(x_depth, y_depth, dist, camera_info)
            coord_vector = np.array([x_w, y_w, z_w, 1])
            world_coords = TRANSFORMATION.dot(coord_vector)

            # Distanz von Becherrand bis Mitte des Bechers berechnen
            offset_z = (0.062 + 0.00748 * world_coords[1]) / 2
            z_new = world_coords[2] - offset_z

            # Festlegen der Strings für Kamera- und Weltkoordinaten
            coords_string = ("Kamera: (%.3f,%.3f,%.3f)" %(x_w, y_w, z_w)) 
            # Für die Höhe wird 0.12 verwendet, da die Becher immer 12cm hoch sind und für die Tiefe die eben berechnete Mitte des Bechers
            world_coords_string = ("Welt: (%.3f,%.3f,%.3f)" %(world_coords[0], 0.12, z_new))

            # Zeichnen eines Punkts auf dem Tiefenbild um Stelle der Messung zu markieren
            cv2.circle(colorized_depth, (x_depth, y_depth), 5, (255,255,255), thickness=10)

            # Einfügen der Koordinaten unterhalb des Bechers
            # Oben: Kamerakoordinaten (am Messpunkt)   Unten: Weltkoordinaten (Mitte von Becher auf 12cm Höhe)
            cv2.putText(crop_img, coords_string, (x - 50, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 0), thickness=1)
            cv2.putText(crop_img, world_coords_string, (x - 50, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 0), thickness=1)

        # Tiefenbild für das Anzeigen auf die gleiche Größe wie das Farbbild bringen
        cropped_depth = cv2.resize(colorized_depth[0:720, 280:1000], (640, 640))
        
        # Die beiden Bilder in eine, Stack nebeneinander legen
        images = np.hstack((crop_img, cropped_depth))

        # Bilder anzeigen und kurz warten
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
    else:
        print("Image is null")

# Streams der Kamera stoppen
pipe.stop()