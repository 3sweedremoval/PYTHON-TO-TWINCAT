import cv2
import numpy as np
import pyads
import csv
import time

# Functie om coördinaten naar TwinCAT te verzenden
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="5.143.138.190.1.1", port=851):
    try:
        plc = pyads.Connection(plc_address, port)
        plc.open()

        if len(x_coords) != len(y_coords):
            print("Aantal X- en Y-coördinaten komt niet overeen.")
            return

        plc.write_by_name("MAIN.xCordinates", x_coords, pyads.PLCTYPE_INT * len(x_coords))
        plc.write_by_name("MAIN.yCordinates", y_coords, pyads.PLCTYPE_INT * len(y_coords))

        print("X- en Y-coördinaten verzonden!")
    except Exception as e:
        print(f"Fout bij verzenden naar TwinCAT: {e}")
    finally:
        plc.close()

# Functie om live video te verwerken (elke seconde)
def process_video(interval=3):
    cap = cv2.VideoCapture(1)  # Gebruik de camera (0 voor standaardcamera)

    if not cap.isOpened():
        print("Kan camera niet openen.")
        return

    last_capture_time = time.time()  # Houd de tijd van de laatste capture bij

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Kan frame niet lezen.")
            break

        current_time = time.time()
        # Controleer of het interval is verstreken
        if current_time - last_capture_time >= interval:
            last_capture_time = current_time  # Reset de timer

            # Verklein frame (optioneel)
            scale_percent = 50
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            # Grijswaardenconversie en Gaussian Blur
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)

            # Thresholding om binaire afbeelding te maken
            _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

            # Vind contouren
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Zoek grootste contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Vereenvoudig contour
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Extracteer coördinaten
                x_coords = []
                y_coords = []
                for point in simplified_contour:
                    x, y = point[0]
                    x_coords.append(x)
                    y_coords.append(y)

                # Teken contour op frame
                cv2.drawContours(resized_frame, [simplified_contour], -1, (0, 255, 0), 3)

                # Verzenden naar TwinCAT
                send_coordinates_to_twincat(x_coords, y_coords)

                # Opslaan in CSV
                save_coordinates_to_csv(list(zip(x_coords, y_coords)), "live_coordinates.csv")

                # Opslaan van het huidige frame (optioneel)
                #cv2.imwrite(f"frame_{int(current_time)}.jpg", resized_frame)

                print(f"Frame vastgelegd en verwerkt op {time.ctime(current_time)}.")

            # Toon het frame met contouren
            cv2.imshow("Live Contour Detection", resized_frame)

        # Druk op 'q' om te stoppen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Video-opname afsluiten
    cap.release()
    cv2.destroyAllWindows()

# Functie om coördinaten naar CSV op te slaan
def save_coordinates_to_csv(coordinates, filename="coordinates.csv"):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y'])
            for coord in coordinates:
                writer.writerow(coord)
        print(f"Coördinaten opgeslagen in {filename}")
    except Exception as e:
        print(f"Fout bij opslaan in CSV: {e}")

# Start live videoverwerking (elke seconde een capture)
process_video(interval=3)
