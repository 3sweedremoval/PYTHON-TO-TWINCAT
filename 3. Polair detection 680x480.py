import cv2                  # OpenCV bibliotheek. Voor beeldverwerking zoals randdetectie en contourherkenning
import pyrealsense2 as rs   # Binding voor de Intel RealSense camera. Voor verwerking diepte en kleur
import numpy as np          # Bibliotheek voor numerieke berekeningen. Arrays en wiskundige berekeningen etc.
import time                 # Python module: tijdfunctie voor nemen van foto
import csv                  # Gegevens lezen en schrijven naar CSV bestanden
import pyads                # Python module voor het communiceren met het ADS protocol van Beckhoff. Uitwisselen gegevens etc.

# Coördinaten naar TwinCAT PLC sturen
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="39.231.85.117.1.1", port=851):
    try:
        # Verbinding maken met plc
        plc = pyads.Connection(plc_address, port)
        plc.open()

        # Controleer of de verbinding open is
        if plc.is_open:
            print(f"Connected to PLC at {plc_address} on port {port}")

            # Schrijf de coördinaten naar de PLC
            for i, (x, angle) in enumerate(zip(x_coords, y_coords)):
                if i < 9999:  # Zorg ervoor dat de index binnen het bereik valt
                    plc.write_by_name(f'Main.x_coords[{i + 1}]', x, pyads.PLCTYPE_REAL)  # X in mm
                    plc.write_by_name(f'Main.y_coords[{i + 1}]', angle, pyads.PLCTYPE_REAL)  # Y in radialen
                else:
                    print(f"Index {i} out of range, skipping.")

            print(f"Coördinaten succesvol verzonden naar PLC.")

        else:
            print("Failed to open connection to PLC.")

        plc.close()

    except Exception as e:
        print(f"Error: {e}")

# Initialiseer Intel RealSense D435 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Parameters voor Canny Edge (aanpassen als die hem niet pakt)
lower_threshold = 25
upper_threshold = 150

# Referentie pixels per mm (dit moet je kalibreren voor je camera, afhankelijk gebruikte resolutie ook)
pixels_per_mm_at_reference_distance = 2.45  

# Referentieafstand in meters (afstand van de camera tot het object bij kalibratie)
reference_distance_m = 0.25

# Minimale en maximale diameter in mm van het tandwiel/object
min_diameter_mm = 40
max_diameter_mm = 180

# CSV-bestand voor de coördinaten
csv_filename = "contour_coordinates_mm_&_rad.csv"

# Hoofdprogramma
try:
    while True:
        # Ivm verlichting 2 seconden wachten
        time.sleep(2)

        # Neem een enkele frame op
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # geef terugmelding als camera niet werkt
        if not color_frame or not depth_frame:
            print("Geen frames ontvangen. Probeer opnieuw.")
            continue

        # Converteer naar NumPy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Voorverwerking van het kleurbeeld
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)

        # Canny Edge-detectie
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

        # Contouren vinden
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Grootste contour selecteren (de buitenste contour)
            largest_contour = max(contours, key=cv2.contourArea)

            # Omcirkel de grootste contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

            # Controleer of de contour voldoende rond is
            circle_area = np.pi * (radius ** 2)
            contour_area = cv2.contourArea(largest_contour)
            if 0.7 < contour_area / circle_area < 1.0:  # Controleer of de contour bijna een cirkel is
                # Haal de diepte op het midden van de contour
                depth_at_center = depth_frame.get_distance(int(x), int(y))  # Diepte in meters

                if depth_at_center > 0:
                    # Bereken de nieuwe pixels per mm op basis van de diepte
                    pixels_per_mm = pixels_per_mm_at_reference_distance * (reference_distance_m / depth_at_center)

                    # Bereken de werkelijke diameter van het tandwiel
                    diameter_mm = (2 * radius) / pixels_per_mm

                    # Controleer of de diameter binnen het bereik ligt
                    if min_diameter_mm <= diameter_mm <= max_diameter_mm:
                        # Bereken het middelpunt van de contour in millimeters
                        center_x_mm = x / pixels_per_mm
                        center_y_mm = y / pixels_per_mm

                        # Converteer contourcoördinaten naar x (mm) en y (radialen)
                        x_coords_mm = []
                        y_coords_rad = []
                        for point in largest_contour:
                            pixel_x, pixel_y = point[0]
                            mm_x = pixel_x / pixels_per_mm
                            mm_y = pixel_y / pixels_per_mm

                            # Bereken de hoek in radialen
                            angle = np.arctan2(mm_y - center_y_mm, mm_x - center_x_mm)
                            if angle < 0:
                                angle += 2 * np.pi  # Converteer naar het bereik [0, 2π]

                            x_coords_mm.append(mm_x)
                            y_coords_rad.append(angle)

                        # Opslaan in een CSV-bestand
                        with open(csv_filename, mode="w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["X (mm)", "Y (rad)"])
                            writer.writerows(zip(x_coords_mm, y_coords_rad))

                        print(f"Coördinaten opgeslagen in {csv_filename}")

                        # Coördinaten naar TwinCAT sturen
                        send_coordinates_to_twincat(x_coords_mm, y_coords_rad)

                        # Teken de grootste contour en de cirkel om de contour
                        cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 2)  # Groene contourlijn
                        cv2.circle(color_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)  # Blauwe cirkel
                        cv2.putText(color_image, f"Diameter: {diameter_mm:.2f} mm",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # Toon resultaat en sla afbeelding op
                        cv2.imshow("Tandwiel Resultaat", color_image)
                        cv2.imwrite("resultaat_tandwiel.png", color_image)
                        cv2.waitKey(0)  # Wacht op een toets om af te sluiten
                        break
                    else:
                        print(f"Contour genegeerd: diameter ({diameter_mm:.2f} mm) ligt niet tussen {min_diameter_mm} mm en {max_diameter_mm} mm.")
                else:
                    print("Geen object diepte gevonden.")
        else:
            print("Geen contouren gevonden. Controleer het beeld en probeer opnieuw.")

finally:
    # Stop het programma en sluit het vensters
    pipeline.stop()
    cv2.destroyAllWindows()
