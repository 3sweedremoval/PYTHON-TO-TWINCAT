import cv2
import pyrealsense2 as rs
import numpy as np
import time
import csv  # Voor het opslaan van coördinaten

# Initialiseer RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Parameters voor Canny Edge
lower_threshold = 25
upper_threshold = 200

# Referentie pixels per mm (dit moet je kalibreren voor je camera)
pixels_per_mm_at_reference_distance = 2.45  # Dit is de referentie bij een bepaalde afstand, bijvoorbeeld 1 meter

# Referentieafstand in meters (afstand van de camera tot het object bij kalibratie)
reference_distance_m = 0.25

# Minimale en maximale diameter in mm
min_diameter_mm = 40
max_diameter_mm = 180

# CSV-bestand voor de coördinaten
csv_filename = "contour_coordinates_mm.csv"

# Live processing loop
try:
    while True:
        # Wacht een tijdje voor betere belichting
        time.sleep(1)

        # Neem een enkele frame op
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

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
                        # Converteer contourcoördinaten naar millimeters
                        contour_mm = []
                        for point in largest_contour:
                            pixel_x, pixel_y = point[0]
                            mm_x = pixel_x / pixels_per_mm
                            mm_y = pixel_y / pixels_per_mm
                            contour_mm.append((mm_x, mm_y))

                        # Opslaan in een CSV-bestand
                        with open(csv_filename, mode="w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["X (mm)", "Y (mm)"])
                            writer.writerows(contour_mm)

                        print(f"Coördinaten opgeslagen in {csv_filename}")

                        # Markeer het startpunt van de contour
                        start_point_pixel = largest_contour[0][0]  # Eerste punt van de contour in pixels
                        start_x_mm = start_point_pixel[0] / pixels_per_mm
                        start_y_mm = start_point_pixel[1] / pixels_per_mm

                        # Teken het startpunt op de afbeelding
                        cv2.circle(color_image, (int(start_point_pixel[0]), int(start_point_pixel[1])), 5, (0, 0, 255), -1)  # Rode cirkel
                        cv2.putText(color_image, f"Start: ({start_x_mm:.2f}, {start_y_mm:.2f}) mm", 
                                    (int(start_point_pixel[0]) + 10, int(start_point_pixel[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Log de startpositie
                        print(f"Startpunt contour: X = {start_x_mm:.2f} mm, Y = {start_y_mm:.2f} mm")

                        # Teken de grootste contour en de cirkel om de contour
                        cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 2)  # Groene contourlijn
                        cv2.circle(color_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)  # Blauwe cirkel
                        cv2.putText(color_image, f"Diameter: {diameter_mm:.2f} mm", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # Print het resultaat in de console
                        print(f"Diameter van het tandwiel: {diameter_mm:.2f} mm op hoogte {depth_at_center:.2f} m")

        # Toon het resultaat in real-time
        cv2.imshow("Tandwiel Resultaat", color_image)

        # Wacht op een toets om de loop te stoppen (bijvoorbeeld 'q' om te stoppen)
        key = cv2.waitKey(1)  # 1 milliseconde wacht
        if key == ord('q'):
            break

finally:
    # Stop de pipeline en sluit vensters
    pipeline.stop()
    cv2.destroyAllWindows()
