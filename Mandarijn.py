import cv2
import numpy as np
import pyads
import csv

# Functie om X- en Y-coördinaten afzonderlijk naar TwinCAT te verzenden
def send_coordinates_to_twincat(x_coords, y_coords, plc_address="5.143.138.190.1.1", port=851):
    try:
        # Verbind met TwinCAT via ADS
        plc = pyads.Connection(plc_address, port)
        plc.open()

        # Controleer of de lengte van de coördinaten gelijk is
        if len(x_coords) != len(y_coords):
            print("Aantal X- en Y-coördinaten komt niet overeen.")
            return

        # Schrijf de X-coördinaten naar TwinCAT
        plc.write_by_name("MAIN.xCordinates", x_coords, pyads.PLCTYPE_INT * len(x_coords))

        # Schrijf de Y-coördinaten naar TwinCAT
        plc.write_by_name("MAIN.yCordinates", y_coords, pyads.PLCTYPE_INT * len(y_coords))

        print("X- en Y-coördinaten succesvol verzonden!")

    except Exception as e:
        print(f"Fout bij verzenden van coördinaten: {e}")
    finally:
        plc.close()

# Functie om de coördinaten naar een CSV-bestand te schrijven
def save_coordinates_to_csv(coordinates, filename="coordinates.csv"):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Schrijf de header (optioneel)
            writer.writerow(['X', 'Y'])
            # Schrijf de coördinaten
            for coord in coordinates:
                writer.writerow(coord)
        print(f"Coördinaten succesvol opgeslagen in {filename}")
    except Exception as e:
        print(f"Fout bij opslaan van coördinaten in CSV: {e}")

# Stap 1: Laad de afbeelding
image_path = "Small_gear flits v0.1.jpg"  # Vervang door het pad naar je afbeelding
image = cv2.imread(image_path)

# Stap 2: Verklein de afbeelding (optioneel)
scale_percent = 25  # Schaal de afbeelding naar 50% van de originele grootte
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Stap 3: Grijswaardenconversie en Gaussian Blur
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Stap 4: Thresholding om een binair beeld te krijgen
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

# Stap 5: Contouren vinden
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Controleer of er contouren zijn gevonden
if not contours:
    print("Geen contouren gevonden.")
    exit()

# Zoek de grootste contour (aangenomen dat dit het object is)
largest_contour = max(contours, key=cv2.contourArea)

# Optioneel: Vereenvoudig de contour om het aantal punten te beperken
epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # Pas aan voor meer/minder punten
simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

# Stap 6: Coördinaten extraheren
x_coords = []
y_coords = []
for point in simplified_contour:
    x, y = point[0]  # Extract (x, y)
    x_coords.append(x)
    y_coords.append(y)

# Teken de contour op de afbeelding (voor visualisatie)
result_image = resized_image.copy()
cv2.drawContours(result_image, [simplified_contour], -1, (0, 255, 0), 3)

# Toon het resultaat
cv2.imshow("Contour", result_image)
cv2.imwrite("contour_detected_resized.jpg", result_image)  # Sla het resultaat op
cv2.waitKey(0)
cv2.destroyAllWindows()

# Stap 7: Verzenden naar TwinCAT
send_coordinates_to_twincat(x_coords, y_coords)

# Stap 8: Opslaan in CSV-bestand
save_coordinates_to_csv(list(zip(x_coords, y_coords)))
