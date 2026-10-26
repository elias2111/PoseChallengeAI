from pathlib import Path
import kagglehub

DATASET = "niharika41298/yoga-poses-dataset"

output_dir = Path("dataset/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Descargando dataset...")
print("Esto se hace una sola vez.\n")

path = kagglehub.dataset_download(
    DATASET,
    output_dir=str(output_dir)
)

print("\n====================================")
print("DATASET DESCARGADO CORRECTAMENTE")
print("====================================")
print("Ubicacion:")
print(path)