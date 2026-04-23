import project
import os
# ── BATCH PROCESSING ────────────────────────────────────────────────────────
input_folder  = "photos"
output_folder = "classified_good"

os.makedirs(output_folder, exist_ok=True)
print("Starting batch processing...")

for i in range(1, 101):
    filename = f"{i}.jpg"
    in_path  = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)
    if os.path.exists(in_path):
        analyze_and_save_dice(in_path, out_path)

print("Done!")

def main():
    print("Hello from pyproj!")


if __name__ == "__main__":
    main()