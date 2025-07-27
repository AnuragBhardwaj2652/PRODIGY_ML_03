import os

folder_path = r"C:\Anurag\dataset\cat"  # ✅ Replace with your actual folder
x = 0

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg") and "cat" not in filename.lower(): # ✅ Replace with scan names
        old_path = os.path.join(folder_path, filename)
        new_filename = f"cat{x}.jpg" # ✅ Replace with your new names
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")
        x += 1