import os
import zipfile

def zip_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    zip_path = os.path.join(base_dir, 'dataset.zip')

    if not os.path.exists(dataset_dir):
        print(f" [ERROR] Dataset folder not found at: {dataset_dir}")
        return

    print(f" [INFO] Packaging dataset from '{dataset_dir}' into '{zip_path}'...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep directory structure relative to the dataset folder
                arcname = os.path.relpath(file_path, base_dir)
                zip_file.write(file_path, arcname)
                print(f"  Added {arcname}")
                
    print(f" [SUCCESS] Dataset successfully zipped into: {zip_path}")

if __name__ == '__main__':
    zip_dataset()
