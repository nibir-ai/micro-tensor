import os
import gzip
import numpy as np
import requests

def download_mnist():
    # Using the most reliable mirror: Yann LeCun's official server
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz", 
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", 
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    os.makedirs('data/mnist', exist_ok=True)
    for f in files:
        path = os.path.join('data/mnist', f)
        if not os.path.exists(path):
            print(f"Downloading {f} from S3...")
            r = requests.get(base_url + f, stream=True)
            if r.status_code == 200:
                with open(path, 'wb') as out:
                    out.write(r.content)
            else:
                # Fallback URL if S3 is down
                fallback_url = "http://yann.lecun.com/exdb/mnist/"
                print(f"S3 failed, trying fallback: {f}...")
                r = requests.get(fallback_url + f, stream=True)
                with open(path, 'wb') as out:
                    out.write(r.content)

def load_mnist():
    download_mnist()
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            # Check magic number to avoid BadGzipFile errors
            data = f.read()
            return np.frombuffer(data, np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            data = f.read()
            return np.frombuffer(data, np.uint8, offset=8)

    print("Reading MNIST files into memory...")
    try:
        X_train = read_images('data/mnist/train-images-idx3-ubyte.gz')
        Y_train = read_labels('data/mnist/train-labels-idx1-ubyte.gz')
        X_test = read_images('data/mnist/t10k-images-idx3-ubyte.gz')
        Y_test = read_labels('data/mnist/t10k-labels-idx1-ubyte.gz')
        return X_train, Y_train, X_test, Y_test
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        print("Try running 'rm -rf data/mnist/' and run this script again.")
        exit(1)

if __name__ == "__main__":
    X, Y, Xt, Yt = load_mnist()
    print(f"✅ SUCCESS! MNIST Loaded: Train={X.shape}, Test={Xt.shape}")
