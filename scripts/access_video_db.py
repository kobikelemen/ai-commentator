import lmdb
import numpy as np
import msgpack
import io
from msgpack_numpy import patch as msgpack_numpy_patch

# This is necessary for unpacking numpy arrays serialized with MessagePack
msgpack_numpy_patch()

def read_features_from_lmdb(lmdb_path, video_key):
    # Open the LMDB database
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        # Retrieve the binary data by key
        binary_data = txn.get(key.encode('utf-8'))
        if binary_data is None:
            print(f"Video key {video_key} does not exist in the database.")
            return None

        # Assuming the data was stored using msgpack
        try:
            # Deserialize the data
            data = msgpack.unpackb(binary_data, raw=False)
            # If data was compressed, you'll need additional steps to decompress
        except Exception as e:
            print(f"Error unpacking data: {e}")
            return None

        return data

lmdb_path = 
key = '-E79qAYLfcU'
features = read_features_from_lmdb(lmdb_path, key)

if features is not None:
    # Assuming the structure contains an item named 'features'
    video_features = features['features']
    print (video_features.shape)
    print(video_features)
else:
    print("No data found for the specified key.")