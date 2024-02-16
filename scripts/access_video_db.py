import lmdb
import numpy as np
import msgpack
import io
from msgpack_numpy import patch as msgpack_numpy_patch
import os
import torch
import pickle

# This is necessary for unpacking numpy arrays serialized with MessagePack
msgpack_numpy_patch()

def read_features_from_lmdb(lmdb_path, video_key):
    # Open the LMDB database
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        # Retrieve the binary data by key

        binary_data = txn.get(video_key.encode('utf-8'))
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

lmdb_path = '/home/kk2720/LLaMA-VID/storage/football_data/video_db/resnet_slowfast_1.5'
# key = '-E79qAYLfcU'
# features = read_features_from_lmdb(lmdb_path, key)


# if features is not None:
#     # Assuming the structure contains an item named 'features'
#     video_features = features['features']
#     print (video_features.shape)
#     print(video_features)
# else:
#     print("No data found for the specified key.")


def print_all_keys_and_data_structures(lmdb_path):
    # Open the LMDB database in read-only mode
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        # Create a cursor to iterate over the database
        cursor = txn.cursor()
        for key, value in cursor:
            # print(f"Key: {key.decode('utf-8')} Val: {value}")
            try:
                # Attempt to deserialize the value
                data = msgpack.unpackb(value, raw=False)
                # print(data["features"].shape)
                # video_info = pickle.load(value)
                # video_info = pickle.load(open(video_file, 'rb'))
                # image = torch.from_numpy(video_info['feats'][:, 1:])
                image = torch.from_numpy(data['features'][:, :])
                print(image.shape)
                # Assuming data is a dictionary, print its keys to represent the "columns"
                # if isinstance(data, dict):
                #     print("Columns/Data Structure:")
                #     # for data_key in data.keys():
                #     #     print(f"  - {data_key}")
                # else:
                #     print("Data is not a dictionary.")
            except Exception as e:
                print(f"Error unpacking data for key {key.decode('utf-8')}: {e}")
            print("----------------------------")



def convert_vids_to_mp4(lmdb_path):
    output_dir = "/home/kk2720/LLaMA-VID/storage/football_data/video_mp4s"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the LMDB database in read-only mode
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        # Create a cursor to iterate over the database
        cursor = txn.cursor()
        i = 0
        for key, value in cursor:
            if i > 10:
                return
            i += 1
            key_str = key.decode('utf-8')
            print(f"Key: {key_str}")
            # Define the output file path
            output_file_path = os.path.join(output_dir, f"{key_str}.mp4")
            try:
                # Write the binary video data to an MP4 file
                with open(output_file_path, 'wb') as file:
                    file.write(value)
                print(f"Video saved to {output_file_path}")
            except Exception as e:
                print(f"Error saving video for key {key_str}: {e}")
            print("----------------------------")


# convert_vids_to_mp4(lmdb_path)
print_all_keys_and_data_structures(lmdb_path)