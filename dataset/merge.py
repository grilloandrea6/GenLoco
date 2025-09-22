import numpy as np
import glob

def merge_datasets(file_list, output_file="merged_dataset.npz"):
    obs_list = []
    act_list = []

    for f in file_list:
        data = np.load(f)
        obs_list.append(data["obs"])
        act_list.append(data["act"])
        print(f"Loaded {f}: obs {data['obs'].shape}, act {data['act'].shape}")

    # Concatenate along axis 0 (samples)
    merged_obs = np.concatenate(obs_list, axis=0)
    merged_act = np.concatenate(act_list, axis=0)

    print("\n=== Merged Dataset ===")
    print(f"obs: {merged_obs.shape}, act: {merged_act.shape}")

    np.savez(output_file, obs=merged_obs, act=merged_act)
    print(f"Saved merged dataset to {output_file}")

if __name__ == "__main__":
    # Example: load all .npz files in the current folder
    files = sorted(glob.glob("*dataset*.npz"))  # adjust pattern if needed
    merge_datasets(files, "merged_data.npz")
