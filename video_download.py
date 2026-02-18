import kagglehub

# Download to your current project folder
path = kagglehub.dataset_download(
    "tuyenldvn/falldataset-imvia",
    output_dir="./datasets/fall_data"
)

print("Dataset is now in:", path)