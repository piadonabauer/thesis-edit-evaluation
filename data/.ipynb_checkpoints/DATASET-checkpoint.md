# Datasets for ViFi-CLIP and OpenCLIP

This repository contains the necessary datasets and preprocessing steps required for ViFi-CLIP and OpenCLIP fine-tuning.

## Required Files

For ViFi-CLIP, the following resources are required:

- **videos/**
- **videos_8_frames/** (optional, based on your setup)
- **train.txt**
- **test.txt**
- **labels.csv**

For ViFi-CLIP, the following resources are required:

- **images/**
- **train_data.csv**
- **dev_data.csv**

---

## Step 1: Download the Datasets

### HumanEdit Dataset

1. Navigate to the `data` directory:

    ```bash
    cd data
    ```

2. Clone the **HumanEdit** dataset:

    ```bash
    cd data/humanedit
    git clone https://huggingface.co/datasets/BryanW/HumanEdit
    ```

3. Execute the `preprocessing-general.ipynb` notebook to preprocess the data.

4. Once done, you can remove the raw dataset folder:

    ```bash
    rm -rf HumanEdit
    ```

### MagicBrush Dataset

1. Navigate to the `data` directory:

    ```bash
    cd data
    ```

2. Clone the **MagicBrush** dataset:

    ```bash
    cd data/magicbrush
    git clone https://huggingface.co/datasets/osunlp/MagicBrush
    ```

3. Execute the `preprocessing-general.ipynb` notebook for preprocessing.

4. Once done, remove the dataset folder:

    ```bash
    rm -rf MagicBrush
    ```


---

## Required Format for ViFi-CLIP and OpenCLIP

### ViFi-CLIP

- **labels.csv**: Contains columns `id` (int) and `name` (unique instruction, string)
- **train.txt**: Contains video name (string) and label id (int) for the training split
- **test.txt**: Analog to train.txt, but for the test split
- **videos/**: A folder containing the video files (.mp4, 2-frames)

### OpenCLIP

- **train_data.csv**
- **val_data.csv**
- **images/**: A folder containing the image files (.png)

Each CSV file should contain the following columns:
- **source_img**: The link to the original image (string)
- **target_img**: The link to the edited image (string)
- **instruction**: The text instruction for the image (string)

Ideally additional columns (optional):
- **img_id**: An identifier for the image (int)
- **turn_index**: The conversation turn (int)

---

After setting up the datasets and preprocessing them, all the required files are in the correct format for fine-tuning with ViFi-CLIP or OpenCLIP.