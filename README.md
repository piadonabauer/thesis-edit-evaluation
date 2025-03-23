# Learning Temporal Relations for Instruction-Guided Image Editing

This repository contains the implementation of the thesis **"Learning Temporal Relations for Instruction-Guided Image Editing"**, a framework that adapts spatio-temporal modeling techniques from video understanding to evaluate instruction-guided image edits.

## Overview

<img src="https://github.com/piadonabauer/thesis-edit-evaluation/blob/main/data/examples/thesis_images/overview.png?raw=true" alt="Pipeline" width="500"/>

### **Abstract**
*Image editing tools are becoming increasingly popular for creative and professional workflows. While users can manually edit images, AI-based tools now allow text-based instructions to guide image edits, often generating multiple variations for users to select from. However, evaluating instruction-guided image editing remains a challenge due to the lack of established frameworks. Traditional automated metrics focus on overall image quality rather than execution fidelity, often failing to align with human judgment. This thesis proposes a novel evaluation approach that leverages spatio-temporal relationships from video understanding to analyze changes between edited image pairs and their corresponding instructions. A key component is a human annotation study, which aligns the evaluation metric with human perception. While experimental results indicate moderate retrieval performance, the method struggles with fine-grained distinctions. Correlation with human ratings is low overall but improves significantly for well-defined edits, specific editing types, and high-quality training data. These findings highlight the potential of this approach in structured scenarios while emphasizing its limitations as a universal evaluation metric. This research aims to lay the groundwork for a systematic, human-centered evaluation framework for image editing, fostering further discussion and innovation in the field.*

---

## **Main Contributions**
- A new approach for assessing instruction-guided image editing, bridging the gap between automated metrics and human perception.
- An extensive annotation study was conducted to analyze the correlation between the proposed metric and human preferences.
- A detailed analysis of the fine-tuning process and annotation study highlights key challenges, strengths, and limitations.

---

## **Installation**

### **Environment Setup**
Most analyses are performed in Jupyter notebooks, with required libraries installed inside them. To avoid dependency conflicts, two separate virtual environments were used:
1. **For ViFi-CLIP fine-tuning**  
2. **For data analysis**

#### **(1) Setup for Fine-tuning ViFi-CLIP**
Follow the official ViFi-CLIP installation guide:  
[Installation ViFi-CLIP](https://github.com/piadonabauer/thesis-edit-evaluation/blob/main/ViFi-CLIP/docs/INSTALL.md).  

The guide describes executing the following commands:

    ```bash
cd ViFi-CLIP
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Also, install Apex:

    ```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### **(2) Setup for Data Analysis**

    ```bash
cd labeling
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## **Data Preparation**

### **API Credentials**
Three APIs are accessed in this repository:

- OpenAI (for GPT-as-a-judge)
- Gradio (for the demo interface)
- MongoDB (for annotation storage)

Credentials need to be stored in a .env file for secure access. Create a .env file in the top directory with the following content:

    ```bash
MONGO_USER`x
MONGO_PASSWORD=x
MONGO_CLUSTER_URL=x

GRADIO_USER=x
GRADIO_PASSWORD=x

OPENAI_API_KEY=x
```


### **Datasets**

Dataset preparation instructions are provided in a separate [README](https://github.com/piadonabauer/thesis-edit-evaluation/blob/main/data/DATASET.md). The necessary files are included in the repository, except for the images, which must be retrieved and processed into video format.


## **ViFi-CLIP**
The vision-language model used for fine-tuning is **ViFi-CLIP**, a video-specific adaptation of CLIP.

### Model Checkpoints
Due to repository size limitations, model checkpoints (baseline models, ablated models, cross-validation models) are not included here.
The baseline model checkpoints (ViT-B/16, 2 frames, fine-tuned on HumanEdit) are uploaded to [Google Drive](https://drive.google.com/drive/folders/1UP84WRTnSQnixj5BcukE14njjrwPQIVh?usp=sharing). These include for every fold:

- Model configurations (.yaml)
- Logs (.txt)
- Evaluation metric files (.json)
- The checkpoint (.pth)

Other models can be reproduced using the fine-tuning instructions below.

### **Fine-tuning Process**
To reproduce a model (e.g., ViT-B/16, 2 frames, trained on HumanEdit, fold 1):

1. Navigate to the corresponding folder:
`thesis-edit-evaluation/ViFi-CLIP/output/crossvalidation/vitb16_2_humanedit_freeze_none/fold`

2. Adjust paths in the .yaml configuration file.

3. Run the training command:

    ```bash
cd ViFi-CLIP
python -m torch.distributed.run --nproc_per_node=1 main.py \
-cfg output/crossvalidation/vitb16_2_humanedit_freeze_none/fold1/16_32_vifi_clip_all_shot.yaml \
--output output/crossvalidation/vitb16_2_humanedit_freeze_none/fold1
```

For more details, refer to the official ViFi-CLIP [Training Guide](https://github.com/piadonabauer/thesis-edit-evaluation/blob/main/ViFi-CLIP/docs/TRAIN.md).


## **Repository Structure**

### `/data`
Contains resources related to the HumanEdit and MagicBrush datasets, including:
- Instructions for data preparation
- Notebooks for data visualization
- Labels for fine-tuning (Note: images are not included)


### `/experiment`
Contains additional analyses unrelated to ViFi-CLIP, including:
- MagicBrush rating analysis (`/experiment/viescor`)
- Automated metric experiments (`metrics.ipyn`)
- GPT-based quality predictions (`gpt.ipynb`)
- Correlation analysis (`correlation.ipynb`)


### `/labeling/`
Contains all labeling-related components, divided into:

#### Gradio Interface (`/labeling/gradio/`)
- [Hugging Face Demo](https://huggingface.co/spaces/piadonabauer/Image-Edit-Annotation) ![Demo](https://img.shields.io/badge/demo-live-blue)
- MongoDB setup (`setup_db.ipynb`)
- Data retrieval (`data_retrieval_db.ipynb`)

#### Analysis of Labeling Study (`/labeling/analysis/`)
- Label preprocessing (`label_preprocessing.ipynb`)
- GPT-based classification (`prompt_classification_gpt.ipynb`)
- Human vs. ViFi-CLIP correlation (`correlation_vifi_human.ipynb`)
- Annotation files (`/labeling/analysis/annotations/`)
- Cross-validation results (`/labeling/analysis/crossvalidation/`)

### `/ViFi-CLIP/`
Used for:
- Fine-tuning (see instructions above)
- Inference (`inference.ipynb`)


    
## Acknowledgements

The code for model fine-tuning is based on [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP)'s repository. 
