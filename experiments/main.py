import torch
import torchvision
from PIL import Image
import numpy as np
import open_clip
import argparse
import pandas as pd
import regex as re
import json
from scipy import stats
from skimage.metrics import structural_similarity
from PIL import Image
import cv2

def get_path(human_rating):
    if human_rating==0:
        return "./open_clip_inference/bad_samples"
    elif human_rating==1:
        return "./open_clip_inference/good_samples"

def get_correlation_df_with_columns(df1, df2):
    merged_df = pd.merge(df1, df2, on=['turn', 'id'], suffixes=('_df1', '_df2'))

    correlation_results = []

    for df1_col in df1.columns:
        if df1_col in ['turn', 'id']:
            continue 

        for df2_col in df2.columns:
            if df2_col in ['turn', 'id']:
                continue 
            
            spearman_corr, spearman_p_value = stats.spearmanr(merged_df[df1_col], merged_df[df2_col])
            pearson_corr, pearson_p_value = stats.pearsonr(merged_df[df1_col], merged_df[df2_col])

            correlation_results.append({
                'df1': df1_col,
                'df2': df2_col,
                'spearman_corr': round(spearman_corr,3),
                'spearman_p_value': round(spearman_p_value,4),
                'pearson_corr': round(pearson_corr,3),
                'pearson_p_value': round(pearson_p_value,4)
            })

    correlation_results_df = pd.DataFrame(correlation_results)
    return correlation_results_df


def main(model_type, checkpoint, name): #, original_image, edit_image, instruction):

    samples = pd.read_csv("./open_clip_inference/samples.csv", sep=";")
    pattern = r'(\d+)-output(\d+)'
    with open('./open_clip_inference/edit_turns.json') as f:
        turns = json.load(f)
        
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type, 
        pretrained=checkpoint
    )
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_type)    
    #linear_layer = torch.nn.Linear(1024, 512)
    
    results = []
    for index,row in samples.iterrows():
        id = row["id"] 
        turn = row["turn"]

        path = get_path(row["human_rating_binary"])

        for entry in turns:
            output = entry["output"]
            match = re.search(pattern, output)

            if match:
                found_id = match.group(1) # get id of sample
                found_turn = match.group(2) # get turn of sample

                if int(found_id) == id and int(found_turn) == turn: # check if turn is within samples
                    input = entry["input"]
                    mask = entry["mask"]
                    instruction = entry["instruction"]
    
                    image_original = Image.open(fr'{path}/{input}')
                    image_edit = Image.open(fr'{path}/{output}')

                    image_original = preprocess(image_original).unsqueeze(0)
                    image_edit = preprocess(image_edit).unsqueeze(0)
                    text = tokenizer([instruction]) # multiple instructions can be specified to obtain multiple probabilities

                    with torch.no_grad(), torch.cuda.amp.autocast():
                        """ concat along channels
                        image = torch.concatenate([image_original, image_edit], axis=1)
                        image_features = model.encode_image(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        """

                        image_original_features = model.encode_image(image_original) # ViT outputs image AND tokens!
                        image_edit_features = model.encode_image(image_edit) # same for here
                        text_features = model.encode_text(text)
                        
                        image_original_features /= image_original_features.norm(dim=-1, keepdim=True)
                        image_edit_features /= image_edit_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    image_features = image_original_features + image_edit_features
                    #image_features = torch.concatenate([image_original_features, image_edit_features], axis=1)
                    #image_features = linear_layer(image_features)

                    # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # when specified multiple instructions
                    
                    similarity = (image_features @ text_features.T).item()  
                    print(similarity)
                    
                    row = {
                        "id": id,
                        "turn": turn,
                        "clip_score": similarity
                    }
                    results.append(row)
    
    clip_scores = pd.DataFrame(results)
    # clip_scores.to_csv("trained_clip_scores.csv", index=False) # save CLIP's predictions
    human_scores = pd.read_csv("./open_clip_inference/human_scores.csv")
    
    human_scores = human_scores.drop_duplicates()
    clip_scores = clip_scores.drop_duplicates()
    
    correlation_df = get_correlation_df_with_columns(human_scores, clip_scores)
    correlation_df.to_csv(f"./open_clip_inference/correlation_{name}.csv")
 
    
    """ for a single image
    original_image = preprocess(Image.open(original_image)).unsqueeze(0)
    edit_image = preprocess(Image.open(edit_image)).unsqueeze(0)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_original = model.encode_image(original_image)
        image_edit = model.encode_image(edit_image)
        
        text = tokenizer(["Add a dolphin"])
        text_features = model.encode_text(text)
        
        image_original /= image_original.norm(dim=-1, keepdim=True)
        image_edit /= image_edit.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features = torch.concatenate([image_original, image_edit], axis=1)

    #text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    similarity = (image_features @ text_features.T).item()
    print(similarity)
    """
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a model with specified checkpoint.')
    parser.add_argument('--model', type=str, default='RN101', help='Type of model to load (default: RN101)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--name', type=str, required=True, help='Name of file to save the correlation df')

    #parser.add_argument('--image_original_path', type=str, required=True, help='Path to the original image')
    #parser.add_argument('--image_edit_path', type=str, required=True, help='Path to the edited image')
    #parser.add_argument('--instruction', type=str, required=True, help='Path to the instruction')

    args = parser.parse_args()
    main(args.model, args.checkpoint, args.name)#, args.image_original_path, args.image_edit_path)#, args.instruction) 