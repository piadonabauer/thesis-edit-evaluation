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
from transformers import AutoModel, XCLIPProcessor, XCLIPModel, XCLIPVisionConfig
import os

np.random.seed(0)


def retrieve_text(frames, 
                  texts, 
                  models={'viclip':None, 
                          'tokenizer':None},
                  topk=5, 
                  device=torch.device('cuda')):
    # clip, tokenizer = get_clip(name, model_cfg['size'], model_cfg['pretrained'], model_cfg['reload'])
    assert(type(models)==dict and models['viclip'] is not None and models['tokenizer'] is not None)
    clip, tokenizer = models['viclip'], models['tokenizer']
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    #probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    similarity = (vid_feat @ text_feats_tensor.T).item()  
    return similarity

    #ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    #return ret_texts, probs.numpy()[0]

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
                'spearman_corr': spearman_corr,
                'spearman_p_value': spearman_p_value,
                'pearson_corr': pearson_corr,
                'pearson_p_value': pearson_p_value
            })

    correlation_results_df = pd.DataFrame(correlation_results)
    return correlation_results_df




def main(name): #, original_image, edit_image, instruction):

    samples = pd.read_csv("./open_clip_inference/samples.csv", sep=";")
    pattern = r'(\d+)-output(\d+)'
    with open('./open_clip_inference/edit_turns.json') as f:
        turns = json.load(f)
        
    model_name = "microsoft/xclip-base-patch32"
    #configuration = XCLIPVisionConfig(num_frames=2)
    #vision_model = XCLIPVisionModel(configuration)
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    
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
    
                    image_original = cv2.imread(fr'{path}/{input}')
                    image_edit = cv2.imread(fr'{path}/{output}')
                    image_mask = cv2.imread(fr'{path}/{mask}')
                
                    target_size = (512, 512)
                    image_original = cv2.resize(image_original, target_size)
                    image_edit = cv2.resize(image_edit, target_size)
                    image_mask = cv2.resize(image_mask, target_size)
                
                    """
                    clip_len = 8
                    images = np.stack([image_original, image_edit])
                    num_repeats = clip_len // 2
                    video = np.tile(images, (num_repeats, 1, 1, 1))
                    video = video[:clip_len]
                    """
                    
                    frames = []
                    for i in range(8):
                        alpha = i / (8 - 1)  # Alpha value for blending
                        morphed_image = cv2.addWeighted(image_original, 1 - alpha, image_edit, alpha, 0)
                        frames.append(morphed_image)
                    
                    inputs = processor(text=[instruction], videos=list(frames), return_tensors="pt", padding=True)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    similarity = outputs.logits_per_video.item()#.softmax(dim=1)
                    #similarity = similarity[0][0].item()
                    print(similarity)

                    row = {
                        "id": id,
                        "turn": turn,
                        "viclip_score": similarity
                    }
                    results.append(row)
    
    clip_scores = pd.DataFrame(results)
    # clip_scores.to_csv("trained_clip_scores.csv", index=False) # save CLIP's predictions
    human_scores = pd.read_csv("./open_clip_inference/human_scores.csv", sep=";")
    
    human_scores = human_scores.drop_duplicates()
    clip_scores = clip_scores.drop_duplicates()
    
    correlation_df = get_correlation_df_with_columns(human_scores, clip_scores)
    correlation_df.to_csv(f"./open_clip_inference/correlation_{name}.csv", sep=",")
 
    
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
    #parser.add_argument('--model', type=str, default='RN101', help='Type of model to load (default: RN101)')
    #parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--name', type=str, required=True, help='Name of file to save the correlation df')

    #parser.add_argument('--image_original_path', type=str, required=True, help='Path to the original image')
    #parser.add_argument('--image_edit_path', type=str, required=True, help='Path to the edited image')
    #parser.add_argument('--instruction', type=str, required=True, help='Path to the instruction')

    args = parser.parse_args()
    main(args.name)#, args.image_original_path, args.image_edit_path)#, args.instruction) 