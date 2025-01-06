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

def get_correlation_df_categories(df1, df2):
    merged_df = pd.merge(df1, df2, on=['turn', 'id'], suffixes=('_df1', '_df2'))

    correlation_results = []
    grouped_df = merged_df.groupby('category')

    for category, group in grouped_df:
        # Nur Gruppen mit mindestens 2 Einträgen berücksichtigen
        if len(group) < 2:
            print(f"Skipping category {category} because it has fewer than 2 entries.")
            continue

        for df1_col in df1.columns:
            if df1_col in ['turn', 'id']:
                continue 

            for df2_col in df2.columns:
                if df2_col in ['turn', 'id', 'category']:
                    continue 
                
                try:
                    # Berechnung der Korrelationen nur, wenn mehr als 1 Wert vorhanden ist
                    spearman_corr, spearman_p_value = stats.spearmanr(group[df1_col], group[df2_col])
                    pearson_corr, pearson_p_value = stats.pearsonr(group[df1_col], group[df2_col])

                    correlation_results.append({
                        'category': category,
                        'df1': df1_col,
                        'df2': df2_col,
                        'spearman_corr': spearman_corr,
                        'spearman_p_value': spearman_p_value,
                        'pearson_corr': pearson_corr,
                        'pearson_p_value': pearson_p_value
                    })
                except ValueError as e:
                    print(f"Skipping correlation for {df1_col} and {df2_col} in category {category} due to error: {e}")

    correlation_results_df = pd.DataFrame(correlation_results)
    return correlation_results_df



def categorize_instruction(instruction):
    '''
    categories = {
        "changes": r"\b(make|let|should be|should have|edit|have)\b",
        "removals": r"\b(remove|get rid of)\b",
        "replacements": r"\b(replace|swap|change|switch)\b",
        "adds": r"\b(add|let there be|fill|give)\b",
        "what if": r"\b(what if|can|could|)\b",
        "other": r"\b(take|turn|put|place|cover|close|leave|open|awake|split)\b",
        
    }
    '''
    categories = {
        "make": r"\bmake\b",
        "remove": r"\bremove\b",
        "get rid of": r"\bget rid of\b",
        "let": r"\blet\b",
        "should be": r"\bshould be\b",
        "should have": r"\bshould have\b",
        "edit": r"\bedit\b",
        "have": r"\bhave\b",
        "replace": r"\breplace\b",
        "swap": r"\bswap\b",
        "change": r"\bchange\b",
        "switch": r"\bswitch\b",
        "add": r"\badd\b",
        "let there be": r"\blet there be\b",
        "fill": r"\bfill\b",
        "give": r"\bgive\b",
        "what if": r"\bwhat if\b",
        "can": r"\bcan\b",
        "could": r"\bcould\b",
        "take": r"\btake\b",
        "turn": r"\bturn\b",
        "put": r"\bput\b",
        "place": r"\bplace\b",
        "cover": r"\bcover\b",
        "close": r"\bclose\b",
        "leave": r"\bleave\b",
        "open": r"\bopen\b",
        "awake": r"\bawake\b",
        "split": r"\bsplit\b",
    }

    for category, pattern in categories.items():
        if re.search(pattern, instruction):
            return category
    return "no category"


def main(name, rater): #, original_image, edit_image, instruction):

    rater = pd.read_csv(f"./open_clip_inference/rater{rater}.csv", sep=",")
    pattern = r'(\d+)-output(\d+)'
    with open('./open_clip_inference/edit_turns.json') as f:
        turns = json.load(f)
        
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)
    
    dev = pd.read_csv("/home/jovyan/BA/Github/MagicBrush/dev_data.csv", sep=",")
    
    results = []
    for index, row in rater.iterrows():
        id = int(row["id"])
        turn = int(row["turn"])

        for entry in turns:
            output = entry["output"]
            match = re.search(pattern, output)

            if match:
                found_id = int(match.group(1)) # get id of sample
                found_turn = int(match.group(2)) # get turn of sample

                if int(found_id) == id and int(found_turn) == turn: # check if turn is within rater
                    input = entry["input"]
                    mask = entry["mask"]
                    instruction = entry["instruction"].lower()
                    #category = categorize_instruction(instruction)  # Neue Kategorie hinzufügen
                    #print(f"instruction: {instruction}, category: {category}")
                    

                    source_img = dev[(dev["turn_index"] == found_turn) & (dev["img_id"] == found_id)]["source_img"].iloc[0]
                    target_img = dev[(dev["turn_index"] == found_turn) & (dev["img_id"] == found_id)]["target_img"].iloc[0]   
    
                    image_original = cv2.imread(source_img)
                    image_edit = cv2.imread(target_img)
                    #image_mask = cv2.imread(fr'{path}/{mask}')
                
                    target_size = (512, 512)
                    image_original = cv2.resize(image_original, target_size)
                    image_edit = cv2.resize(image_edit, target_size)
                    #image_mask = cv2.resize(image_mask, target_size)
                    
                    #frames = morph_images(image_original, image_edit, 8)
                    
                    frames = []
                    for i in range(8):
                        alpha = i / (8 - 1)  # Alpha value for blending
                        morphed_image = cv2.addWeighted(image_original, 1 - alpha, image_edit, alpha, 0)
                        frames.append(morphed_image)
                        #output_path = f"frame_{i:03d}.jpg"
                        #cv2.imwrite(output_path, morphed_image)
                        #print(f"Saved frame {i}: {output_path}")
 
                    inputs = processor(text=[instruction], videos=list(frames), return_tensors="pt", padding=True)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    similarity = outputs.logits_per_video.item()
                    print(similarity)

                    row = {
                        "id": id,
                        "turn": turn,
                        "x-clip_score": similarity
                    }
                    results.append(row)
    
    clip_scores = pd.DataFrame(results)
    # clip_scores.to_csv("trained_clip_scores.csv", index=False) # save CLIP's predictions
    #human_scores = pd.read_csv("./open_clip_inference/rater.csv", sep=";")
    
    #human_scores = human_scores.drop_duplicates()
    clip_scores = clip_scores.drop_duplicates()
    
    correlation_df = get_correlation_df_with_columns(rater, clip_scores)
    #correlation_df = get_correlation_df_categories(rater, clip_scores)
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
    parser.add_argument('--rater', type=int, required=True, help='Number of rater: either 1, 2, or 3')

    #parser.add_argument('--image_original_path', type=str, required=True, help='Path to the original image')
    #parser.add_argument('--image_edit_path', type=str, required=True, help='Path to the edited image')
    #parser.add_argument('--instruction', type=str, required=True, help='Path to the instruction')

    args = parser.parse_args()
    main(args.name, args.rater)#, args.image_original_path, args.image_edit_path)#, args.instruction) 