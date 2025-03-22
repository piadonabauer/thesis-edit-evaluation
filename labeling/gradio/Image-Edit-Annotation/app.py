import gradio as gr
import os
from PIL import Image
from pymongo import MongoClient
import requests
from io import BytesIO
from dotenv import load_dotenv

instruction_beginning = """
## 🔍 Evaluation of AI Quality

\n**Background**

\nIn this task, you will evaluate the quality of image edits based on a textual instruction and two input images regarding three aspects: **Instruction-Edit Alignment**, **Visual Quality** and **Consistency**.
Each aspect should be rated on a scale from 1 to 10, where 1 indicates 'very poor' and 10 represents 'excellent'.

Please ensure you have read the detailed instructions provided in this [document](https://www.canva.com/design/DAGP0UTTygI/rYkYZtLUipuKbXPbRcj9kQ/edit?utm_content=DAGP0UTTygI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) before starting the labeling process.

\n**Dataset Source**

This project uses the [MagicBrush dataset](https://osu-nlp-group.github.io/MagicBrush/) (dev split), released under the CC-BY-4.0 license.

\n**Labeling**

\nPlease enter a nickname to avoid repeating image pairs. Make sure to remember this nickname for future sessions. If you choose not to enter a nickname, your ratings will not be saved.
"""

alignment_info = """
How well does the edited area align with the text instruction? (e.g. numbers, colors, and objects)
"""

quality_info = """
How realistic and aesthetically pleasing is the edited area? (e.g. color realism and overall aesthetics)
"""

consistency_info = """
How seamlessly does the edit integrate with the rest of the original image? (e.g. consistency in style, lighting, logic, and spatial coherence)
"""

overall_info = """
How do you perceive and like the edit as a whole, how well does it meet your expectations and complements the original image?
"""

# load_dotenv()

mongo_user = os.getenv('MONGO_USER')
mongo_password = os.getenv('MONGO_PASSWORD')
cluster_url = os.getenv('MONGO_CLUSTER_URL')
gradio_user = os.getenv('GRADIO_USER')
gradio_password = os.getenv('GRADIO_PASSWORD')

connection_url = f"mongodb+srv://{mongo_user}:{mongo_password}@{cluster_url}"
client = MongoClient(connection_url)
db = client["thesis"]
collection = db["labeling"]


def download_image(url):
    """Download image from a given URL."""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def fetch_random_entry(annotator):
    """Fetch a random entry from the database that hasn't been rated by the specified annotator."""
    pipeline = [
        {
            "$match": {
                "ratings.rater": {"$ne": annotator}  # exclude entries where rater is the specified annotator
            }
        },
        {"$sample": {"size": 1}}  # randomly select one entry
    ]

    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def save_rating(entry_id, turn, annotator, alignment, quality, consistency, overall):
    """Save the given ratings into the database."""
    if annotator and entry_id != '' and turn != '':
        rating = {
            "rater": annotator,
            "alignment": alignment,
            "quality": quality,
            "consistency": consistency,
            "overall": overall
        }

        collection.update_one(
            {"meta_information.id": int(entry_id), "meta_information.turn": int(turn)},
            {"$push": {"ratings": rating}}
        )


def count_labeled_images(annotator):
    """Count how many images a person has labeled based on the 'ratings' field."""
    pipeline = [
        {
            "$match": {
                "ratings.rater": annotator  # where 'rater' is the given annotator
            }
        },
        {
            "$count": "labeled_images"  # count the number of documents that match
        }
    ]

    result = list(collection.aggregate(pipeline))

    return result[0]['labeled_images'] if result else 0


def prepare_next_image(annotator):
    """Fetch the next image and its metadata."""
    entry = fetch_random_entry(annotator)

    if not entry:
        return None, None, None, None, "No more images to rate!", None

    meta_info = entry["meta_information"]
    input_image = download_image(meta_info["input_img_link"])
    output_image = download_image(meta_info["output_img_link"])
    instruction = meta_info["instruction"]
    progress_message = f"**Rate this image edit! ({count_labeled_images(annotator)}/528 labeled)**"

    return meta_info["id"], input_image, output_image, instruction, progress_message, meta_info["turn"]


def start(annotator):
    return prepare_next_image(annotator)


def record_input(id, turn, annotator, alignment, quality, consistency, overall):
    save_rating(id, turn, annotator, alignment, quality, consistency, overall)
    img_id, img_block1, img_block2, prompt, progress_text, turn = prepare_next_image(annotator)
    return img_id, img_block1, img_block2, prompt, progress_text, turn, 5, 5, 5, 5


# Gradio Interface
def create_interface():
    with gr.Blocks(theme=gr.themes.Origin()) as demo:
        gr.Markdown(instruction_beginning)

        # annotator = gr.Textbox(label="Nickname", interactive=True)
        annotator = gr.Textbox(label="Annotator Nickname")

        start_btn = gr.Button("Start", variant="primary")
        progress_text = gr.Markdown("Waiting to start.")
        # progress_text = gr.Markdown("You have labeled **0** out of 528 potential images.")

        with gr.Row():
            img_block1 = gr.Image(visible=True, width=300, height=300, label="Original Image", interactive=False)
            img_block2 = gr.Image(visible=True, width=300, height=300, label="Edited Image", interactive=False)

        prompt = gr.Textbox(label="Instruction", visible=True, interactive=False)
        img_id = gr.Textbox(visible=False)
        turn = gr.Textbox(visible=False)

        with gr.Row():
            slider_alignment = gr.Slider(label="Instruction-Edit Alignment", minimum=0, maximum=10, step=1, value=5,
                                         info=alignment_info)
            slider_quality = gr.Slider(label="Visual Quality", minimum=0, maximum=10, step=1, value=5,
                                       info=quality_info)
            slider_consistency = gr.Slider(label="Consistency", minimum=0, maximum=10, step=1, value=5,
                                           info=consistency_info)

        slider_overall = gr.Slider(label="Overall Impression", minimum=0, maximum=10, step=1, value=5,
                                   info=overall_info)

        save_and_continue_btn = gr.Button("Save & Continue", variant="primary")

        start_btn.click(
            fn=start,
            inputs=[annotator],
            outputs=[img_id, img_block1, img_block2, prompt, progress_text, turn]
        )

        save_and_continue_btn.click(
            fn=record_input,
            inputs=[img_id, turn, annotator, slider_alignment, slider_quality, slider_consistency, slider_overall],
            outputs=[img_id, img_block1, img_block2, prompt, progress_text, turn, 
                     slider_alignment, slider_quality, slider_consistency, slider_overall]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    #demo.launch(share=True, debug=True, auth=(gradio_user, gradio_password))
    demo.launch(share=True, debug=True)
