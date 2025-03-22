# Image Edit Annotation - Labeling Study

This folder hosts the **Image-Edit-Annotation** labeling study, created with [Gradio](https://www.gradio.app/) and hosted on [Hugging Face Spaces](https://huggingface.co/spaces): [Image-Edi-Annotation](https://huggingface.co/spaces/piadonabauer/Image-Edit-Annotation).

## Project Overview

The study is designed to collect human ratings for edited images. Since Hugging Face Spaces does not allow local data storage, the following setup is used:

- **Images:** Stored in a GitHub repository and retrieved using unique IDs.
- **Instructions:** Included as part of a JSON file.
- **Ratings:** Stored securely in a MongoDB database, with credentials managed safely.

## Hosting on Hugging Face Spaces

Hugging Face Spaces provides free infrastructure for hosting machine learning models and applications. 

For detailed setup instructions, see the official Gradio guide:  
🔗 [Sharing Your App](https://www.gradio.app/guides/sharing-your-app)

### Deploying to Hugging Face Spaces

1. **Create a Hugging Face account**.
2. **Deploye from the terminal** by running:
   ```bash
   gradio deploy
