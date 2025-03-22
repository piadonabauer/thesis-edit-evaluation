# Image Edit Annotation - Labeling Study

This folder contains the **Image-Edit-Annotation** labeling study, created with [Gradio](https://www.gradio.app/) and hosted on [HuggingFace Spaces](https://huggingface.co/spaces): [Image-Edi-Annotation](https://huggingface.co/spaces/piadonabauer/Image-Edit-Annotation).

## Overview

The study is designed to collect human ratings for edited images. Since HuggingFace Spaces does not allow local data storage, the following setup is used:

- **Images:** Stored in a GitHub repository and retrieved using unique IDs.
- **Instructions:** Included as part of a JSON file.
- **Ratings:** Stored securely in a MongoDB database.

The database credentials (**user, password, cluster URL**) and the **Gradio authentication details** (**user and password**) must be stored in a `.env` file. This ensures they can be safely retrieved without being hardcoded in the project.

The .env file should follow this format:

  ```bash
MONGO_USER=x
MONGO_PASSWORD=x
MONGO_CLUSTER_URL=x
GRADIO_USER=x
GRADIO_PASSWORD=x
```

## Hosting on HuggingFace Spaces

HuggingFace Spaces provides infrastructure for hosting the labeling application. 
For detailed setup instructions, see the official Gradio guide:  
🔗 [Sharing Your App](https://www.gradio.app/guides/sharing-your-app)

### Deploying to Hugging Face Spaces

1. **Create a Hugging Face account**.
2. **Deploye from the terminal** by running:
   ```bash
   gradio deploy
