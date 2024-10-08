#Stable Diffusion Image Generator

## Project Overview
This project is a **Stable Diffusion Image Generator** built with **Streamlit**. It allows users to generate images from text prompts using a pre-trained Stable Diffusion model. The app provides an intuitive interface where users can input a prompt, choose the desired image size, and generate an image on the fly.

## Features
- **Text-based image generation** using Stable Diffusion.
- **Customizable image size** options (256x256, 512x512, 1024x1024).
- **User-friendly interface** with a sidebar for prompt input and settings.
- **Real-time image generation** with display of the elapsed time.
- **Custom background** and themed UI to enhance user experience.

## Prerequisites
- Python 3.x
- CUDA-compatible GPU for acceleration (recommended)
- Required libraries:
  - `torch`
  - `diffusers`
  - `streamlit`
  - `PIL` (Pillow)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stable-diffusion-image-generator.git
   cd stable-diffusion-image-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install torch diffusers streamlit Pillow
   ```

3. Ensure you have a CUDA-compatible environment for running the model with GPU acceleration.

## Running the App
1. Download the pre-trained **Stable Diffusion v1-4** model and extract it to the `temp_extracted_model` folder (or your preferred location).

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open the app in your browser at `http://localhost:8501/`.

## How to Use
1. **Enter a text prompt**: Describe the image you want to generate (e.g., "A futuristic cityscape at sunset").
2. **Select image size**: Choose from 256x256, 512x512, or 1024x1024.
3. **Click the "Generate Image" button** to create the image.
4. **View the generated image** and the time it took to generate.

## Customization
### Background Image
- You can change the background by replacing `download.jpeg` with another image in the `add_bg_from_local()` function.

### Model Path
- Modify the `model_load_path` to point to the directory where the Stable Diffusion model is stored.

## Notes
- This app uses a **mixed-precision** approach (`torch.float16`) for faster computation with GPUs.
- Make sure your system supports **CUDA** to leverage the full potential of this app.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Built using **Stable Diffusion v1-4**.
- Thanks to **Streamlit** for providing a simple interface for deploying machine learning apps.

---

