import time
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
torch.set_default_dtype(torch.float16)
torch.backends.cudnn.benchmark = True 
def load_model():
    model_load_path = "temp_extracted_model"  # Path to your saved model
    pipe = StableDiffusionPipeline.from_pretrained(model_load_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe
prompt = " flying cars and neon lights"
start_time = time.time()
with torch.no_grad():
    image = load_model(prompt).images[0]
end_time = time.time()
elapsed_time = end_time - start_time
plt.imshow(image)
plt.axis("off") 
plt.show()
print(f"Time taken to generate the image: {elapsed_time:.2f} seconds")