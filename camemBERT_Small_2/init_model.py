from transformers import CamembertConfig, CamembertForMaskedLM
import os

os.makedirs("model_init", exist_ok=True)

config = CamembertConfig.from_pretrained("config")
model = CamembertForMaskedLM(config)

model.save_pretrained("model_init")
