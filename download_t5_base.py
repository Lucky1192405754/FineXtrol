from transformers import T5Tokenizer, T5EncoderModel

model_name = "google-t5/t5-base"

local_path = "./t5-base-local"

print(f"Downloading model '{model_name}' to '{local_path}'...")

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
tokenizer.save_pretrained(local_path)

model = T5EncoderModel.from_pretrained(model_name)
model.save_pretrained(local_path)

print("Download complete.")
print(f"You can now point your CMDM class to load from '{local_path}'.")