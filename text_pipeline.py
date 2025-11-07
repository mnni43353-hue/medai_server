# text_pipeline.py
from transformers import MarianTokenizer, MarianMTModel
import torch

# تحميل نموذج الترجمة EN→AR
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar").to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def translate_en_to_ar(text: str):
    batch = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
    gen = model.generate(**batch, num_beams=4)
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return out
