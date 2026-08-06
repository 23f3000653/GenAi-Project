import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# CONFIG — update HF_REPO_ID after you upload your trained weights
# (see the deployment guide for how to create this Hugging Face repo)
# ----------------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-small"
HF_REPO_ID = "dev-1234/deberta-mcq-solver"   # <-- CHANGE THIS
WEIGHTS_FILENAME = "deberta_mcq_model.pt"
MAX_LEN = 64

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")

# ----------------------------------------------------------------------
# Model architecture — must match EXACTLY what was used during training
# (copied from the notebook's DeBERTaMCQModel class)
# ----------------------------------------------------------------------
class DeBERTaMCQModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, hidden_dropout=0.3):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = self.deberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        batch_size, num_options, seq_len = input_ids.shape
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        scores = self.classifier(cls_output)
        return scores.view(batch_size, num_options)


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """Downloads your fine-tuned weights + tokenizer from Hugging Face Hub
    and loads them once (cached across reruns/users)."""
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILENAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)

    model = DeBERTaMCQModel()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, prompt, options, max_len=MAX_LEN):
    input_ids_list, attn_mask_list = [], []
    for opt in options:
        encoded = tokenizer(
            prompt, opt,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids_list.append(encoded["input_ids"].squeeze(0))
        attn_mask_list.append(encoded["attention_mask"].squeeze(0))

    input_ids = torch.stack(input_ids_list).unsqueeze(0)        # (1, 5, max_len)
    attention_mask = torch.stack(attn_mask_list).unsqueeze(0)   # (1, 5, max_len)

    with torch.no_grad():
        scores = model(input_ids, attention_mask)
        probs = torch.softmax(scores, dim=1).squeeze(0).numpy()

    letters = ["A", "B", "C", "D", "E"]
    ranked = sorted(zip(letters, probs), key=lambda x: x[1], reverse=True)
    return ranked


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🧠 Smart MCQ Solver")
st.caption("Fine-tuned DeBERTa-v3-small model — enter a question and its 5 options to get a prediction.")

with st.spinner("Model load ho raha hai... (pehli baar 30-60 sec lag sakte hain)"):
    model, tokenizer = load_model_and_tokenizer()

prompt = st.text_area("Question / Prompt", height=100, placeholder="Apna question yahan likhein...")

col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A")
    opt_b = st.text_input("Option B")
    opt_c = st.text_input("Option C")
with col2:
    opt_d = st.text_input("Option D")
    opt_e = st.text_input("Option E")

if st.button("Predict Answer", type="primary", use_container_width=True):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    if not prompt.strip() or any(not o.strip() for o in options):
        st.warning("Please question aur sabhi 5 options fill karein.")
    else:
        with st.spinner("Predicting..."):
            ranked = predict(model, tokenizer, prompt, options)

        best_letter, best_prob = ranked[0]
        st.success(f"### Predicted Answer: **{best_letter}**  \nConfidence: {best_prob:.1%}")

        st.subheader("Top-3 Ranking (MAP@3 style)")
        for letter, prob in ranked[:3]:
            st.write(f"**{letter}** — {prob:.1%}")
            st.progress(float(prob))
