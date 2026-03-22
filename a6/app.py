import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# =========================
# 1. LOAD MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model (CPU)
embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
embed_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to("cpu")

# Generation model
gen_tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

gen_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16
).to(device)


# =========================
# 2. FUNCTIONS
# =========================
def get_embedding(text):
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to("cpu")

    with torch.no_grad():
        outputs = embed_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, db, top_k=3):
    q_emb = get_embedding(query)
    sims = []
    for chunk, emb in db:
        sims.append((chunk, cosine_similarity(q_emb, emb)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]



def rag_pipeline(query, db):
    retrieved = retrieve(query, db)

    context = "\n".join([f"- {c[:200]}" for c, _ in retrieved])

    prompt = f"""
You are a QA system.

Rules:
- Answer in ONE short sentence
- Use ONLY the context
- Do NOT explain
- If not found, say: I don't know

Context:
{context}

Question: {query}
Answer:
"""

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():   
        outputs = gen_model.generate(
            inputs.input_ids,
            max_new_tokens=120,   # 🔥 比 max_length 更好
            pad_token_id=gen_tokenizer.eos_token_id
        )

    decoded = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    torch.cuda.empty_cache()

    # 🔥 提取答案
    if "Answer:" in decoded:
        answer = decoded.split("Answer:")[-1].strip()
    else:
        answer = decoded.strip()

    answer = answer.split("\n")[0]   # 🔥 防止乱输出

    return answer, retrieved   # ✅ 返回两个值

# =========================
# 3. LOAD VECTOR DB
# =========================
import pickle

with open("vector_db_context.pkl", "rb") as f:
    VECTOR_DB_CONTEXT = pickle.load(f)


# =========================
# 4. UI
# =========================
st.title("📚 RAG Chatbot (Chapter 10)")

query = st.text_input("Ask a question:")

if query:
    answer, sources = rag_pipeline(query, VECTOR_DB_CONTEXT)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for chunk, score in sources:
        st.write(f"Score: {score:.3f}")
        st.write(chunk[:300])
        st.write("---")