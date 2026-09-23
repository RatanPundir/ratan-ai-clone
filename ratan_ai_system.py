
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from unsloth import FastLanguageModel
import torch

# ====================== PATHS ======================
BASE = "/content/drive/MyDrive/AI personality clone/ratan_final_system"
PROFILE_PATH = os.path.join(BASE, "profile.json")
RAG_DB_PATH = os.path.join(BASE, "ratan_rag_db")
MODEL_PATH = "/content/drive/MyDrive/AI personality clone/ratan_qwen_lora_model"

# ====================== LOAD PROFILE ======================
with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    profile = json.load(f)

PUBLIC_DEMO = True

def direct_answer(question):
    q = question.lower().strip()

    # ---- Strict matching (sirf clear questions pe jawab) ----
    if q in ["tera naam kya hai", "tumhara naam kya hai", "naam kya hai", "your name", "full name"]:
        return f"Mera naam {profile['name']} h bhai."
    
    if q in ["nickname", "nick name", "anuj", "tera nickname"]:
        return f"Mujhe {profile['nickname']} bolte hain bhai."

    if any(w in q for w in ["konsa course", "kaunsa course", "kya padh raha", "kya pdh rha"]):
        return f"{profile['course']} kar raha hu bhai, {profile['college']} me."

    if any(w in q for w in ["konsa college", "kaunsa college", "college ka naam"]):
        return f"{profile['college']} me {profile['course']} kar raha hu bhai."

    if any(w in q for w in ["semester", "kaunse sem", "kaunsa year"]):
        return f"Abhi {profile['year_or_semester']} me hu bhai."

    if any(w in q for w in ["gym kab", "gym time", "gym schedule", "gym jata", "gym jaata"]):
        return f"Gym {profile['gym_schedule']}."

    if any(w in q for w in ["interest", "hobby", "pasand kya"]):
        return f"Mujhe {', '.join(profile['interests'])} me interest h bhai."

    if any(w in q for w in ["goal", "aim", "career goal"]):
        return f"Mera goal {' aur '.join(profile['Goals'])} banna h bhai."

    if any(w in q for w in ["close friend", "best friend", "sabse close"]):
        return f"Mera close friend {profile['Close_Friend']} h bhai."

    if any(w in q for w in ["friends kaun", "dost kaun", "mere friends", "tere friends"]):
        return f"Mere dost hain: {', '.join(profile['Friends'])}."

    # Privacy
    if any(w in q for w in ["email", "gmail", "mail id"]):
        return "Bhai email private h, share nhi kar sakta."
    if any(w in q for w in ["address", "poora address", "ghar ka pata"]):
        return "Bhai exact address private h, share nhi kar sakta."
    if any(w in q for w in ["papa", "father", "sister", "mummy", "family"]):
        return "Bhai family ki private details share nhi karta."

    return None   # baaki sab Model + RAG pe jayega

# ====================== LOAD RAG ======================
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=RAG_DB_PATH, embedding_function=embedding)

def get_rag_context(question, k=3):
    results = vectorstore.similarity_search(question, k=k)
    return "\\n".join([doc.page_content for doc in results])

# ====================== LOAD MODEL ======================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# ====================== FINAL CHAT FUNCTION ======================
def chat_with_ratan(user_question):
    direct = direct_answer(user_question)
    if direct is not None:
        return direct

    rag_context = get_rag_context(user_question)

    system_prompt = """Tum Ratan Kumar Pundir ho (nickname Anuj).
Tumhari language natural Hinglish hai. Short, casual aur real WhatsApp style mein reply dete ho.
Thoda bakchodi aur bhai-style maintain karo.

Neeche diye gaye CONTEXT ko dhyan me rakh ke jawab dena.
Agar context me information nahi hai to guess mat karna. Short mein jawab dena."""

    full_prompt = f"""<|im_start|>system
{system_prompt}

CONTEXT:
{rag_context}
<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

print("✅ Improved Ratan AI System Loaded!")
