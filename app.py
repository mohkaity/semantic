import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np

# واجهة المستخدم
st.set_page_config(page_title="البحث الدلالي في نصوص ابن تيمية", layout="wide")
st.title("📚 البحث الدلالي في نصوص شيخ الإسلام ابن تيمية")

# الشريط الجانبي
openai_key = st.sidebar.text_input("🔐 أدخل مفتاح OpenAI", type="password")

model_choice = st.sidebar.selectbox(
    "🔍 اختر نموذج OpenAI",
    options=[
        "gpt-4o",         # أحدث نموذج
        "gpt-4-turbo",    # نسخة محسنة من GPT-4
        "gpt-3.5-turbo"   # النموذج الأسرع والأرخص
    ],
    index=0
)

embedding_model = "text-embedding-ada-002"

# تحميل البيانات
@st.cache_resource
def load_data():
    df = pd.read_csv("dar1_with_embeddings.csv")
    index = faiss.read_index("faiss_dar1.index")
    return df, index

df, index = load_data()

# دالة استخراج embedding
def get_embedding(text):
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(input=[text.replace("\n", " ")], model=embedding_model)
    return response.data[0].embedding

# البحث الدلالي
def search_semantic(query, top_k=5, threshold=0.35):
    query_vec = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, dist in enumerate(distances[0]):
        if dist < threshold:
            match = df.iloc[indices[0][i]]
            results.append((match, dist))

    return results

# تفسير العلاقة
def explain_match(query, match_text):
    prompt = f"""سؤال المستخدم: "{query}"
النص من كلام ابن تيمية: "{match_text}"

اشرح العلاقة بين النص والسؤال بلغة علمية واضحة:"""
    
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# إدخال المستخدم
query = st.text_input("📝 أدخل استفسارك", placeholder="مثال: ما موقف ابن تيمية من تقديم العقل على النقل؟")

if query and openai_key:
    with st.spinner("🔎 جاري البحث..."):
        results = search_semantic(query, top_k=3)

        for i, row in results.iterrows():
            st.markdown(f"### 🔹 النص {i+1}")
            st.write(row['text'])

            with st.expander("🧠 تفسير العلاقة"):
                explanation = explain_match(query, row['text'])
                st.write(explanation)
