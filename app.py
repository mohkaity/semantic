import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np

# إعداد الواجهة
st.set_page_config(page_title="البحث الدلالي في نصوص ابن تيمية", layout="wide")

# إدخال مفتاح OpenAI
openai.api_key = st.sidebar.text_input("🔐 أدخل مفتاح OpenAI", type="password")

# تحميل البيانات
@st.cache_resource
def load_data():
    df = pd.read_csv("dar1_with_embeddings.csv")
    index = faiss.read_index("faiss_dar1.index")
    return df, index

df, index = load_data()

# دالة توليد embedding
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# البحث
def search_semantic(query, top_k=3):
    query_vec = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    return df.iloc[indices[0]]

# شرح العلاقة
def explain_match(query, match_text):
    prompt = f"""سؤال المستخدم: "{query}"
النص من كلام ابن تيمية: "{match_text}"

اشرح العلاقة بين النص والسؤال بلغة واضحة وعلمية:"""
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# الواجهة
st.title("📚 البحث الدلالي في نصوص شيخ الإسلام ابن تيمية")

query = st.text_input("📝 أدخل سؤالك", placeholder="مثال: ما موقف ابن تيمية من تقديم العقل على النقل؟")

if query and openai.api_key:
    with st.spinner("🔎 جاري البحث..."):
        results = search_semantic(query, top_k=3)

        for i, row in results.iterrows():
            st.markdown(f"### 🔹 النص {i+1}")
            st.write(row['text'])

            with st.expander("🧠 تفسير الذكاء الاصطناعي"):
                explanation = explain_match(query, row['text'])
                st.write(explanation)
