from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from dotenv import load_dotenv
from pythainlp.tokenize import word_tokenize
from datetime import datetime
import os

# โหลดตัวแปรจาก .env
load_dotenv()

# กำหนดโฟลเดอร์เทมเพลต
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# CORS (ถ้ามี frontend แยก)
CORS(app, resources={r"/chat": {"origins": "*"}})

# โหลดโมเดล NLP
embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
kw_model = KeyBERT(model=embedding_model)
print("✅ โหลดโมเดลเรียบร้อย")

# เชื่อม Google Sheets
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("IT-Department-Bot").sheet1
    print("✅ เชื่อม Google Sheets สำเร็จ")
except Exception as e:
    print("❌ ไม่สามารถเชื่อม Google Sheets ได้:", e)
    sheet = None

# จำลอง user
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "user123", "role": "user"}
}

# Logs
login_log = []        # เก็บ log การ login
message_log = []      # เก็บคำถามที่พิมพ์ในระบบ

# =================== NLP Matching ===================

def embed_text(text):
    tokens = word_tokenize(text, keep_whitespace=False)
    tokenized_text = " ".join(tokens)
    keywords = kw_model.extract_keywords(tokenized_text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=5)
    key_texts = [kw[0] for kw in keywords]
    combined_text = " ".join(key_texts)
    embedding = embedding_model.encode([combined_text])[0]
    return embedding

def find_best_match(user_input, rows, threshold=0.4):
    user_embedding = embed_text(user_input)
    best_score = 0
    best_row = None

    for row in rows:
        sheet_keywords = row.get("คีย์เวิร์ดที่เกี่ยวข้อง", "")
        sheet_title = row.get("หัวข้อ", "")
        combined_text = f"{sheet_title} {sheet_keywords}".strip()
        if not combined_text:
            continue

        sheet_embedding = embed_text(combined_text)
        score = cosine_similarity([user_embedding], [sheet_embedding])[0][0]
        if score > best_score and score >= threshold:
            best_score = score
            best_row = row
    return best_row, best_score

# =================== ROUTES ===================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = users.get(username)

        if user and user["password"] == password:
            session["username"] = username
            session["role"] = user["role"]
            login_log.append({"username": username, "time": str(datetime.now())})
            return redirect(url_for("chat_page"))
        else:
            return render_template("login.html", error="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    return render_template("login.html", error=None)

@app.route("/chat")
def chat_page():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("chatbot.html",
        username=session["username"],
        role=session["role"]
    )

@app.route("/chat", methods=["POST"])
def chat_api():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if sheet is None:
        return jsonify({"error": "ไม่สามารถเชื่อมต่อกับฐานข้อมูล Google Sheets ได้"}), 500

    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "ไม่มีข้อความส่งมา"}), 400

        rows = sheet.get_all_records()
        matched, score = find_best_match(user_input, rows)

        # ✅ สร้าง topic จากผลลัพธ์ที่ match ได้ (หรือข้อความ default)
        topic_result = matched.get("หัวข้อ", "ไม่พบหัวข้อที่เกี่ยวข้อง") if matched else "ไม่พบหัวข้อที่เกี่ยวข้อง"
        response_result = matched.get("คำตอบที่ใช้ตอบกลับ", "ขออภัย ฉันยังไม่เข้าใจคำถามนี้ ลองพิมพ์ใหม่อีกครั้งครับ") if matched else "ขออภัย ฉันยังไม่เข้าใจคำถามนี้ ลองพิมพ์ใหม่อีกครั้งครับ"

        # ✅ Log ทั้งคำถามและหัวข้อ
        message_log.append({
            "username": session.get("username"),
            "message": user_input,
            "topic": topic_result,
            "time": str(datetime.now())
        })

        return jsonify({
            "topic": topic_result,
            "response": response_result
        })

    except Exception as e:
        print("❌ ERROR ใน chat_api:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/admin")
def admin_page():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    return render_template("admin.html", logs=login_log, messages=message_log)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/faq")
def faq_page():
    if "username" not in session:
        return redirect(url_for("login"))

    if sheet is None:
        return render_template("faq.html", faqs=[])

    try:
        rows = sheet.get_all_records()
        return render_template("faq.html", faqs=rows)
    except Exception as e:
        print("❌ ERROR ใน /faq:", e)
        return render_template("faq.html", faqs=[])
    
@app.route("/stats")
def stats_page():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    # นับ frequency
    freq = {}
    for m in message_log:
        t = m.get("topic")
        freq[t] = freq.get(t, 0) + 1
    # เรียงและเลือก top10
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    return render_template("stats.html", top_topics=top)



# =================== RUN ===================
if __name__ == "__main__":
    app.run(debug=True)
