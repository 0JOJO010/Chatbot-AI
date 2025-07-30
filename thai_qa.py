from transformers import pipeline

pipe = pipeline("question-answering", model="Thammarak/wangchanBERTa-QA-thaiqa_squad")

context = "ประเทศไทยตั้งอยู่ในเอเชียตะวันออกเฉียงใต้ มีประชากรราว 70 ล้านคน"
question = "ประเทศไทยตั้งอยู่ที่ไหน?"

result = pipe(question=question, context=context)

print("คำตอบ:", result['answer'])
