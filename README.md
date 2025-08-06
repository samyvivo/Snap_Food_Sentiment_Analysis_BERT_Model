# Snap_Food_Sentiment_Analysis_BERT_Model

📌 معرفی پروژه
این پروژه یک مدل تحلیل احساسات فارسی بر اساس مدل ترنسفورمر BERT برای بررسی نظرات کاربران اسنپ‌فود است.
هدف این پروژه تشخیص احساسات مثبت یا منفی در نظرات مربوط به رستوران‌ها و سفارش‌های غذایی است.

🛠 قابلیت‌ها
• پیش‌پردازش متن فارسی (حذف کاراکترهای غیر فارسی، حذف فاصله‌های اضافی، نرمال‌سازی)

• توکنایز کردن متن با AutoTokenizer از Hugging Face

• مدل BERT فارسی برای طبقه‌بندی احساسات

• ارزیابی مدل با معیارهای Accuracy، Precision، Recall، F1-Score

• رابط وب Streamlit برای پیش‌بینی احساسات به صورت زنده

📂 ساختار پروژه
📁 Snap_Food_Sentiment_Analysis_BERT_Model_fa
│── 📄 app.py                # اپلیکیشن Streamlit
│── 📄 model_loader.py       # بارگذاری مدل و پیش‌پردازش
│── 📄 requirements.txt      # لیست کتابخانه‌ها
│── 📄 README.md             # مستندات پروژه
│── 📁 model/                # مدل BERT آموزش دیده
│── 📁 data/                 # دیتاست نظرات اسنپ‌فود


🚀 اجرای پروژه به صورت محلی

git clone https://github.com/samyvivo/Snap_Food_Sentiment_Analysis_BERT_Model_fa.git
cd Snap_Food_Sentiment_Analysis_BERT_Model_fa


2️⃣ نصب پیش‌نیازها

pip install -r requirements.txt

3️⃣ اجرای اپلیکیشن
streamlit run app.py


📊 نمونه پیش‌بینی
نظر (فارسی)	احساس پیش‌بینی‌شده
کیفیت پیتزا عالی بود و خیلی سریع رسید.	مثبت ✅
غذا دیر رسید و خیلی سرد بود.	منفی ❌

🖼 اسکرین‌شات
 <img width="1800" height="836" alt="image" src="https://github.com/user-attachments/assets/964d4ed4-8727-4cc1-91b9-e3c448275290" />

 📜 License
This project is licensed under the MIT License.


