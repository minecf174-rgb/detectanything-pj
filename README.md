# Detect Anything

เว็บจัดการภาพและ annotation สำหรับ Object Detection / Pose Estimation พร้อมสร้าง dataset version, เทรน YOLO, ทดสอบโมเดล และตรวจผล auto-label

## เตรียมเครื่องใหม่

คู่มือนี้เป็นการเริ่มด้วย **ฐานข้อมูลเปล่า** หลัง clone: ผู้ใช้ โปรเจกต์ รูป annotation dataset version และโมเดลที่เคยเทรน **จะไม่ตามมาด้วย** เพราะไฟล์เหล่านี้ไม่อยู่ใน Git

ต้องมี Git, Python 3.11 หรือ 3.12, Node.js 20+ และอินเทอร์เน็ตสำหรับติดตั้งแพ็กเกจ/ดาวน์โหลด pretrained weights คำสั่งด้านล่างใช้ shell แบบ macOS/Linux

ติดตั้ง Node.js บนเครื่องใหม่ก่อน โดยใช้ตัวติดตั้ง Node.js หรือเครื่องมือจัดการเวอร์ชัน Node ที่ใช้อยู่ เมื่อติดตั้ง Node.js ตามปกติจะได้ `npm` มาด้วย ไม่ต้องติดตั้ง npm, React หรือ Vite แยกต่างหาก ถ้าคำสั่ง `node --version` หรือ `npm --version` ใช้ไม่ได้ ให้แก้การติดตั้ง Node.js ก่อนทำขั้นตอน frontend

ตรวจเวอร์ชันก่อน:

```bash
git --version
python3 --version
node --version
npm --version
```

### 1. Clone และเข้าโฟลเดอร์โปรเจกต์

รันจากโฟลเดอร์ที่ต้องการเก็บโปรเจกต์:

```bash
git clone https://github.com/minecf174-rgb/detectanything-pj.git "detect anything"
cd "detect anything"
```

คำสั่งในขั้นตอน 2-4 ต้องรันจาก **root ของโปรเจกต์** (โฟลเดอร์ที่มี `alembic.ini`) เพราะ path ของ SQLite, storage และโมเดลอ้างอิงตำแหน่งนี้

### 2. ติดตั้ง Backend

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

ครั้งแรกอาจใช้เวลานาน: dependencies มี PyTorch, Ultralytics, OpenCV และ SAM โดยบางแพ็กเกจติดตั้งจาก GitHub

### 3. สร้างฐานข้อมูลเปล่า

ขณะที่ยังอยู่ใน virtual environment และ root ของโปรเจกต์:

```bash
alembic upgrade head
alembic current
```

`alembic upgrade head` จะสร้าง `cv_platform.sqlite3` และตารางตาม migration ล่าสุด ไม่ต้องสร้างตารางเอง โฟลเดอร์ `storage/projects`, `storage/models`, `temp/training` และ `temp/auto_label` มี `.gitkeep` อยู่แล้ว ระบบจะเก็บไฟล์งานในโฟลเดอร์เหล่านี้

**ไม่จำเป็นต้องสร้าง `.env` สำหรับการรันบนเครื่องตัวเอง**: backend มีค่า default สำหรับ SQLite, storage, temp, CORS และ local authentication อยู่แล้ว ถ้าต้องการเปลี่ยนค่า ให้คัดลอก `.env.example` เป็น `.env` และเปลี่ยน `AUTH_JWT_SECRET` เป็นค่าสุ่มใหม่ก่อนเปิดให้เครื่องอื่นเข้าใช้:

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

นำผลลัพธ์ไปใส่ใน `AUTH_JWT_SECRET` ของ `.env` ห้าม commit ไฟล์นี้

### 4. ดาวน์โหลด pretrained weights

สำหรับโปรเจกต์ Pose Estimation ให้ดาวน์โหลด `yolo26n-pose.pt` **ที่ root ของโปรเจกต์** ก่อนใช้ `Generate pose suggestions` หรือทดสอบด้วย pretrained pose:

```bash
python -c 'from ultralytics import YOLO; YOLO("yolo26n-pose.pt")'
ls -lh yolo26n-pose.pt
```

Backend อ่าน pretrained pose จากไฟล์ root นี้โดยตรง ไม่ต้องนำเข้า database หรืออัปโหลดผ่านเว็บ

สำหรับการเทรน Object Detection ด้วย base model เริ่มต้น `yolo26s.pt` สามารถดาวน์โหลดไว้ล่วงหน้าได้:

```bash
python -c 'from ultralytics import YOLO; YOLO("yolo26s.pt")'
```

หากเลือก base model ขนาดอื่นตอนเทรน Ultralytics จะต้องดาวน์โหลด weight รุ่นนั้นด้วย ไฟล์ `.pt` ทั้งหมดถูก Git ignore จึงต้องดาวน์โหลดใหม่บนเครื่องที่ clone มา

### 5. ติดตั้ง Frontend

```bash
cd frontend
npm ci
cd ..
```

`npm ci` จะติดตั้งแพ็กเกจ frontend ทั้งหมดจาก `frontend/package-lock.json` ลงใน `frontend/node_modules/` โดยอัตโนมัติ รันครั้งแรกหลัง clone หรือเมื่อ lockfile เปลี่ยน ไม่ต้องสั่ง `npm install` หรือ `npm install -g vite` เพิ่ม

## รันระบบ

เปิด **สอง terminal** โดยเริ่มทั้งสองอันจากโฟลเดอร์แม่ที่ใช้ `git clone` ในขั้นตอน 1

Terminal 1: Backend

```bash
cd "detect anything"
source venv/bin/activate
alembic upgrade head
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2: Frontend

```bash
cd "detect anything/frontend"
npm run dev
```

เปิดเว็บที่ <http://127.0.0.1:5173> และ API docs ที่ <http://127.0.0.1:8000/docs> เครื่องใหม่ให้สร้าง account และ project ใหม่ในเว็บ

ครั้งถัดไปใช้แค่สอง terminal ด้านบน ไม่ต้องติดตั้ง dependencies หรือดาวน์โหลด weights ซ้ำ `alembic upgrade head` รันซ้ำได้และควรรันหลังดึงโค้ดที่มี migration ใหม่

Frontend ใช้ backend ที่ `http://127.0.0.1:8000` โดยค่าเริ่มต้น หากเปลี่ยน URL ให้กำหนด `VITE_API_BASE_URL` ฝั่ง frontend และปรับ `CORS_ORIGINS` ฝั่ง backend ให้ตรงกัน

### ถ้าพอร์ตถูกใช้อยู่

อย่าเปิด server ซ้อน: ตรวจว่ามี backend/frontend เดิมกำลังทำงานหรือไม่ แล้วใช้ของเดิม หรือหยุด process ที่ **ตัวเองตั้งใจจะเปลี่ยน** ด้วย `Ctrl+C` ใน terminal เดิม

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

Vite ตั้ง `strictPort: true` จึงไม่ย้ายจาก 5173 ไป 5174 เอง

## ตรวจว่าติดตั้งสำเร็จ

จาก root โปรเจกต์:

```bash
source venv/bin/activate
pytest
cd frontend
npm run build
```

## สิ่งที่ Git ไม่ย้ายให้

`.gitignore` ไม่เก็บ `.env`, `cv_platform.sqlite3`, `storage/projects/` (ภาพและ dataset), `storage/models/` (โมเดลที่เทรนแล้ว), pretrained `*.pt`, `temp/`, `runs/`, `venv/`, `node_modules/` และ build output ดังนั้น **โมเดลที่เทรนบนเครื่องเก่าจะไม่ปรากฏในเว็บบนเครื่องใหม่** เมื่อเริ่มจาก DB เปล่า ต้องเทรนใหม่ หรือย้ายฐานข้อมูลและไฟล์ storage มาด้วยกันจาก snapshot เดียวกัน

รายละเอียดกรณีย้ายข้อมูลเดิม: [docs/new-machine-setup.md](docs/new-machine-setup.md)
