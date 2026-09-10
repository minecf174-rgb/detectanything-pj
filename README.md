# Detect Anything - Computer Vision Training Platform

เว็บสำหรับจัดการรูปภาพ, annotation, dataset versions, YOLO training, model testing และ auto-label review รองรับทั้ง Object Detection และ Pose Estimation

## System requirements

- Git
- Python 3.11 หรือ 3.12
- Node.js 20 ขึ้นไป และ npm
- อินเทอร์เน็ตสำหรับติดตั้ง ML dependencies และดาวน์โหลด pretrained weights

## First-time setup

คำสั่งทั้งหมดในหัวข้อนี้ให้เริ่มจาก root ของโปรเจกต์

### 1. สร้าง Backend environment

```bash
cd "detectanything-pj"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

แพ็กเกจ ML เช่น PyTorch, Ultralytics, SAM และ OpenCV มีขนาดใหญ่ การติดตั้งครั้งแรกอาจใช้เวลาสักพัก

### 2. สร้าง environment config

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

นำข้อความที่คำสั่งสร้างให้ไปแทนค่า `AUTH_JWT_SECRET` ใน `.env`

ตัวอย่าง local config:

```dotenv
DATABASE_URL=sqlite:///./cv_platform.sqlite3
STORAGE_ROOT=storage
TEMP_ROOT=temp
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
AUTH_JWT_SECRET=replace-this-with-the-generated-secret
AUTH_ACCESS_TOKEN_MINUTES=30
AUTH_REFRESH_TOKEN_DAYS=14
AUTH_ALLOW_LEGACY_ACCOUNT_CLAIM=false
```

ห้าม commit `.env` เพราะเป็นค่าลับเฉพาะเครื่อง

### 3. ดาวน์โหลด pretrained models ให้เว็บไซต์ใช้

ดาวน์โหลดโมเดล Pose ที่ใช้ใน `Generate pose suggestions` และหน้า Test model:

```bash
python -c 'from ultralytics import YOLO; YOLO("yolo26n-pose.pt")'
```

ดาวน์โหลด Detect base model เริ่มต้นที่ใช้ตอน Train:

```bash
python -c 'from ultralytics import YOLO; YOLO("yolo26s.pt")'
```

ตรวจว่าไฟล์อยู่ที่ root ของโปรเจกต์:

```bash
ls -lh yolo26n-pose.pt yolo26s.pt
```

ระบบ Backend ตรวจหา pretrained pose จาก path นี้โดยตรง:

```text
detect anything/yolo26n-pose.pt
```

ไม่ต้องเพิ่ม pretrained pose เข้า database และไม่ต้อง import ผ่านหน้าเว็บ เมื่อไฟล์อยู่ถูกตำแหน่ง โปรเจกต์ประเภท Pose Estimation จะใช้ตัวเลือก `YOLO26 Pose pretrained` ได้ทันที

ไฟล์ `.pt` ถูก `.gitignore` ไว้ จึงต้องรันคำสั่งดาวน์โหลดนี้อีกครั้งหลัง clone ลงเครื่องใหม่ ส่วนโมเดลที่ผู้ใช้ Train ผ่านเว็บไซต์จะถูกบันทึกใน `storage/models/` และลงทะเบียนในฐานข้อมูลให้อัตโนมัติหลังการเทรนสำเร็จ

### 4. สร้างฐานข้อมูลและโฟลเดอร์เก็บข้อมูล

```bash
mkdir -p storage/projects storage/models temp/training temp/auto_label
alembic upgrade head
```

คำสั่ง `alembic upgrade head` จะสร้าง `cv_platform.sqlite3` และตารางทั้งหมดตาม migration ล่าสุดโดยอัตโนมัติ ไม่ต้องสร้าง database หรือตารางเอง

ตรวจสอบ migration ปัจจุบัน:

```bash
alembic current
```

เครื่องใหม่ที่ไม่ได้ย้ายฐานข้อมูลเดิมจะเริ่มแบบว่าง ต้อง Create account และสร้างโปรเจกต์ใหม่ในหน้าเว็บ

### 5. ติดตั้ง Frontend

```bash
cd frontend
npm ci
cd ..
```

## Running the application

ใช้ 2 terminal แยกกัน

### Terminal 1: Backend

```bash
cd "detect anything"
source venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Terminal 2: Frontend

```bash
cd "detect anything/frontend"
npm run dev
```

เปิดใช้งาน:

- Website: `http://127.0.0.1:5173`
- API documentation: `http://127.0.0.1:8000/docs`

หลังจากติดตั้งครั้งแรกแล้ว วันต่อไปไม่ต้องติดตั้ง dependencies หรือสร้าง DB ซ้ำ ใช้เพียงคำสั่ง Backend และ Frontend ด้านบน

## Port already in use

ตรวจ process ที่ใช้ Frontend port:

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
kill <PID>
```

ตรวจ process ที่ใช้ Backend port:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

Frontend กำหนด `strictPort: true` จึงรันเฉพาะ port 5173 และจะไม่เปลี่ยนไป 5174 เอง

## Verification

Backend tests:

```bash
cd "detect anything"
source venv/bin/activate
pytest
```

Frontend production build:

```bash
cd "detect anything/frontend"
npm run build
```

## Data and Git

Git เก็บเฉพาะ source code, migrations, tests, documentation และ dependency lock files โดยไม่เก็บ:

- `cv_platform.sqlite3`
- `.env`
- รูปที่อัปโหลดใน `storage/projects/`
- โมเดลที่เทรนใน `storage/models/`
- pretrained `.pt` files
- `temp/`, `runs/`, `venv/`, `node_modules/` และ generated builds

หากต้องการย้ายเฉพาะโค้ด ไม่ต้องสำรองรายการเหล่านี้ เครื่องใหม่ใช้ `alembic upgrade head` เพื่อสร้างฐานข้อมูลเปล่าได้เลย หากต้องการย้ายข้อมูลเดิมในอนาคต ให้อ่าน [คู่มือย้ายเครื่องฉบับเต็ม](docs/new-machine-setup.md)

## Architecture

```text
React + Vite Frontend
  -> FastAPI Backend
  -> Service / Use Case Layer
  -> Repository Layer
  -> SQLAlchemy / SQLite

Backend API
  -> Local Storage Service
  -> Dataset Builder / Ultralytics Trainer / Predictor
  -> Auto-label background task and review workflow
```
