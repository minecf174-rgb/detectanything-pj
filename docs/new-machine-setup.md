# ตั้งค่าโปรเจกต์บนเครื่องใหม่

คู่มือนี้ใช้สำหรับย้ายเฉพาะ source code ไปเครื่องใหม่ โดยไม่ย้ายฐานข้อมูลเดิม รูปที่อัปโหลด ผลการเทรน หรือไฟล์โมเดลที่สร้างในเครื่องเดิม

## สิ่งที่ Git จะเก็บ

- Backend ใน `app/`
- Frontend ใน `frontend/src/`
- Alembic migrations ใน `alembic/`
- Tests และเอกสาร
- `pyproject.toml`, `package.json` และ `package-lock.json`
- `.env.example` สำหรับใช้เป็นแม่แบบ

## สิ่งที่ Git จะไม่เก็บ

- ฐานข้อมูล `cv_platform.sqlite3` และไฟล์ SQLite อื่น
- รูปและ dataset ที่อัปโหลดใน `storage/projects/`
- โมเดลที่เทรนแล้วใน `storage/models/`
- ผลลัพธ์ใน `temp/` และ `runs/`
- pretrained weights เช่น `yolo26n-pose.pt` และ `yolo26x-pose.pt`
- `venv`, `node_modules`, frontend build และ cache
- `.env` ซึ่งมี secret ของแต่ละเครื่อง
- ไฟล์ backup ภายใน `backups/`

ดังนั้นเครื่องใหม่จะเริ่มจากฐานข้อมูลเปล่า ผู้ใช้ โปรเจกต์ รูป annotation dataset version และโมเดลเดิมจะไม่ตามไปด้วย

## โปรแกรมที่ต้องติดตั้ง

- Git
- Python 3.11 หรือ 3.12
- Node.js 20 ขึ้นไป และ npm
- อินเทอร์เน็ตสำหรับติดตั้ง dependencies และดาวน์โหลด YOLO weights

ตรวจสอบเวอร์ชัน:

```bash
git --version
python3 --version
node --version
npm --version
```

## 1. Clone โปรเจกต์

```bash
git clone <repository-url>
cd "detect anything"
```

ถ้าใช้ `setup-new-machine.sh` สคริปต์จะสร้าง root `.env` พร้อม JWT secret แบบสุ่มให้เมื่อยังไม่มีไฟล์ และจะรักษา `.env` เดิมไว้ หากติดตั้งด้วยมือ สำหรับ local development สามารถข้ามการสร้าง `.env` ได้ เพราะ Backend มีค่า default สำหรับ SQLite, storage, temp และ CORS อยู่แล้ว

หากต้องการเปลี่ยน config หรือเตรียม deploy ให้สร้าง `.env`:

```bash
cp .env.example .env
```

จากนั้นสร้าง JWT secret ใหม่สำหรับเครื่องนั้น:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
```

นำค่าที่ได้ไปแทน `AUTH_JWT_SECRET` ใน `.env` ห้าม commit `.env` ขึ้น Git

ค่า local development ที่สำคัญ:

```dotenv
DATABASE_URL=sqlite:///./cv_platform.sqlite3
STORAGE_ROOT=storage
TEMP_ROOT=temp
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

## 2. ติดตั้ง Backend

รันจาก root ของโปรเจกต์:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

การติดตั้ง ML dependencies อาจใช้เวลานาน เพราะมี PyTorch, Ultralytics, SAM และ OpenCV

ดาวน์โหลด pretrained pose weights: หน้า Auto label ใช้รุ่น Extra Large (`x`) ส่วนหน้า Test pretrained และการเทรนค่าเริ่มต้นใช้ Nano (`n`):

```bash
python -c 'from ultralytics import YOLO; YOLO("yolo26x-pose.pt"); YOLO("yolo26n-pose.pt")'
```

คำสั่งนี้ต้องสร้าง `yolo26x-pose.pt` และ `yolo26n-pose.pt` ไว้ที่ root ของโปรเจกต์ ไฟล์ทั้งสองถูก ignore และจะไม่ขึ้น Git

## 3. สร้างฐานข้อมูลใหม่

ยังอยู่ใน virtual environment และ root ของโปรเจกต์:

```bash
mkdir -p storage/projects storage/models temp/training temp/auto_label
alembic upgrade head
```

Alembic จะสร้าง `cv_platform.sqlite3` เปล่าและสร้าง schema ล่าสุดให้ ไม่ต้องสร้างตารางด้วยมือ หลังเปิดเว็บให้กด Create account เพื่อสร้างผู้ใช้ใหม่

ตรวจสอบ migration:

```bash
alembic current
```

## 4. ติดตั้ง Frontend

```bash
cd frontend
npm ci
```

`npm ci` จะติดตั้งตาม `package-lock.json` เพื่อให้เวอร์ชันใกล้เคียงกับเครื่องเดิม; setup script จะรัน `npm run build` ตรวจ frontend ให้ด้วย

## 5. รันระบบ

Terminal 1 สำหรับ Backend:

```bash
cd "detect anything"
source venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 สำหรับ Frontend:

```bash
cd "detect anything/frontend"
npm run dev
```

เปิดเว็บที่ `http://127.0.0.1:5173` และ API docs ที่ `http://127.0.0.1:8000/docs`

Vite ถูกตั้งค่า `strictPort: true` จึงใช้เฉพาะ port 5173 หากขึ้น `Port 5173 is already in use` ให้หยุด process เก่าก่อน:

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
kill <PID>
```

ตรวจ Backend port ในลักษณะเดียวกัน:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

## 6. ตรวจสอบก่อนเริ่มพัฒนา

```bash
source venv/bin/activate
pytest
cd frontend
npm run build
```

## กรณีต้องการย้ายข้อมูลในอนาคต

Git ไม่เหมาะกับฐานข้อมูล รูปจำนวนมาก หรือ model weights หากต้องการย้ายสถานะเดิมครบทั้งหมด ต้องสำรองอย่างน้อย:

- `cv_platform.sqlite3`
- `storage/projects/`
- `storage/models/`
- pretrained weights ที่ต้องการใช้

ต้องย้ายฐานข้อมูลและ `storage` จาก snapshot ช่วงเวลาเดียวกัน ไม่เช่นนั้น record ในฐานข้อมูลอาจชี้ไปยังไฟล์ที่ไม่มีอยู่
