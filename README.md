# Detect Anything

เว็บจัดการภาพและ annotation สำหรับ Object Detection / Pose Estimation พร้อมสร้าง dataset version, เทรน YOLO, ทดสอบโมเดล และตรวจผล auto-label

คีย์ลัดหน้า Annotate: ดู [frontend/shortcut-key.md](frontend/shortcut-key.md)

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

หากต้องการติดตั้งทั้งหมดด้วยคำสั่งเดียวบน macOS หรือ Linux (รวมเครื่อง NVIDIA CUDA) ให้รันจาก root โปรเจกต์:

```bash
bash setup-scripts/setup-new-machine.sh
```

สคริปต์ตรวจระบบและ GPU แล้วเลือก PyTorch ที่เหมาะสม ติดตั้ง Backend ใน `venv/`, ติดตั้ง Frontend ด้วย `npm ci`, ดาวน์โหลดและตรวจ `yolo26n-pose.pt` กับ `yolo26s.pt` ที่ root โปรเจกต์ แล้วสำรอง SQLite เดิมก่อนทำ `alembic upgrade head` ต้องมี Git, Python 3.11/3.12, Node.js 20+ และ npm อยู่ก่อน; สคริปต์จะแจ้งหากขาดอย่างใดอย่างหนึ่ง และไม่เขียนทับ `.env` หากใช้สคริปต์นี้แล้วข้ามขั้นตอนติดตั้ง 2-5 ด้านล่างได้สำหรับโมเดลค่าเริ่มต้น รุ่นอื่นที่เลือกเทรนจะดาวน์โหลดเมื่อใช้งาน

ตรวจว่าเครื่องจะใช้ทางเลือกใดโดยไม่ติดตั้งหรือแก้ฐานข้อมูลได้ด้วย `bash setup-scripts/setup-new-machine.sh --check`

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
cd "frontend"
npm run dev
```

เปิดเว็บที่ <http://127.0.0.1:5173> และ API docs ที่ <http://127.0.0.1:8000/docs> เครื่องใหม่ให้สร้าง account และ project ใหม่ในเว็บ

### นำเข้าโมเดลจากภายนอก

เปิดโปรเจกต์ที่ต้องการ แล้วไปที่ **Models → Import model** (แยกจากหน้า Download) เลือกไฟล์ Ultralytics YOLO `.pt` ขนาดไม่เกิน 200 MB ระบบจะตรวจว่าเป็น Detect หรือ Pose ให้ตรงกับชนิดโปรเจกต์ อ่านชื่อคลาสจากไฟล์ และเพิ่มคลาสที่ยังไม่มีในโปรเจกต์ให้อัตโนมัติ หลังนำเข้า โมเดลจะปรากฏในหน้า Test, Auto Label, Training setup และ Download ของ **โปรเจกต์นั้นเท่านั้น**

ถ้าโปรเจกต์มีคลาส index เดียวกันแต่ชื่อไม่ตรงกับโมเดล ระบบจะปฏิเสธการนำเข้าโดยไม่แก้หรือเขียนทับคลาส/annotation เดิม ให้ใช้โปรเจกต์ใหม่หรือจัดคลาสให้ตรงก่อน คลาสที่โมเดลนำเข้าใช้อยู่จะเปลี่ยนชื่อหรือลบไม่ได้จนกว่าจะลบโมเดลนั้น โปรเจกต์ Pose รองรับโมเดล 17 keypoints เท่านั้น ข้อมูลโมเดลถูกบันทึกในฐานข้อมูลพร้อม `project_id`, ผู้ที่อัปโหลด, ชนิดงาน และชื่อคลาส; ไฟล์ weights อยู่ใน `storage/models/` จึงต้องย้าย **ฐานข้อมูลและ storage พร้อมกัน** เมื่อต้องการย้ายเครื่อง

นำเข้าเฉพาะ `.pt` จากแหล่งที่เชื่อถือได้: ระบบอ่าน checkpoint ด้วย safe loading แล้วบันทึกไฟล์ที่สร้างใหม่จากโมเดลที่ผ่านการตรวจ แต่ไฟล์ PyTorch/Ultralytics ไม่ควรถูกถือว่าปลอดภัยโดยอาศัยนามสกุลไฟล์อย่างเดียว

หลังดึงโค้ดนี้ไปเครื่องอื่น ให้รัน `python -m pip install -e ".[dev]"` และ `alembic upgrade head` จาก root โปรเจกต์ก่อนเปิด backend เพื่อให้เวอร์ชัน Ultralytics รองรับ safe loading และฐานข้อมูลมี schema นำเข้าโมเดล

ครั้งถัดไปใช้แค่สอง terminal ด้านบน ไม่ต้องติดตั้ง dependencies หรือดาวน์โหลด weights ซ้ำ `alembic upgrade head` รันซ้ำได้และควรรันหลังดึงโค้ดที่มี migration ใหม่

Frontend เรียก API ผ่าน path `/api` ที่ origin เดียวกันโดยค่าเริ่มต้น แล้ว Vite proxy ส่งต่อไปยัง backend ที่ `http://127.0.0.1:8000` วิธีนี้ทำให้ทั้งการใช้งาน local และการเปิด tunnel เฉพาะ Frontend ใช้ค่า API เดียวกันได้ Backend และ Frontend อ่านค่าจาก root `.env` ไฟล์เดียวกัน

### ถ้าพอร์ตถูกใช้อยู่

อย่าเปิด server ซ้อน: ตรวจว่ามี backend/frontend เดิมกำลังทำงานหรือไม่ แล้วใช้ของเดิม หรือหยุด process ที่ **ตัวเองตั้งใจจะเปลี่ยน** ด้วย `Ctrl+C` ใน terminal เดิม

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

หากต้องการหยุด server เดิมและกลับไปที่ terminal ที่เปิดไว้ไม่ได้ ให้ดูเลข `PID` จากผล `lsof` ในแถวของพอร์ตที่ต้องการ (8000 คือ Backend, 5173 คือ Frontend) แล้วแทน `12345` ในตัวอย่างด้วยเลข PID ที่พบ เพื่อตรวจชื่อ process ก่อนสั่งหยุด:

```bash
ps -p 12345 -o pid=,command=
kill -TERM 12345
```

เช็กพอร์ตเดิมอีกครั้งด้วยคำสั่ง `lsof` ด้านบน ถ้าไม่แสดง process ที่ `LISTEN` แล้วจึงเปิด server ใหม่ หากมีหลาย `PID` ให้ตรวจและหยุดเฉพาะ process ที่ต้องการทีละตัว

### ตรวจรูป Negative หลัง Auto-label

ปุ่ม **Accept all** ยังอนุมัติผล Auto-label ทั้งชุดตามเดิม แต่รูปที่โมเดลไม่เจอกรอบจะ **ยังไม่ถูกใส่ใน dataset** โดยอัตโนมัติ เพราะอาจมีวัตถุที่โมเดลพลาด หลัง Accept all ให้กด **Review no-box** จากหน้าต่าง **Select image** (หรือจากหน้าผล Auto-label) แล้วเปิดดูภาพใหญ่ทีละรูปในหน้าต่างเดียวกัน:

- ถ้าไม่มีวัตถุจริง กด **Confirm no objects** เพื่อบันทึกเป็น verified negative
- ถ้ามีวัตถุ กด **Object visible · draw label** แล้วลากกรอบตามปกติ

ตอนสร้าง Dataset version จะเลือกได้ทั้งรูปที่มี approved labels และ verified negatives โดยรูป Negative จะมีไฟล์ YOLO label ว่าง (`.txt`) ส่วนรูปที่ไม่มีกรอบแต่ยังไม่ยืนยันจะไม่เข้า dataset หลังดึงโค้ดรุ่นที่มีฟีเจอร์นี้ ให้รัน `alembic upgrade head` ก่อนเปิด Backend เพื่อเพิ่มคอลัมน์สถานะใหม่ โดยไม่ลบข้อมูลเดิม

Vite ตั้ง `strictPort: true` จึงไม่ย้ายจาก 5173 ไป 5174 เอง

## เปิดให้ทดสอบผ่าน ngrok URL เดียว

โครงสร้างนี้เปิด public tunnel แค่ Frontend ส่วน Backend ยังฟังเฉพาะในเครื่อง:

```text
Browser -> https://example.ngrok-free.app/api/... -> ngrok -> Vite 127.0.0.1:5173
                                                               -> Backend 127.0.0.1:8000
```

สร้าง root `.env` จากตัวอย่างและสร้าง secret แบบสุ่มก่อนเปิดให้เครื่องอื่นเข้าใช้:

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

นำค่าที่สุ่มได้ไปใส่ `AUTH_JWT_SECRET` หากยังไม่รู้ hostname ให้รัน `ngrok http 5173` เพื่อดู Forwarding URL ก่อน แล้วตั้งค่า hostname จริงโดยใส่เฉพาะ hostname ไม่ใส่ `https://` หรือ path:

```dotenv
AUTH_JWT_SECRET=<GENERATED_SECRET>
VITE_API_BASE_URL=/api
API_PROXY_TARGET=http://127.0.0.1:8000
VITE_ALLOWED_HOSTS=example.ngrok-free.app
```

เปิด Backend และ Frontend ตามหัวข้อรันระบบ แล้วให้ ngrok ทำงานเป็น process ที่สาม:

```bash
ngrok http 5173
```

ผู้ใช้เปิด URL HTTPS ของ ngrok ได้โดยไม่ต้องเข้าถึงพอร์ต 8000 โดยตรง Request เช่น `/api/auth/login` จะถูก Vite ตัด prefix `/api` แล้วส่งเป็น `/auth/login` ไปยัง Backend หลังสร้างหรือแก้ `.env` ต้อง restart Backend และ Frontend หาก hostname ของ ngrok เปลี่ยน ให้แก้ `VITE_ALLOWED_HOSTS` และ restart Frontend อีกครั้ง

โหมดนี้ไม่ต้องเพิ่ม public ngrok origin ใน `CORS_ORIGINS` เพราะ Browser เรียกหน้าเว็บและ `/api` จาก origin เดียวกัน ไม่ต้องเปลี่ยน Uvicorn เป็น `0.0.0.0` และไม่ควรตั้ง `VITE_ALLOWED_HOSTS=true`

ถ้าติดตั้ง `screen` และ `ngrok` พร้อมตั้ง `AUTH_JWT_SECRET` ใน root `.env` แล้ว สามารถเปิด Backend, Frontend และ ngrok ใน screen สาม session ด้วยคำสั่งเดียวจาก root โปรเจกต์:

```bash
bash start-scripts.sh
```

สคริปต์หา root จากตำแหน่งไฟล์เอง จึงใช้ได้แม้โฟลเดอร์ชื่อ `detectanything-pj` โดยจะอ่าน HTTPS hostname ที่ ngrok ออกให้ แล้วอนุญาต hostname นั้นเฉพาะ Vite process ที่เปิดขึ้น ไม่เขียนทับ `.env` ดู log ได้ที่ `backend.log`, `fontend.log`, `ngrok.log` และใช้ `screen -ls` ตรวจ session ที่รันอยู่ หากพอร์ต 8000, 5173 หรือ 4040 ถูกใช้อยู่ สคริปต์จะหยุดก่อนเปิด server ซ้อน

ลิงก์ tunnel ไม่ใช่กลไกควบคุมสิทธิ์ ระบบเปิดให้สมัครบัญชีได้และงาน upload, training หรือ prediction ใช้พื้นที่และทรัพยากรเครื่อง จึงควรแชร์เฉพาะผู้ทดสอบที่ไว้ใจหรือเพิ่ม access protection ที่ tunnel

## เปิดให้ทดสอบผ่าน Cloudflare Tunnel

โครงสร้างที่รองรับคือใช้ hostname คงที่สองชื่อ โดยผู้ใช้เปิด Frontend และ Frontend เรียก Backend ผ่าน API:

```text
https://app.example.com -> http://127.0.0.1:5173
https://api.example.com -> http://127.0.0.1:8000
```

ควรใช้ Named Tunnel เพื่อให้ URL ไม่เปลี่ยนทุกครั้ง หากใช้โดเมนอื่นให้แทน `app.example.com` และ `api.example.com` ในทุกขั้นตอนด้านล่าง

### 1. ตั้งค่า Backend และ Frontend

สร้าง `.env` จากตัวอย่าง:

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

แก้ค่าต่อไปนี้ใน root `.env` โดยนำค่าสุ่มจากคำสั่งด้านบนมาใช้เป็น secret:

```dotenv
CORS_ORIGINS=https://app.example.com,http://127.0.0.1:5173,http://localhost:5173
AUTH_JWT_SECRET=<GENERATED_SECRET>
VITE_API_BASE_URL=http://127.0.0.1:8000,https://api.example.com
VITE_ALLOWED_HOSTS=app.example.com
```

Backend ใช้ `CORS_ORIGINS` และ `AUTH_JWT_SECRET` ส่วน Vite อ่าน `VITE_` settings และ `API_PROXY_TARGET` จากไฟล์เดียวกัน `VITE_ALLOWED_HOSTS` ใช้เฉพาะ hostname ไม่ใส่ `https://` ส่วน `VITE_API_BASE_URL` รองรับทั้ง same-origin path `/api` และ API URL เต็มพร้อม `http://` หรือ `https://`; คั่นหลาย URL ด้วยเครื่องหมายจุลภาคได้

เมื่อเปิด Frontend ผ่าน `127.0.0.1` หรือ `localhost` ระบบจะเลือก Backend local ที่ระบุไว้ก่อน แล้วลอง public API หาก local เชื่อมต่อไม่ได้ เมื่อเปิดผ่าน hostname สาธารณะ ระบบจะใช้ HTTPS public API เท่านั้น เพราะ `127.0.0.1` ใน browser ของผู้ใช้อื่นหมายถึงเครื่องของผู้ใช้นั้นเอง หากเชื่อมต่อ API ระหว่างอ่านข้อมูลไม่ได้ จะลอง URL ถัดไป; ก่อนส่งคำสั่งเขียนข้อมูลครั้งแรก ระบบจะตรวจว่าปลายทางใช้งานได้ และไม่ส่งคำสั่งเดิมซ้ำอัตโนมัติหากการเชื่อมต่อขาดกลางทาง เพื่อเลี่ยงข้อมูลซ้ำ การสลับ URL ไม่แก้ HTTP error เช่น 401/500 หรือ CORS ที่ตั้งค่าไม่ถูกต้อง

URL ทั้งสองควรชี้ไปยัง Backend และฐานข้อมูลชุดเดียวกัน เพื่อให้บัญชีและข้อมูลโปรเจกต์ตรงกันหลังสลับปลายทาง

localhost ยังคงอยู่ใน `CORS_ORIGINS` จึงใช้เว็บแบบ local ได้ด้วย หากต้องการใช้ Backend local อย่างเดียวให้ตั้ง `VITE_API_BASE_URL=http://127.0.0.1:8000` หลังแก้ `.env` ต้อง restart ทั้ง Backend และ Frontend ห้าม commit `.env`

### 2. ตั้งค่า Tunnel

ติดตั้งและ login `cloudflared` ตามวิธีของ Cloudflare จากนั้นสร้าง Named Tunnel และ DNS routes สำหรับสอง hostname คัดลอกไฟล์ตัวอย่าง:

```bash
cp cloudflare/config.example.yml cloudflare/config.yml
```

ใส่ Tunnel ID, path ของ credentials และ hostname จริงใน `cloudflare/config.yml` แล้วรัน:

```bash
cloudflared tunnel --config cloudflare/config.yml run
```

`cloudflare/config.yml` และ credentials JSON ถูก Git ignore ห้าม commit ไฟล์จริง หากตั้ง Public Hostnames ผ่าน Cloudflare Dashboard อยู่แล้ว สามารถใช้คำสั่ง run/token ที่ Dashboard สร้างให้โดยไม่ต้องสร้าง `config.yml`

### 3. รันทั้งหมด

ต้องมีสาม process ทำงานพร้อมกัน:

```text
Backend    127.0.0.1:8000
Frontend   127.0.0.1:5173
cloudflared tunnel
```

Cloudflare เชื่อมเข้าบริการ local จากเครื่องเดียวกัน จึงไม่ต้องเปลี่ยน Uvicorn หรือ Vite ให้ bind `0.0.0.0` และไม่ต้องเปิดพอร์ตฐานข้อมูลออกอินเทอร์เน็ต หากเครื่องปิดหรือ sleep เว็บไซต์จะเข้าไม่ได้

สำหรับการทดสอบด้วย production build ให้รัน `npm run build` และ `npm run preview` ซึ่งใช้ port `4173` จากนั้นเปลี่ยน service ของ Frontend ใน Tunnel เป็น `http://127.0.0.1:4173`

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

## สำรองข้อมูลเพื่อย้ายเครื่อง

ไม่ควร commit ไฟล์ backup เข้า Git repository เพราะมีฐานข้อมูล รูป และโมเดลขนาดใหญ่ รวมถึงข้อมูลบัญชีภายในระบบ ให้เก็บ archive ผ่าน external drive หรือ private cloud storage แยกจาก source code โฟลเดอร์ `backups/` ถูก `.gitignore` ไว้แล้ว

ก่อนสำรองให้หยุด Backend ด้วย `Ctrl+C` เพื่อให้ไฟล์ SQLite และ `storage/` เป็น snapshot ช่วงเวลาเดียวกัน จากนั้นรันที่ root โปรเจกต์:

```bash
mkdir -p backups
tar -czf "backups/detect-anything-data-$(date +%Y%m%d-%H%M%S).tar.gz" \
  cv_platform.sqlite3 \
  storage/projects \
  storage/models
```

ไฟล์ชุดนี้มี:

- `cv_platform.sqlite3`: account, project, class, annotation, dataset version, training run และ model records
- `storage/projects/`: รูปต้นฉบับของแต่ละโปรเจกต์
- `storage/models/`: weights ของโมเดลที่เทรนสำเร็จหรืออัปโหลดเข้าโปรเจกต์

ไม่ต้องใส่ pretrained weights เช่น `yolo26n-pose.pt` เพราะดาวน์โหลดใหม่ตามขั้นตอนด้านบนได้ และไม่ต้องใส่ `temp/` สำหรับงานที่เทรนเสร็จแล้ว

หากมีงานสถานะ `INTERRUPTED` ที่ต้องการ Resume บนเครื่องใหม่ ให้เพิ่ม `temp/training` เข้า archive:

```bash
mkdir -p backups
tar -czf "backups/detect-anything-data-with-checkpoint-$(date +%Y%m%d-%H%M%S).tar.gz" \
  cv_platform.sqlite3 \
  storage/projects \
  storage/models \
  temp/training
```

ตรวจว่า archive อ่านได้ก่อนย้าย:

```bash
tar -tzf backups/detect-anything-data-*.tar.gz | head
du -sh backups/*.tar.gz
```

ถ้าใช้ `.env` ที่ตั้งค่าเอง ให้ย้ายไฟล์นั้นผ่านช่องทางที่ปลอดภัยแยกต่างหาก ห้ามใส่ `.env` หรือ archive ข้อมูลขึ้น Git

### Restore บนเครื่องใหม่

1. Clone โปรเจกต์และติดตั้ง Backend/Frontend ตามขั้นตอนต้น README
2. นำไฟล์ `.tar.gz` ไปวางใน `backups/` ของโปรเจกต์ใหม่
3. หยุด Backend ถ้ากำลังทำงาน แล้วแตก archive จาก root โปรเจกต์:

```bash
tar -xzf backups/<backup-file-name>.tar.gz -C .
```

4. อัปเดต schema ให้ตรงกับ source code เครื่องใหม่:

```bash
source venv/bin/activate
alembic upgrade head
```

5. ดาวน์โหลด pretrained weights ที่ต้องใช้ แล้วเปิด Backend และ Frontend ตามปกติ

ต้องย้าย `cv_platform.sqlite3` และ `storage/` จาก archive เดียวกันเสมอ หากย้ายคนละช่วงเวลา record ในฐานข้อมูลอาจอ้างถึงรูปหรือโมเดลที่ไม่มีอยู่จริง

รายละเอียดกรณีย้ายข้อมูลเดิม: [docs/new-machine-setup.md](docs/new-machine-setup.md)



### กรณีต้องการเปิด tunnel free cloundflare

```bash
# Terminal 1 Frontend
cloudflared tunnel --url http://127.0.0.1:5173 2>&1 \
  | awk 'match($0, /https:\/\/[-a-zA-Z0-9]+\.trycloudflare\.com/) && !shown {
      print substr($0, RSTART, RLENGTH)
      fflush()
      shown=1
    }'
```

```bash
#Terminal 2 Backend
cloudflared tunnel --url http://127.0.0.1:8000 2>&1 \
  | awk 'match($0, /https:\/\/[-a-zA-Z0-9]+\.trycloudflare\.com/) && !shown {
      print substr($0, RSTART, RLENGTH)
      fflush()
      shown=1
    }'
```
