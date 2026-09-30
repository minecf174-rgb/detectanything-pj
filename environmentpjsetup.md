## Environment Setup

ตรวจสอบระบบปฏิบัติการ:

```bash
uname -s
```

ผลลัพธ์:

```text
Darwin = macOS
Linux  = Linux
```

สำหรับ Linux ให้ตรวจสอบ GPU:

```bash
lspci | grep -Ei 'vga|3d|display'
```

ผลลัพธ์จะช่วยระบุว่าเครื่องใช้:

```text
NVIDIA
AMD / ATI
Intel
```

ถ้าเป็น NVIDIA สามารถตรวจสอบ Driver เพิ่มด้วย:

```bash
nvidia-smi
```

#### macOS

```bash
bash setup-scripts/setup-mac.sh
```

### Linux — CPU / Intel / GPU ที่ไม่ได้ใช้ Hardware Acceleration

```bash
bash setup-scripts/setup-linux-cpu.sh
```

### Linux — NVIDIA GPU / CUDA

```bash
bash setup-scripts/setup-linux-cuda.sh
```

### Linux — AMD GPU / ROCm

สำหรับ AMD GPU ที่รองรับ ROCm และต้องการใช้ GPU Acceleration:

```bash
bash setup-scripts/setup-linux-amd.sh
```

หากเป็น AMD Integrated Graphics หรือ GPU ที่ไม่รองรับ ROCm ให้ใช้:

```bash
bash setup-scripts/setup-linux-cpu.sh
```

หลัง Setup เสร็จ ให้เปิด Virtual Environment:

```bash
source venv/bin/activate
```

ตรวจสอบ PyTorch:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA/ROCm:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

สำหรับ Linux CPU ควรได้ประมาณ:

```text
Torch: 2.5.1+cpu
CUDA/ROCm: None
GPU: False
```

สำหรับ Linux NVIDIA หรือ AMD ROCm ที่ตั้งค่าถูกต้องควรได้:

```text
GPU: True
```