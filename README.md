# 🐍 Snake Reinforcement Learning - Installation Guide

> **คู่มือการติดตั้ง / Installation Guide**  
> รองรับ Windows และ macOS / Supports Windows and macOS

---

## 📋 Table of Contents / สารบัญ

- [ภาษาไทย](#ภาษาไทย)
  - [ความต้องการของระบบ](#ความต้องการของระบบ)
  - [การติดตั้งสำหรับ macOS](#การติดตั้งสำหรับ-macos)
  - [การติดตั้งสำหรับ Windows](#การติดตั้งสำหรับ-windows)
  - [การใช้งาน](#การใช้งาน)
  - [การแก้ปัญหา](#การแก้ปัญหา)
- [English](#english)
  - [System Requirements](#system-requirements)
  - [Installation for macOS](#installation-for-macos)
  - [Installation for Windows](#installation-for-windows)
  - [Usage](#usage)
  - [Troubleshooting](#troubleshooting)

---

# ภาษาไทย

## ความต้องการของระบบ

### ข้อกำหนดพื้นฐาน
- **Python**: เวอร์ชัน 3.8 หรือสูงกว่า
- **RAM**: อย่างน้อย 4GB (แนะนำ 8GB สำหรับการเทรนแบบ parallel)
- **พื้นที่ว่าง**: อย่างน้อย 2GB
- **GPU**: ไม่จำเป็น แต่แนะนำสำหรับการเทรนที่เร็วขึ้น (CUDA-compatible)

### ตรวจสอบเวอร์ชัน Python

**macOS:**
```bash
python3 --version
```

**Windows:**
```cmd
python --version
```

> ⚠️ **หมายเหตุ**: หากยังไม่มี Python ติดตั้ง ให้ดาวน์โหลดจาก [python.org](https://www.python.org/downloads/)

---

## การติดตั้งสำหรับ macOS

### ขั้นตอนที่ 1: ติดตั้ง Homebrew (ถ้ายังไม่มี)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### ขั้นตอนที่ 2: ติดตั้ง Python

```bash
brew install python@3.12
```

### ขั้นตอนที่ 3: Clone โปรเจค

```bash
# ไปยังโฟลเดอร์ที่ต้องการ
cd ~/Documents

# Clone repository (หรือดาวน์โหลด ZIP)
git clone <repository-url>
cd snake-reinforcement-learning
```

### ขั้นตอนที่ 4: สร้าง Virtual Environment

```bash
# สร้าง virtual environment
python3 -m venv venv

# เปิดใช้งาน virtual environment
source venv/bin/activate
```

> 💡 **เคล็ดลับ**: เมื่อเปิดใช้งานสำเร็จ จะเห็น `(venv)` ด้านหน้า terminal

### ขั้นตอนที่ 5: ติดตั้ง Dependencies

```bash
# อัพเกรด pip
pip install --upgrade pip

# ติดตั้ง packages ทั้งหมด
pip install -r requirements.txt
```

### ขั้นตอนที่ 6: สร้างโฟลเดอร์สำหรับโมเดล

```bash
mkdir -p model
```

---

## การติดตั้งสำหรับ Windows

### ขั้นตอนที่ 1: ติดตั้ง Python

1. ดาวน์โหลด Python จาก [python.org](https://www.python.org/downloads/)
2. รันไฟล์ติดตั้ง
3. ✅ **สำคัญ**: เลือก "Add Python to PATH" ก่อนกด Install

### ขั้นตอนที่ 2: ติดตั้ง Git (ถ้ายังไม่มี)

1. ดาวน์โหลดจาก [git-scm.com](https://git-scm.com/download/win)
2. รันไฟล์ติดตั้งด้วยการตั้งค่าเริ่มต้น

### ขั้นตอนที่ 3: Clone โปรเจค

```cmd
# เปิด Command Prompt หรือ PowerShell
# ไปยังโฟลเดอร์ที่ต้องการ
cd %USERPROFILE%\Documents

# Clone repository (หรือดาวน์โหลด ZIP)
git clone <repository-url>
cd snake-reinforcement-learning
```

### ขั้นตอนที่ 4: สร้าง Virtual Environment

```cmd
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน virtual environment
venv\Scripts\activate
```

> 💡 **เคล็ดลับ**: เมื่อเปิดใช้งานสำเร็จ จะเห็น `(venv)` ด้านหน้า command prompt

### ขั้นตอนที่ 5: ติดตั้ง Dependencies

```cmd
# อัพเกรด pip
python -m pip install --upgrade pip

# ติดตั้ง packages ทั้งหมด
pip install -r requirements.txt
```

### ขั้นตอนที่ 6: สร้างโฟลเดอร์สำหรับโมเดล

```cmd
mkdir model
```

---

## การใช้งาน

### 1. เทรนโมเดล (Training)

**โหมดปกติ:**
```bash
# macOS/Linux
python3 agent.py

# Windows
python agent.py
```

**เทรนกับ Level เฉพาะ:**
```bash
# macOS/Linux
python3 agent.py train empty

# Windows
python agent.py train empty
```

**โหมด Parallel Training (เร็วกว่า):**
```bash
# macOS/Linux
python3 agent.py --parallel --workers 4

# Windows
python agent.py --parallel --workers 4
```

> 📊 **Level ที่มี**: `empty`, `walls`, `maze`, `spiral`, `random`

### 2. ทดสอบโมเดล (Testing)

**ทดสอบโมเดลที่เทรนแล้ว:**
```bash
# macOS/Linux
python3 agent.py test

# Windows
python agent.py test
```

**ทดสอบกับ Level เฉพาะ:**
```bash
# macOS/Linux
python3 agent.py test empty

# Windows
python agent.py test empty
```

### 3. วิเคราะห์การเรียนรู้

```bash
# macOS/Linux
python3 diagnose.py

# Windows
python diagnose.py
```

### 4. ปิด Virtual Environment

```bash
# ทั้ง macOS และ Windows
deactivate
```

---

## การแก้ปัญหา

### ❌ ปัญหา: "python: command not found" (macOS)

**วิธีแก้:**
```bash
# ใช้ python3 แทน python
python3 agent.py
```

### ❌ ปัญหา: "pip: command not found"

**วิธีแก้ (macOS):**
```bash
python3 -m pip install --upgrade pip
```

**วิธีแก้ (Windows):**
```cmd
python -m pip install --upgrade pip
```

### ❌ ปัญหา: pygame ไม่แสดงหน้าต่าง (macOS)

**วิธีแก้:**
```bash
# ติดตั้ง pygame ใหม่
pip uninstall pygame
pip install pygame
```

### ❌ ปัญหา: PyTorch ติดตั้งไม่สำเร็จ

**วิธีแก้ (CPU only):**
```bash
# ติดตั้ง PyTorch แบบ CPU
pip install torch torchvision torchaudio
```

**วิธีแก้ (GPU - CUDA):**
```bash
# ตรวจสอบเวอร์ชัน CUDA ก่อน แล้วติดตั้งตาม
# ดูคำแนะนำที่ https://pytorch.org/get-started/locally/
```

### ❌ ปัญหา: Permission denied (macOS/Linux)

**วิธีแก้:**
```bash
# เพิ่มสิทธิ์ให้ไฟล์
chmod +x agent.py
```

### ❌ ปัญหา: ModuleNotFoundError

**วิธีแก้:**
```bash
# ตรวจสอบว่าเปิด virtual environment แล้ว
# ถ้ายังไม่ได้เปิด:

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# จากนั้นติดตั้ง dependencies อีกครั้ง
pip install -r requirements.txt
```

### ❌ ปัญหา: หน่วยความจำไม่พอ (Memory Error)

**วิธีแก้:**
```python
# แก้ไขใน agent.py
MAX_MEMORY = 50_000  # ลดจาก 100_000
BATCH_SIZE = 2000    # ลดจาก 4000
```

---

## 📚 เอกสารเพิ่มเติม

- **Parallel Training**: อ่าน `PARALLEL_TRAINING.md`
- **Level Design**: ดูใน `levels.py`
- **Model Architecture**: ดูใน `model.py`

---

# English

## System Requirements

### Basic Requirements
- **Python**: Version 3.8 or higher
- **RAM**: At least 4GB (8GB recommended for parallel training)
- **Storage**: At least 2GB free space
- **GPU**: Optional but recommended for faster training (CUDA-compatible)

### Check Python Version

**macOS:**
```bash
python3 --version
```

**Windows:**
```cmd
python --version
```

> ⚠️ **Note**: If Python is not installed, download from [python.org](https://www.python.org/downloads/)

---

## Installation for macOS

### Step 1: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python

```bash
brew install python@3.12
```

### Step 3: Clone the Project

```bash
# Navigate to desired folder
cd ~/Documents

# Clone repository (or download ZIP)
git clone <repository-url>
cd snake-reinforcement-learning
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

> 💡 **Tip**: When activated successfully, you'll see `(venv)` prefix in terminal

### Step 5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

### Step 6: Create Model Directory

```bash
mkdir -p model
```

---

## Installation for Windows

### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. ✅ **Important**: Check "Add Python to PATH" before clicking Install

### Step 2: Install Git (if not installed)

1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Run installer with default settings

### Step 3: Clone the Project

```cmd
# Open Command Prompt or PowerShell
# Navigate to desired folder
cd %USERPROFILE%\Documents

# Clone repository (or download ZIP)
git clone <repository-url>
cd snake-reinforcement-learning
```

### Step 4: Create Virtual Environment

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

> 💡 **Tip**: When activated successfully, you'll see `(venv)` prefix in command prompt

### Step 5: Install Dependencies

```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

### Step 6: Create Model Directory

```cmd
mkdir model
```

---

## Usage

### 1. Train Model

**Normal Mode:**
```bash
# macOS/Linux
python3 agent.py

# Windows
python agent.py
```

**Train on Specific Level:**
```bash
# macOS/Linux
python3 agent.py train empty

# Windows
python agent.py train empty
```

**Parallel Training Mode (Faster):**
```bash
# macOS/Linux
python3 agent.py --parallel --workers 4

# Windows
python agent.py --parallel --workers 4
```

> 📊 **Available Levels**: `empty`, `walls`, `maze`, `spiral`, `random`

### 2. Test Model

**Test Trained Model:**
```bash
# macOS/Linux
python3 agent.py test

# Windows
python agent.py test
```

**Test on Specific Level:**
```bash
# macOS/Linux
python3 agent.py test empty

# Windows
python agent.py test empty
```

### 3. Analyze Learning

```bash
# macOS/Linux
python3 diagnose.py

# Windows
python diagnose.py
```

### 4. Deactivate Virtual Environment

```bash
# Both macOS and Windows
deactivate
```

---

## Troubleshooting

### ❌ Issue: "python: command not found" (macOS)

**Solution:**
```bash
# Use python3 instead of python
python3 agent.py
```

### ❌ Issue: "pip: command not found"

**Solution (macOS):**
```bash
python3 -m pip install --upgrade pip
```

**Solution (Windows):**
```cmd
python -m pip install --upgrade pip
```

### ❌ Issue: pygame window not showing (macOS)

**Solution:**
```bash
# Reinstall pygame
pip uninstall pygame
pip install pygame
```

### ❌ Issue: PyTorch installation fails

**Solution (CPU only):**
```bash
# Install PyTorch CPU version
pip install torch torchvision torchaudio
```

**Solution (GPU - CUDA):**
```bash
# Check CUDA version first, then install accordingly
# See instructions at https://pytorch.org/get-started/locally/
```

### ❌ Issue: Permission denied (macOS/Linux)

**Solution:**
```bash
# Add execute permission
chmod +x agent.py
```

### ❌ Issue: ModuleNotFoundError

**Solution:**
```bash
# Check if virtual environment is activated
# If not activated:

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Then reinstall dependencies
pip install -r requirements.txt
```

### ❌ Issue: Memory Error

**Solution:**
```python
# Edit in agent.py
MAX_MEMORY = 50_000  # Reduce from 100_000
BATCH_SIZE = 2000    # Reduce from 4000
```

---

## 📚 Additional Documentation

- **Parallel Training**: See `PARALLEL_TRAINING.md`
- **Level Design**: Check `levels.py`
- **Model Architecture**: Review `model.py`

---

## 🎮 Quick Start Guide

### First Time Setup (5 minutes)

1. **Install Python** (if needed)
2. **Clone/Download** this project
3. **Open Terminal/Command Prompt** in project folder
4. **Run these commands:**

   **macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   mkdir -p model
   python3 agent.py
   ```

   **Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   mkdir model
   python agent.py
   ```

5. **Watch the snake learn!** 🐍🎯

---

## 💡 Tips for Best Results

### Training Tips
- Start with `empty` level for fastest learning
- Use parallel training with 4-8 workers for speed
- Let it train for at least 500 games for good results
- Monitor the mean score - it should increase over time

### Performance Tips
- Close other applications to free up RAM
- Use GPU if available (requires CUDA setup)
- Reduce `BATCH_SIZE` if running out of memory
- Set `render=False` in `game.py` for faster training

### Testing Tips
- Test on different levels to see generalization
- Watch the epsilon value - lower means more exploitation
- Check `model/training_state.json` for progress
- Use `diagnose.py` to analyze learning curves

---

## 🆘 Need Help?

If you encounter issues not covered here:

1. Check that Python version is 3.8+
2. Verify virtual environment is activated
3. Ensure all dependencies installed correctly
4. Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`
5. Check for error messages in terminal

---

## 📝 License

This project is for educational purposes. Feel free to modify and experiment!

---

**Happy Training! / สนุกกับการเทรน! 🐍🎮**
