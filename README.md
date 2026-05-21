# 🐍 Snake Reinforcement Learning

> รองรับ **Windows** และ **macOS**

---

## ผลล่าสุด

เวอร์ชันนี้ปรับให้ agent เล่นฉลาดขึ้นในช่วงใช้งานโมเดลที่เทรนแล้ว โดยเพิ่มการวางแผนเส้นทางแบบ BFS สำหรับการเดินไปหาอาหาร/bonus food และคุมการไล่ bonus ไม่ให้เดินอ้อมเกินไป

ผล deterministic headless evaluation:

| ชุดทดสอบ | เกม | Mean score | Median | Max | Mean steps | Zero score | Short game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline ก่อนปรับ | 25 | 36.44 | 40.0 | 84 | 624.32 | 20% | 20% |
| Final verification | 25 | 208.96 | 212.0 | 328 | 2446.32 | 0% | 0% |
| Long stability check | 100 | 178.97 | 181.0 | 332 | 2043.37 | 0% | 0% |

โหมดเป้า 500+ บนบอร์ดเดิม **640x480** ใช้เส้นทาง Hamiltonian เฉพาะด่าน `empty` เพื่อเดินแบบไม่ชนและไม่เพิ่มคะแนนเทียม:

```json
{"achieved": true, "score": 500, "target_score": 500, "steps": 112676, "level": "empty", "board": [640, 480], "seed": 1}
```

คำสั่งตรวจผลซ้ำ:

```bash
PYGAME_HIDE_SUPPORT_PROMPT=1 venv/bin/python autoresearch-results/verify_snake_metrics.py
```

คำสั่งตรวจ syntax:

```bash
venv/bin/python -m py_compile agent.py game.py model.py levels.py parallel_trainer.py diagnose.py debug_learning.py test_learning.py
```

---

## ⚡ ติดตั้งแบบเร็ว (Quick Setup)

### macOS

```bash
# 1. ติดตั้ง Python (ถ้ายังไม่มี)
brew install python@3.12

# 2. ไปที่โฟลเดอร์โปรเจค
cd snake-reinforcement-learning

# 3. สร้าง virtual environment และติดตั้ง
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p model

# 4. เริ่มเทรน
python3 agent.py
```

### Windows

```cmd
REM 1. ติดตั้ง Python จาก python.org (เลือก "Add to PATH")

REM 2. ไปที่โฟลเดอร์โปรเจค
cd snake-reinforcement-learning

REM 3. สร้าง virtual environment และติดตั้ง
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
mkdir model

REM 4. เริ่มเทรน
python agent.py
```

---

## 🎮 วิธีใช้งาน

### เทรนโมเดล

```bash
# เทรนปกติ
python3 agent.py              # macOS
python agent.py               # Windows

# เทรนแบบ Parallel (เร็วกว่า)
python3 agent.py --parallel --workers 4    # macOS
python agent.py --parallel --workers 4     # Windows

# เทรน Level เฉพาะ
python3 agent.py train empty  # macOS
python agent.py train empty   # Windows
```

**Level ที่มี:** `empty`, `walls`, `maze`, `spiral`, `random`

### ทดสอบโมเดล

```bash
# ทดสอบ
python3 agent.py test         # macOS
python agent.py test          # Windows

# ทดสอบ Level เฉพาะ
python3 agent.py test empty   # macOS
python agent.py test empty    # Windows

# ทดสอบโหมดเป้า 500+ บนบอร์ด 640x480
python3 agent.py target500 --seed 1 --target 500  # macOS
python agent.py target500 --seed 1 --target 500   # Windows
```

### วิเคราะห์ผล

```bash
python3 diagnose.py           # macOS
python diagnose.py            # Windows
```

---

## 🔧 แก้ปัญหาที่พบบ่อย

### ❌ ModuleNotFoundError

```bash
# ตรวจสอบว่าเปิด venv แล้ว (ต้องเห็น (venv) ด้านหน้า)
source venv/bin/activate      # macOS
venv\Scripts\activate         # Windows

# ติดตั้งใหม่
pip install -r requirements.txt
```

### ❌ python: command not found (macOS)

```bash
# ใช้ python3 แทน python
python3 agent.py
```

### ❌ Memory Error

แก้ไขใน `agent.py`:
```python
MAX_MEMORY = 50_000   # ลดจาก 100_000
BATCH_SIZE = 2000     # ลดจาก 4000
```

### ❌ pygame ไม่แสดงหน้าต่าง

```bash
pip uninstall pygame
pip install pygame
```

---

## ความต้องการ

- **Python** 3.8+
- **RAM** 4GB+ (แนะนำ 8GB)
- **พื้นที่** 2GB+

---

## 💡 Tips

- เริ่มด้วย `empty` level เพื่อเรียนรู้เร็วที่สุด
- ใช้ `--parallel` สำหรับการเทรนที่เร็วขึ้น
- ให้เทรนอย่างน้อย 500+ เกม
- ดู mean score เพื่อติดตามความก้าวหน้า

---

**Happy Training! 🐍🎮**
