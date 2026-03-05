INSTALL:
Create Virtual Environment:
```
python3 -m venv .venv
```
Install dependency:
```
pip install opencv-contrib-python --break-system-packages
```

FIRST TIME SETUP (enroll your face):
```
python3 face-detect.py --enroll
```

DAILY USE:
```
python3 face-detect.py
```

AUTO-START ON LOGIN:
```
python3 face-detect.py --install
```
