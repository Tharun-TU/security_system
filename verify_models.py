import torch, cv2, numpy as np, os, sys
sys.path.insert(0, '.')
from ultralytics import YOLO

print('=== MODEL VERIFICATION ===\n')

# 1. Check files exist
models = {
    'fire_detector.pt':      'models/fire_detector.pt',
    'weapon_detector.pt':    'models/weapon_detector.pt',
    'activity_classifier.pt':'models/activity_classifier.pt',
}
for name, path in models.items():
    if os.path.exists(path):
        size = round(os.path.getsize(path)/1024/1024, 1)
        print(f'  [OK] {name}  ({size} MB)')
    else:
        print(f'  [MISSING] {name} — place in models/ folder!')

print()

# 2. Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {device}')
if device == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

print()

# Blank frame for dummy inference
dummy = np.zeros((480, 640, 3), dtype=np.uint8)

# 3. Fire model
print('Testing fire_detector.pt ...')
fire = YOLO('models/fire_detector.pt')
fire(dummy, verbose=False, device=device)
print(f'  [OK] classes = {list(fire.names.values())}')

# 4. Weapon model
print('Testing weapon_detector.pt ...')
weapon = YOLO('models/weapon_detector.pt')
weapon(dummy, verbose=False, device=device)
print(f'  [OK] classes = {list(weapon.names.values())}')

# 5. Activity model
print('Testing activity_classifier.pt ...')
ckpt = torch.load('models/activity_classifier.pt', map_location=device)
cls  = ckpt.get('class_names', '?')
seq  = ckpt.get('seq_len',     '?')
img  = ckpt.get('img_size',    '?')
print(f'  [OK] classes={cls}  seq_len={seq}  img_size={img}')

print()
print('=== ALL CHECKS PASSED — READY TO RUN ===')
print('Run:  python realtime_detection.py')
