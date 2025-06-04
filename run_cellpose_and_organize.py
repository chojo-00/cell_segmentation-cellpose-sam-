import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# 사용자 설정
image_path = r"C:\Users\muyer\Desktop\cellpose\test\im_20230921111405_004_1825472413.png"
image_name = os.path.splitext(os.path.basename(image_path))[0]
result_dir = os.path.join(r"C:\Users\muyer\Desktop\cellpose\result", image_name)
os.makedirs(result_dir, exist_ok=True)

# 1. Cellpose-SAM 실행
print("[🧠] Cellpose 실행 중...")
command = [
    "python", "-m", "cellpose",
    "--image_path", image_path,
    "--use_gpu",
    "--model", "cpsam",
    "--diameter", "50",
    "--chan", "0", "--chan2", "2",
    "--save_png",
    "--save_flows",
    "--verbose",
    "--no_resample",
    "--flow_threshold", "0.4",
    "--cellprob_threshold", "-0.5",
    "--min_size", "200",
    "--exclude_on_edges",
    "--savedir", result_dir
]
subprocess.run(command)

# 2. seg.npy 파일 이동
seg_src = image_path.replace(".png", "_seg.npy")
seg_dst = os.path.join(result_dir, os.path.basename(seg_src))
if os.path.exists(seg_src):
    shutil.move(seg_src, seg_dst)
    print(f"[✅] seg.npy 이동 완료: {seg_dst}")
else:
    print("[⚠️] seg.npy 파일이 존재하지 않음")

# 3. 시각화 생성
print("[🎨] 시각화 생성 중...")

# 마스크 로드
data = np.load(seg_dst, allow_pickle=True)
if isinstance(data, dict) or hasattr(data, 'item'):
    mask = data.item().get("masks")
else:
    mask = data

if mask is None or not np.any(mask):
    raise ValueError("유효한 마스크가 존재하지 않습니다.")

# === 세포 수 출력 ===
n_cells = len(np.unique(mask)) - 1
print("세포 수:", n_cells)


# 컬러 마스크 이미지 생성
cmap = plt.get_cmap('tab20', np.max(mask)+1)
norm = mcolors.BoundaryNorm(np.arange(0, np.max(mask)+2)-0.5, cmap.N)
colored_mask = cmap(norm(mask))[..., :3]  # RGB만

colored_path = os.path.join(result_dir, f"{image_name}_colored_mask.png")
Image.fromarray((colored_mask * 255).astype(np.uint8)).save(colored_path)

# 오버레이 이미지 생성
orig_img = Image.open(image_path).convert("RGB").resize((mask.shape[1], mask.shape[0]))
overlay = (np.array(orig_img) * 0.5 + colored_mask * 255 * 0.5).astype(np.uint8)
overlay_path = os.path.join(result_dir, f"{image_name}_overlay.png")
Image.fromarray(overlay).save(overlay_path)

# 3분할 이미지 생성
concat = np.concatenate([
    np.array(orig_img),
    (colored_mask * 255).astype(np.uint8),
    overlay
], axis=1)
concat_path = os.path.join(result_dir, f"{image_name}_concat.png")
Image.fromarray(concat).save(concat_path)

print(f"[✅] 시각화 저장 완료: {result_dir}")
