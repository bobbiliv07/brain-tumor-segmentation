import time
from pathlib import Path

log_dir = Path(
    r"C:\Users\Admin\Desktop\meghana\Optimized\nnunet_results"
    r"\Dataset500_BraTSGlioma"
    r"\nnUNetTrainer__nnUNetPlans__3d_fullres"
    r"\fold_0"
)

print("📂 Monitoring directory:", log_dir)
print("=" * 60)

last_size = 0
log_file = None

while True:
    logs = sorted(log_dir.glob("training_log_*.txt"))

    if not logs:
        print("⏳ Waiting for log file...")
        time.sleep(5)
        continue

    latest_log = logs[-1]

    if log_file != latest_log:
        print(f"\n📄 Switched to: {latest_log.name}\n")
        log_file = latest_log
        last_size = 0

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(last_size)
        new_data = f.read()

        if new_data:
            print(new_data, end="")

        last_size = f.tell()

    time.sleep(2)           