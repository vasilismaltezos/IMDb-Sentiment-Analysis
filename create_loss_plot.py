
import re
from pathlib import Path

LOG_FILE = Path("imdb_rnn_console_log.txt")

# Accepts lines like:
# Epoch 1/6: Train Loss=0.6932 Acc=0.5012 | Dev Loss=0.6810 Acc=0.5320
EPOCH_RE = re.compile(
    r"Epoch\s+(?P<ep>\d+)\s*/\s*(?P<tot>\d+)\s*:\s*"
    r"Train\s+Loss\s*=\s*(?P<trloss>\d+\.\d+)\s+Acc\s*=\s*(?P<tracc>\d+\.\d+)\s*\|\s*"
    r"Dev\s+Loss\s*=\s*(?P<dvloss>\d+\.\d+)\s+Acc\s*=\s*(?P<dvacc>\d+\.\d+)",
    re.IGNORECASE
)

# If your script prints "Best epoch: X | dev loss: Y"
BEST_RE = re.compile(r"Best\s+epoch\s*:\s*(\d+)", re.IGNORECASE)

def parse_log(text: str):
    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group("ep"))
            tr_loss = float(m.group("trloss"))
            dv_loss = float(m.group("dvloss"))
            rows.append((ep, tr_loss, dv_loss))

    best_epoch_printed = None
    m2 = BEST_RE.search(text)
    if m2:
        best_epoch_printed = int(m2.group(1))

    return rows, best_epoch_printed

def main():
    if not LOG_FILE.exists():
        print("ERROR: Could not find imdb_rnn_console_log.txt in this folder.\n")
        print("Create it WITHOUT changing any code like this (PowerShell):")
        print("  py imdb_rnn.py | Tee-Object -FilePath imdb_rnn_console_log.txt\n")
        print("Then run:")
        print("  py create_loss_plot.py")
        return

    text = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
    rows, best_epoch_printed = parse_log(text)

    if not rows:
        print("ERROR: I couldn't find any epoch lines in the log.")
        print("Expected lines like:")
        print("  Epoch 1/6: Train Loss=... Acc=... | Dev Loss=... Acc=...")
        print("\nOpen imdb_rnn_console_log.txt and confirm it contains those lines.")
        return

    # Determine best epoch: prefer printed best epoch; otherwise min dev loss
    if best_epoch_printed is not None:
        best_epoch = best_epoch_printed
    else:
        best_epoch = min(rows, key=lambda x: x[2])[0]  # epoch with min dev loss

    # Print ASCII table
    print("=" * 70)
    print("IMDB RNN LOSS CURVES (ASCII Visualization)")
    print("=" * 70)
    print("\nEpoch | Train Loss | Dev Loss   | Note")
    print("------|------------|------------|" + "-" * 30)

    for ep, tr_loss, dv_loss in rows:
        marker = " <- BEST" if ep == best_epoch else ""
        print(f"{ep:5d} | {tr_loss:10.4f} | {dv_loss:10.4f} |{marker}")

    print("\n" + "=" * 70)
    print(f"Best epoch: {best_epoch}")
    print("Tip for report: If train loss keeps decreasing while dev loss rises → overfitting.")
    print("=" * 70)

if __name__ == "__main__":
    main()

