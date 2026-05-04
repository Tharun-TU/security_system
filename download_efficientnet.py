"""
Downloads EfficientNet-B0 weights with SSL verification disabled.
Run this if the normal download fails due to network/firewall issues.
"""
import os, ssl, urllib.request

URL  = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
DEST = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                    "checkpoints", "efficientnet_b0_rwightman-7f5810bc.pth")

os.makedirs(os.path.dirname(DEST), exist_ok=True)

if os.path.exists(DEST):
    print(f"[OK] Already exists: {DEST}")
else:
    print(f"Downloading EfficientNet-B0 weights...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def progress(count, block, total):
        pct = min(count * block / total * 100, 100)
        print(f"\r  {pct:.1f}%  ({count*block/1e6:.1f} MB / {total/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(URL, DEST, reporthook=progress,
                               cafile=None)
    # Override ssl context
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx)
    )
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(URL, DEST, reporthook=progress)
        print(f"\n[OK] Saved to: {DEST}")
    except Exception as e:
        print(f"\n[FAIL] {e}")
        print("\nManual download:")
        print(f"  1. Open in browser: {URL}")
        print(f"  2. Save file to:    {DEST}")
