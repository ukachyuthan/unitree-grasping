#!/usr/bin/env python3
"""
Downloads the YCB object set (google_16k meshes) for use as real-object
training/eval seeds in scripts/generate_ycb_meshes.py.

Standalone script — no Isaac Lab needed.

Usage:
    ./rl_unitree/bin/python scripts/fetch_ycb.py --out data/ycb_raw
    ./rl_unitree/bin/python scripts/fetch_ycb.py --out data/ycb_raw --names 006_mustard_bottle 011_banana
"""

import argparse
import os
import sys
import tarfile
import urllib.error
import urllib.request

YCB_BASE_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{name}_google_16k.tgz"

# Canonical YCB object set (Calli et al. 2015). Not guaranteed byte-perfect —
# any name that 404s on the server is just logged and skipped.
YCB_NAMES = [
    "001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box",
    "009_gelatin_box", "010_potted_meat_can", "011_banana", "012_strawberry",
    "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum",
    "019_pitcher_base", "021_bleach_cleanser", "022_windex_bottle", "024_bowl",
    "025_mug", "026_sponge", "027_skillet", "028_skillet_lid", "029_plate",
    "030_fork", "031_spoon", "032_knife", "033_spatula", "035_power_drill",
    "036_wood_block", "037_scissors", "038_padlock", "040_large_marker",
    "041_small_marker", "042_adjustable_wrench", "043_phillips_screwdriver",
    "044_flat_screwdriver", "048_hammer", "050_medium_clamp", "051_large_clamp",
    "052_extra_large_clamp", "053_mini_soccer_ball", "054_softball", "055_baseball",
    "056_tennis_ball", "057_racquetball", "058_golf_ball", "059_chain",
    "061_foam_brick", "062_dice", "063-a_marbles", "065-a_cups", "065-b_cups",
    "065-c_cups", "065-d_cups", "065-e_cups", "065-f_cups", "065-g_cups",
    "065-h_cups", "065-i_cups", "065-j_cups", "070-a_colored_wood_blocks",
    "070-b_colored_wood_blocks", "071_nine_hole_peg_test", "072-a_toy_airplane",
    "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo", "073-d_lego_duplo",
    "073-e_lego_duplo", "073-f_lego_duplo", "073-g_lego_duplo", "076_timer",
    "077_rubiks_cube",
]


def fetch_one(name: str, out_dir: str, timeout: float = 60.0) -> bool:
    dest_dir = os.path.join(out_dir, name)
    if os.path.isdir(dest_dir) and os.listdir(dest_dir):
        print(f"  [{name}] cached, skipping")
        return True

    url = YCB_BASE_URL.format(name=name)
    tgz_path = os.path.join(out_dir, f"{name}.tgz")
    try:
        urllib.request.urlretrieve(url, tgz_path)
    except urllib.error.HTTPError as e:
        print(f"  [{name}] HTTP {e.code}, skipping")
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [{name}] download failed ({e}), skipping")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tgz_path) as tf:
        tf.extractall(dest_dir)
    os.remove(tgz_path)
    print(f"  [{name}] ok")
    return True


def main():
    p = argparse.ArgumentParser(description="Download YCB object meshes (google_16k)")
    p.add_argument("--out", type=str, default="data/ycb_raw")
    p.add_argument("--names", type=str, nargs="*", default=None,
                   help="Subset of YCB object names to fetch (default: full set)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = args.names if args.names else YCB_NAMES

    print(f"Fetching {len(names)} YCB objects -> {args.out}/")
    ok = 0
    for i, name in enumerate(names):
        print(f"[{i+1:3d}/{len(names)}] {name}")
        if fetch_one(name, args.out):
            ok += 1

    print(f"\nDone. {ok}/{len(names)} objects fetched to {args.out}/")
    if ok == 0:
        sys.exit("No objects fetched — check network access / YCB_NAMES.")


if __name__ == "__main__":
    main()
