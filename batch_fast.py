#!/usr/bin/env python3
"""
Batch analyze all images using fast grid detection
Outputs results in a nice table format
"""

import cv2
import glob
import os
import time
from pathlib import Path
from tabulate import tabulate

# Import fast detection
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sifter_fast import detect_seeds_fast, draw_detections


def batch_analyze_directory(directory, save_annotated=True):
    """
    Analyze all images in a directory using fast grid detection.

    Args:
        directory: Path to directory containing images
        save_annotated: Whether to save annotated images

    Returns:
        List of result dictionaries
    """
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    image_files = sorted(image_files)

    if not image_files:
        print(f"❌ No images found in {directory}")
        return []

    print(f"\n📁 Found {len(image_files)} images in {directory}")
    print("⚡ Using Fast Grid Detection")
    print("="*80)

    results = []
    total_time = 0

    for i, img_path in enumerate(image_files, 1):
        filename = Path(img_path).name

        # Skip already annotated images
        if '_annotated' in filename or '_fast' in filename or '_sam' in filename:
            continue

        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ {filename}: Could not load")
                continue

            # Detect seeds
            start_time = time.time()
            detections = detect_seeds_fast(image, grid_size=3)
            detection_time = time.time() - start_time
            total_time += detection_time

            # Count by type
            pumpkin_count = sum(1 for d in detections if d['type'] == 'pumpkin')
            sunflower_count = sum(1 for d in detections if d['type'] == 'sunflower')
            total = len(detections)

            # Save annotated version if requested
            if save_annotated:
                annotated, _, _ = draw_detections(image, detections)
                output_path = str(Path(img_path).with_stem(Path(img_path).stem + '_fast'))
                cv2.imwrite(output_path, annotated)

            results.append({
                'filename': filename,
                'pumpkin': pumpkin_count,
                'sunflower': sunflower_count,
                'total': total,
                'time': detection_time
            })

            # Progress indicator
            print(f"[{i}/{len(image_files)}] ✓ {filename:40s} | 🎃 {pumpkin_count:3d} | 🌻 {sunflower_count:3d} | Total: {total:3d} | {detection_time:.3f}s")

        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
            continue

    print("="*80)
    print(f"\n✅ Processed {len(results)} images in {total_time:.2f}s (avg {total_time/len(results):.3f}s per image)")

    return results


def print_summary_table(results):
    """Print results in a nice table format."""
    if not results:
        print("\n❌ No results to display")
        return

    # Prepare table data
    table_data = []
    for r in results:
        table_data.append([
            r['filename'],
            r['pumpkin'],
            r['sunflower'],
            r['total'],
            f"{r['time']:.3f}s"
        ])

    # Print table
    headers = ['Filename', 'Pumpkin', 'Sunflower', 'Total', 'Time']
    print("\n" + "="*80)
    print("📊 DETECTION RESULTS")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Print statistics
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)

    total_pumpkin = sum(r['pumpkin'] for r in results)
    total_sunflower = sum(r['sunflower'] for r in results)
    total_seeds = sum(r['total'] for r in results)
    avg_pumpkin = total_pumpkin / len(results)
    avg_sunflower = total_sunflower / len(results)
    avg_total = total_seeds / len(results)
    avg_time = sum(r['time'] for r in results) / len(results)

    stats_data = [
        ['Total Images', len(results), '', '', ''],
        ['Total Seeds', total_seeds, total_pumpkin, total_sunflower, ''],
        ['Average per Image', f"{avg_total:.1f}", f"{avg_pumpkin:.1f}", f"{avg_sunflower:.1f}", f"{avg_time:.3f}s"],
        ['Min per Image', min(r['total'] for r in results), min(r['pumpkin'] for r in results), min(r['sunflower'] for r in results), ''],
        ['Max per Image', max(r['total'] for r in results), max(r['pumpkin'] for r in results), max(r['sunflower'] for r in results), ''],
    ]

    stats_headers = ['Metric', 'Total', 'Pumpkin', 'Sunflower', 'Time']
    print(tabulate(stats_data, headers=stats_headers, tablefmt='grid'))


def export_to_csv(results, output_file='results.csv'):
    """Export results to CSV file."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'pumpkin', 'sunflower', 'total', 'time'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results exported to: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch analyze images with fast grid detection')
    parser.add_argument('directory', nargs='?', default='.', help='Directory containing images (default: current directory)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save annotated images')
    parser.add_argument('--csv', type=str, help='Export results to CSV file')

    args = parser.parse_args()

    # Analyze images
    results = batch_analyze_directory(args.directory, save_annotated=not args.no_save)

    if results:
        # Print summary table
        print_summary_table(results)

        # Export to CSV if requested
        if args.csv:
            export_to_csv(results, args.csv)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
