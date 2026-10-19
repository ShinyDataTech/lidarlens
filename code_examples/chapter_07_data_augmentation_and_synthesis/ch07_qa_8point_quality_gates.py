"""
Chapter 7: 3D Data Augmentation & Synthesis Quality Gates
Code Example 7.3: Hardware-Level 8-Point Quality Gates for 3D Synthesis Validation.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第7章 7.3节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.augment.qa import run_8point_quality_gates


def generate_synthesis_candidate(inject_flaws: bool = False) -> dict:
    """Generate synthesized LiDAR scene points and bounding boxes for QA auditing."""
    num_pts = 20_000
    points = np.zeros((num_pts, 4), dtype=np.float32)
    # Natural 1/R^2 distribution
    r = np.random.exponential(scale=15.0, size=num_pts) + 2.0
    r = np.clip(r, 2.0, 70.0)
    theta = np.random.uniform(-np.pi/3, np.pi/3, num_pts) # 120-degree front FOV
    
    points[:, 0] = r * np.cos(theta)
    points[:, 1] = r * np.sin(theta)
    points[:, 2] = -1.6 + np.random.uniform(0.0, 3.0, num_pts)
    points[:, 3] = np.random.beta(a=2.0, b=5.0, size=num_pts).astype(np.float32)
    
    if inject_flaws:
        # Inject colliding boxes
        gt_boxes = np.array([
            [10.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0],
            [10.5, 0.2, 0.0, 4.0, 1.8, 1.5, 0.0], # Severe overlap collision!
        ], dtype=np.float32)
    else:
        # Well-separated valid boxes
        gt_boxes = np.array([
            [12.0, -2.5, -0.8, 4.2, 1.8, 1.5, 0.1],
            [25.0,  3.5, -0.6, 1.9, 0.8, 1.4, -0.2],
            [8.0,   1.2, -0.7, 0.6, 0.6, 1.7, 0.0]
        ], dtype=np.float32)
        
    return {
        "points": points,
        "gt_boxes": gt_boxes,
        "gt_names": ["Car", "Cyclist", "Pedestrian"][:len(gt_boxes)],
        "frame_id": "synth_00128"
    }


def main():
    print("=== Chapter 7: Hardware-Level 8-Point Synthesis Quality Gates ===")
    
    # 1. Test clean, physically-consistent synthesis sample
    clean_sample = generate_synthesis_candidate(inject_flaws=False)
    t0 = time.perf_counter()
    report_clean = run_8point_quality_gates(clean_sample)
    qa_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"Quality Gates Evaluation completed in {qa_ms:.2f} ms:")
    print("\n" + "=" * 55)
    print("         8-Point Hardware-Level QA Gate Checklist")
    print("=" * 55)
    for gate_name, status in report_clean.get("checks", {}).items():
        verdict = "PASSED" if status.get("passed", False) else "FAILED"
        print(f"  Gate [{gate_name:<24}]: {verdict} ({status.get('details', '')})")
    print("=" * 55)
    print(f"Overall Quality Verdict: {'ACCEPT (Clean)' if report_clean['all_passed'] else 'REJECT'}")
    
    assert report_clean["all_passed"], "Clean synthetic scene unexpectedly failed quality gates!"
    
    # 2. Test flawed synthesis sample (Bounding Box Collision)
    flawed_sample = generate_synthesis_candidate(inject_flaws=True)
    report_flawed = run_8point_quality_gates(flawed_sample)
    print(f"\nFlawed Scene Overall Verdict: {'ACCEPT' if report_flawed['all_passed'] else 'REJECT (Collision Caught)'}")
    assert not report_flawed["all_passed"], "Quality gates failed to catch severe bounding box collision!"

    print("Chapter 7.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
