"""
Global Code Examples Test & Benchmark Suite.
Discovers, executes, and audits 100% of textbook code examples across Chapters 1 through 11.

Textbook:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def run_example(script_path: Path) -> Tuple[bool, float, str]:
    """Execute a single chapter example script in a subprocess and return status."""
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    repo_root = str(script_path.parent.parent.parent)
    curr_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{curr_pythonpath}" if curr_pythonpath else repo_root
    
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    elapsed = time.perf_counter() - t0
    success = (proc.returncode == 0)
    output = proc.stdout if success else (proc.stdout + "\n" + proc.stderr)
    return success, elapsed, output


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    root_dir = Path(__file__).resolve().parent
    chapter_dirs = sorted([d for d in root_dir.glob("chapter_*") if d.is_dir()])
    
    print("=" * 95)
    print("      LidarLens Textbook Code Examples Master Verification Runner (Chapters 1 - 11)")
    print("=" * 95)
    print(f"Scanning Directory: {root_dir}")
    print(f"Found {len(chapter_dirs)} Chapter directories.\n")

    results: List[Dict] = []
    total_examples = 0
    passed_examples = 0
    failed_examples = 0
    start_total_time = time.perf_counter()

    for chap_dir in chapter_dirs:
        chap_name = chap_dir.name
        scripts = sorted([f for f in chap_dir.glob("ch*.py") if f.is_file()])
        if not scripts:
            continue

        print(f"[{chap_name}] ({len(scripts)} examples):")
        for script in scripts:
            total_examples += 1
            success, elapsed_s, log_output = run_example(script)
            status_str = "PASS" if success else "FAIL"
            
            if success:
                passed_examples += 1
                print(f"  [PASS] {script.name:<46} ({elapsed_s * 1000.0:>7.2f} ms)")
            else:
                failed_examples += 1
                print(f"  [FAIL] {script.name:<46} ({elapsed_s * 1000.0:>7.2f} ms)")
                print("-" * 60)
                print(f"Traceback / Error Log:\n{log_output.strip()}")
                print("-" * 60)

            results.append({
                "chapter": chap_name,
                "script": script.name,
                "success": success,
                "elapsed_s": elapsed_s,
                "output": log_output
            })
        print()

    total_time_s = time.perf_counter() - start_total_time

    # Summary Audit Table
    print("=" * 95)
    print("                              FINAL VERIFICATION REPORT")
    print("=" * 95)
    print(f"{'Chapter':<32} {'Script':<42} {'Status':<8} {'Latency'}")
    print("-" * 95)
    for res in results:
        status_tag = "PASS" if res["success"] else "FAIL"
        print(f"{res['chapter']:<32} {res['script']:<42} {status_tag:<8} {res['elapsed_s'] * 1000.0:>7.2f} ms")

    print("=" * 95)
    print(f"Total Examples Executed: {total_examples}")
    print(f"Total Passed:            {passed_examples} / {total_examples} ({passed_examples / max(total_examples, 1) * 100:.1f}%)")
    print(f"Total Failed:            {failed_examples}")
    print(f"Total Audit Wall Time:   {total_time_s:.2f} s")
    print("=" * 95)

    if failed_examples > 0:
        print(f"\n[ERROR] {failed_examples} example(s) failed verification. Please review tracebacks above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] 100% of all textbook code examples executed and passed all assertions!")
        sys.exit(0)


if __name__ == "__main__":
    main()
