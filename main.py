import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Executing: {script_path}")
    print(f"{'='*50}")
    
    # Use sys.executable to ensure we use the same python environment
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_path}")
        sys.exit(1)

def main():
    # Set working directory to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # 1. Preprocessing
    run_script(os.path.join('src', 'preprocessing.py'))

    # 2. Training
    run_script(os.path.join('src', 'train.py'))

    # 3. Evaluation
    run_script(os.path.join('src', 'evaluate.py'))

    print("\n" + "="*50)
    print("CARDIO RISK NET PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print("You can find the trained model in 'models/' and visualizations in 'reports/'.")

if __name__ == "__main__":
    main()
