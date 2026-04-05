import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    """Launch Kejafi full stack: FastAPI + Stage 1 + Stage 2"""
    
    # Track processes for cleanup
    processes = []
    
    try:
        print("🚀 Starting Kejafi Full Stack...\n")
        
        # 1. FastAPI Backend
        print("📡 Starting FastAPI on http://127.0.0.1:8000")
        api_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
            cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        processes.append(("API", api_proc))
        time.sleep(3)
        
        # 2. Stage 1 (Research Engine)
        print("🔬 Starting Stage 1 on http://localhost:8501")
        s1_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "research_engine.py", "--server.port", "8501"],
            cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        processes.append(("Stage 1", s1_proc))
        time.sleep(2)
        
        # 3. Stage 2 (Tokenization)
        print("🏠 Starting Stage 2 on http://localhost:8506")
        s2_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "tokenization_platform.py", "--server.port", "8506"],
            cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        processes.append(("Stage 2", s2_proc))
        
        print("\n" + "="*50)
        print("✅ All services running!")
        print("="*50)
        print("📡 API:     http://127.0.0.1:8000/docs")
        print("🔬 Stage 1: http://localhost:8501")
        print("🏠 Stage 2: http://localhost:8506")
        print("="*50)
        print("\nPress Ctrl+C to stop all services")
        
        # Keep main process alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        for name, proc in processes:
            print(f"  Stopping {name}...")
            proc.terminate()
            proc.wait()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()