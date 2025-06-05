import os
import sys
import subprocess
from datetime import datetime

def check_dependencies():
    try:
        import grpc_tools
        return True
    except ImportError:
        print("grpc_tools not found. Please install with: pip install grpcio-tools")
        return False

def regenerate_grpc():
    """Regenerate gRPC files from proto definition with robust logging."""
    proto_file = "model.proto"
    out_dir = "."
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting gRPC code regeneration...")

    if not os.path.exists(proto_file):
        print(f"[{timestamp}] ERROR: {proto_file} not found!")
        return False

    if not check_dependencies():
        print(f"[{timestamp}] ERROR: grpcio-tools not installed.")
        return False

    cmd = [
        sys.executable, '-m', 'grpc_tools.protoc',
        f'-I.',
        f'--python_out={out_dir}',
        f'--grpc_python_out={out_dir}',
        proto_file
    ]
    print(f"[{timestamp}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[{timestamp}] gRPC code generated successfully.")
        print(f"[{timestamp}] Updated files: model_pb2.py, model_pb2_grpc.py")
        return True
    else:
        print(f"[{timestamp}] ERROR: gRPC code generation failed!")
        print(result.stderr)
        return False

if __name__ == "__main__":
    print("=== gRPC File Regeneration Tool ===")
    success = regenerate_grpc()
    if not success:
        sys.exit(1) 