import os
import sys
import subprocess
import pkg_resources

def check_dependencies():
    """Check if required packages are installed with correct versions"""
    required_packages = {
        'grpcio-tools': '>=1.41.0',
        'protobuf': '>=3.19.0'
    }
    
    print("Checking dependencies...")
    for package, version in required_packages.items():
        try:
            pkg_resources.require(f"{package}{version}")
            installed_version = pkg_resources.get_distribution(package).version
            print(f"✓ {package} version {installed_version} is installed")
        except pkg_resources.VersionConflict as e:
            print(f"✗ {package} version conflict: {e}")
            return False
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package} is not installed")
            return False
    return True

def regenerate_grpc():
    """Regenerate gRPC files from proto definition"""
    
    # First check dependencies
    if not check_dependencies():
        print("\nPlease install required packages:")
        print("pip install grpcio-tools>=1.41.0 protobuf>=3.19.0")
        return False

    # Updated proto content with comments and validation
    proto_content = """// model.proto
syntax = "proto3";

package fl;

// Service definition for Federated Learning peer communication
service FLPeer {
  // SendModel: Transfer model weights between peers
  rpc SendModel (ModelWeights) returns (Ack);
  
  // HealthCheck: Monitor peer availability and network status
  rpc HealthCheck (HealthCheckRequest) returns (HealthCheckResponse);
}

// ModelWeights: Contains serialized model parameters
message ModelWeights {
  bytes weights = 1;  // Serialized PyTorch state dict
}

// Acknowledgment message for operations
message Ack {
  string message = 1;  // Status or error message
}

// HealthCheck request message
message HealthCheckRequest {
  string peer_id = 1;    // ID of the requesting peer
  string timestamp = 2;  // ISO format timestamp of request
}

// HealthCheck response message
message HealthCheckResponse {
  string status = 1;     // OK, ERROR, etc.
  string peer_id = 2;    // ID of the responding peer
  string timestamp = 3;  // ISO format timestamp of response
}
"""
    
    try:
        print("\n1. Writing updated proto file...")
        with open('model.proto', 'w') as f:
            f.write(proto_content)
        print("✓ Proto file updated successfully")
        
        print("\n2. Removing old generated files...")
        files_to_remove = ['model_pb2.py', 'model_pb2_grpc.py']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                print(f"✓ Removed {file}")
        
        print("\n3. Generating new gRPC files...")
        cmd = [sys.executable, '-m', 'grpc_tools.protoc', '-I.', '--python_out=.', '--grpc_python_out=.', 'model.proto']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ gRPC files generated successfully")
            
            # Verify the generated files
            print("\n4. Verifying generated files...")
            required_files = ['model_pb2.py', 'model_pb2_grpc.py']
            all_files_exist = True
            for file in required_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file)
                    print(f"✓ {file} generated (size: {file_size} bytes)")
                else:
                    print(f"✗ {file} is missing!")
                    all_files_exist = False
            
            if all_files_exist:
                print("\nSuccess! All files have been regenerated correctly.")
                print("\nNext steps:")
                print("1. Run the test connection:")
                print("   python test_connection.py")
                print("2. If successful, start your node:")
                print("   python start_node.py")
                return True
            else:
                print("\nError: Some files are missing. Please check the error messages above.")
                return False
                
        else:
            print("✗ Error generating gRPC files:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n✗ Error during regeneration: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== gRPC File Regeneration Tool ===")
    success = regenerate_grpc()
    if not success:
        sys.exit(1) 