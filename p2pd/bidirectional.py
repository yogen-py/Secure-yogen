import asyncio
from p2pd import P2PNode
import time

# --- Public STUN Servers ---
PUBLIC_STUN_SERVERS = [
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun2.l.google.com:19302",
    "stun3.l.google.com:19302",
    "stun4.l.google.com:19302",
]

async def chat_handler(pipe, node_name):
    """
    Handles continuous sending and receiving on an established pipe.
    """
    peer_name = pipe.peer_address.nickname if pipe.peer_address.nickname else str(pipe.peer_address)
    print(f"[{node_name}] Connected to {peer_name}. Starting chat...")

    try:
        message_counter = 0
        while True:
            # Send a message
            send_message = f"[{node_name}] Ping {message_counter}".encode()
            print(f"[{node_name}] Sending: {send_message.decode()}")
            await pipe.send(send_message)

            # Wait for a reply
            print(f"[{node_name}] Waiting for reply...")
            received_data = await pipe.recv()
            if received_data:
                print(f"[{node_name}] Received: {received_data.decode()}")
            else:
                print(f"[{node_name}] Received empty message or connection closed.")
                break # Peer disconnected

            message_counter += 1
            await asyncio.sleep(2) # Wait a bit before sending next message

    except asyncio.CancelledError:
        print(f"[{node_name}] Chat for pipe to {peer_name} cancelled.")
    except Exception as e:
        print(f"[{node_name}] Error in chat with {peer_name}: {e}")
    finally:
        print(f"[{node_name}] Chat session with {peer_name} ended.")
        await pipe.close()


async def message_callback(msg, client_tup, pipe):
    """
    This callback is for handling incoming messages received by the P2PNode.
    It demonstrates how a node can react to messages and send replies.
    """
    node_name = pipe.node.nicknames[0] if pipe.node.nicknames else "Unknown Node"
    peer_address_str = client_tup[1] # IP address of the sender

    print(f"[{node_name}] Received message from {peer_address_str}: {msg.decode()}")

    # Example of a bidirectional response
    if b"ping" in msg.lower():
        reply_msg = f"[{node_name}] Pong! From {node_name}".encode()
        await pipe.send(reply_msg, client_tup) # Send reply back to the sender
        print(f"[{node_name}] Replied: {reply_msg.decode()}")
    elif b"hello" in msg.lower():
        reply_msg = f"[{node_name}] Hi there! From {node_name}".encode()
        await pipe.send(reply_msg, client_tup)
        print(f"[{node_name}] Replied: {reply_msg.decode()}")


async def run_node(my_nickname, peer_nickname=None):
    """
    Generic function to run a P2P node.
    If peer_nickname is provided, it will try to connect to that peer.
    """
    node = await P2PNode(
        nicknames=[my_nickname],
        stuns=PUBLIC_STUN_SERVERS
    )
    node.add_msg_cb(message_callback) # This handles incoming messages from any connected peer

    print(f"Node '{my_nickname}' started with address: {node.address}")

    active_tasks = []

    # If this node should initiate a connection
    if peer_nickname:
        print(f"[{my_nickname}] Attempting to connect to '{peer_nickname}'...")
        try:
            pipe = await node.connect(peer_nickname)
            print(f"[{my_nickname}] Successfully connected to {peer_nickname}.")
            # Start a chat task for this initiated connection
            task = asyncio.create_task(chat_handler(pipe, my_nickname))
            active_tasks.append(task)
        except Exception as e:
            print(f"[{my_nickname}] Could not connect to {peer_nickname}: {e}")

    # Keep the node running indefinitely to listen for incoming connections and messages
    # and to manage its own active connections
    print(f"[{my_nickname}] Node running, waiting for connections and messages...")
    try:
        # Wait for all active chat tasks to complete, or indefinitely if no active chats
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        else:
            while True:
                await asyncio.sleep(1) # Keep the event loop running
    except asyncio.CancelledError:
        print(f"[{my_nickname}] Node stopped.")
    except KeyboardInterrupt:
        print(f"[{my_nickname}] Node interrupted by user.")
    finally:
        print(f"[{my_nickname}] Closing node...")
        await node.close()
        print(f"[{my_nickname}] Node closed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'laptopA':
            # Laptop A will be the one listening, but can also send if connected
            asyncio.run(run_node('laptop.a.node'))
        elif sys.argv[1] == 'laptopB':
            # Laptop B will connect to Laptop A and start chatting
            asyncio.run(run_node('laptop.b.node', 'laptop.a.node'))
        else:
            print("Usage: python p2pd_experiment_bidirectional.py [laptopA|laptopB]")
            sys.exit(1)
    else:
        print("Usage: python p2pd_experiment_bidirectional.py [laptopA|laptopB]")
        sys.exit(1)