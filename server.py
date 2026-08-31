import asyncio
import websockets
import json
import random

clients = {}  # websocket -> dict(role, points)
prize_pool = 0

async def broadcast(message):
    if clients:
        await asyncio.gather(*[ws.send(json.dumps(message)) for ws in clients])

async def handler(websocket, path=None):
    global prize_pool
    # Initialize client
    clients[websocket] = {"role": "active", "points": 0}
    try:
        # Send initial state
        await websocket.send(json.dumps({"type": "state", "prize_pool": prize_pool, "role": "active", "points": 0}))
        
        async for message in websocket:
            data = json.loads(message)
            event_type = data.get("type")
            
            if event_type == "player_died":
                # Update this client's role to spectator and give points
                clients[websocket]["role"] = "spectator"
                clients[websocket]["points"] += 50
                # Increase global prize pool
                prize_pool += 10000000
                
                await websocket.send(json.dumps({
                    "type": "state",
                    "role": "spectator",
                    "points": clients[websocket]["points"]
                }))
                await broadcast({"type": "prize_pool", "amount": prize_pool})
                
            elif event_type == "glitch":
                # Ensure they are spectator and have points
                if clients[websocket]["role"] == "spectator" and clients[websocket]["points"] >= 25:
                    clients[websocket]["points"] -= 25
                    await websocket.send(json.dumps({"type": "state", "points": clients[websocket]["points"], "role": "spectator"}))
                    
                    # Find a random active player to flashbang
                    actives = [ws for ws, info in clients.items() if info["role"] == "active"]
                    if actives:
                        target = random.choice(actives)
                        await target.send(json.dumps({"type": "flashbang"}))
                        
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        del clients[websocket]

async def main():
    print("Spectator Interference Server starting on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
