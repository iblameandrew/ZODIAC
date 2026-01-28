from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import json
from simulation import zodiac_simulation, SimParams, state

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulation_task = None

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

async def run_async_generator(gen):
    try:
        async for _ in gen:
            pass
    except Exception as e:
        print(f"Generator Error: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/start")
async def start_simulation(params: SimParams):
    global simulation_task

    if state.running:
        return {"status": "error", "message": "Simulation already running"}

    # We run the generator in the background to drive the state machine
    # The actual frames are consumed by the /api/stream endpoint reading from state.frame_buffer
    # OR we can have the stream endpoint be the driver? 
    # Current architecture: `zodiac_simulation` yields frames. 
    # Ideally, we start a task that iterates the generator and populates the buffer.
    
    # Let's use the existing pattern: 
    # Create a task that runs the generator. The generator Updates `state` object.
    # The /stream endpoint reads from `state`.
    
    # Wait, `zodiac_simulation` yields frames but also updates `state.frame_buffer`.
    # So we can just consume it in a background task.
    
    simulation_task = asyncio.create_task(run_async_generator(zodiac_simulation(params)))
    return {"status": "running", "message": "Simulation started"}

@app.post("/api/stop")
async def stop_simulation():
    global simulation_task
    state.running = False
    if simulation_task:
        simulation_task.cancel()
    return {"status": "stopped", "message": "Simulation stopped"}

@app.get("/api/stream")
async def stream_frames():
    async def event_stream():
        last_index = 0
        while True:
            # Check frame buffer in the shared state
            if state.frame_buffer and last_index < len(state.frame_buffer):
                for i in range(last_index, len(state.frame_buffer)):
                    yield f"data: {json.dumps(state.frame_buffer[i])}\n\n"
                last_index = len(state.frame_buffer)
            
            # Check completion
            if not state.running and last_index >= len(state.frame_buffer):
                yield f"data: {json.dumps({'event': 'complete'})}\n\n"
                break
            
            await asyncio.sleep(0.05)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
