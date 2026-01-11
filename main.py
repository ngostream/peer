from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from camera import VideoCamera
import time

app = FastAPI(title="PEER // Visual Accountability")
templates = Jinja2Templates(directory="templates")

camera = VideoCamera()

# mock user for demo
MOCK_USER = {
    "user_id": "gaucho_001",
    "name": "Nathan Ngo",
    "email": "nathan_ngo@ucsb.edu",
}

async def get_current_user(request: Request):
    if request.cookies.get("access_token") == "mock_valid_token":
        return MOCK_USER
    return None

async def require_auth(request: Request):
    user = await get_current_user(request)
    if not user: raise HTTPException(401)
    return user

@app.get("/auth/login")
async def login():
    return RedirectResponse("/auth/callback")

@app.get("/auth/callback")
async def callback():
    res = RedirectResponse("/")
    res.set_cookie("access_token", "mock_valid_token")
    return res

@app.get("/auth/logout")
async def logout():
    res = RedirectResponse("/")
    res.delete_cookie("access_token")
    return res

@app.get("/api/user/me")
async def me(request: Request):
    user = await get_current_user(request)
    return {"authenticated": True, "user": user} if user else {"authenticated": False}

def generate_frames():
    # stream video frames
    while True:
        frame_bytes, _, _ = camera.get_frame()
        if frame_bytes:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else: break
        time.sleep(0.033)

@app.get("/video_feed")
async def feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def stats(user: dict = Depends(require_auth)):
    _, status, score = camera.get_frame()
    return {"focus_score": int(score), "status": status}

@app.post("/session/start")
async def start(user: dict = Depends(require_auth)):
    return {"status": "started"}

@app.post("/session/stop")
async def stop(user: dict = Depends(require_auth)):
    return {"status": "stopped"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    