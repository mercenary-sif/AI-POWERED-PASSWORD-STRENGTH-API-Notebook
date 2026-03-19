from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.schemas.password_schema import PasswordRequest, PasswordResponse
from app.model.model_loader import get_model, get_vocab
from app.model.predictor import predict
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
# load model + vocab once at startup 
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()    
    get_vocab()   
    print(" Model and vocabulary loaded.")
    yield
    print(" Shutting down.")

app = FastAPI(
    title="Password Strength API",
    description="BiLSTM character-level password strength classifier trained on ~850K passwords.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "BiLSTM", "max_len": 30}

@app.post("/analyze", response_model=PasswordResponse)
def analyze_password(body: PasswordRequest):
    try:
        result = predict(body.password)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": True, "message": str(e), "code": "SERVER_ERROR"}
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    msg = first.get("msg", "Invalid input.")
   
    msg = msg.replace("Value error, ", "")
    return JSONResponse(
        status_code=422,
        content={"error": True, "message": msg, "code": "VALIDATION_ERROR"},
    )