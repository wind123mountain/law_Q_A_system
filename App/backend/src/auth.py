import httpx
from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, JWT_SECRET, REDIRECT_URI
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from jose import jwt
from models import users

router = APIRouter()


@router.get("/auth/login")
def login():
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"response_type=code&"
        f"scope=openid%20email%20profile&"
        f"redirect_uri={REDIRECT_URI}"
    )


@router.get("/auth/callback")
async def callback(request: Request):
    code = request.query_params["code"]
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(token_url, data=data)
        token = r.json()
        id_token = token.get("id_token")
        access_token = token["access_token"]
        if not id_token:
            raise HTTPException(400, "Không nhận được id_token từ Google")

        payload = jwt.decode(
            id_token,
            key=None,
            options={"verify_signature": False},
            access_token=access_token,
            audience=GOOGLE_CLIENT_ID,
        )
        email = payload["email"]

        if not users.find_one({"email": email}):
            users.insert_one({"email": email, "name": payload["name"]})

        jwt_token = jwt.encode({"email": email}, JWT_SECRET, algorithm="HS256")
        resp = RedirectResponse(url="http://localhost:8000")
        resp.set_cookie(
            key="token",
            value=jwt_token,
            httponly=False,
        )
        return resp
