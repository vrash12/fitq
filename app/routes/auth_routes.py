# backend/app/routes/auth_routes.py

import base64
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import face_recognition
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
from sqlalchemy import or_

from .. import db
from ..models.user import User

auth_bp = Blueprint("auth", __name__)

# IMPORTANT: helps you confirm which file Flask is actually loading
print("[auth_routes] LOADED FROM:", __file__)

# -----------------------------
# Face storage (stable path)
# -----------------------------
def _faces_dir() -> str:
    base = getattr(current_app, "instance_path", None) or os.getcwd()
    folder = os.path.join(base, "users_faces")
    os.makedirs(folder, exist_ok=True)
    return folder


def _strip_data_url_prefix(image_base64: str) -> str:
    if "," in image_base64:
        return image_base64.split(",", 1)[1]
    return image_base64


def _decode_base64_image(image_base64: str) -> Optional[np.ndarray]:
    try:
        b64 = _strip_data_url_prefix(image_base64)
        image_data = base64.b64decode(b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception:
        return None


def save_reference_face(user_id: int, image_base64: str) -> Tuple[bool, str]:
    """
    Decode base64 image, detect a single face, save reference jpg + encoding .npy.
    Returns (success, message).
    """
    try:
        img_bgr = _decode_base64_image(image_base64)
        if img_bgr is None:
            return False, "Invalid image format"

        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img, model="hog")
        if not face_locations:
            return False, "No face detected. Please ensure good lighting."
        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please be alone in the photo."

        encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if not encodings:
            return False, "Face detected but encoding failed. Try again with clearer lighting."
        encoding = encodings[0]

        folder = _faces_dir()
        jpg_path = os.path.join(folder, f"{user_id}_reference.jpg")
        enc_path = os.path.join(folder, f"{user_id}_encoding.npy")

        cv2.imwrite(jpg_path, img_bgr)
        np.save(enc_path, encoding)

        return True, "Face registered successfully"

    except Exception as e:
        current_app.logger.exception(f"Face Save Error: {e}")
        return False, "Server error processing image"


def _load_reference_encoding(user_id: int) -> Optional[np.ndarray]:
    path = os.path.join(_faces_dir(), f"{user_id}_encoding.npy")
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


# -----------------------------
# Routes
# -----------------------------
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""  # do NOT strip passwords
    image_base64 = data.get("image_base64")  # OPTIONAL for now

    if not email or not username or not password:
        return jsonify({"message": "email, username and password are required"}), 400

    if len(password) < 6:
        return jsonify({"message": "password must be at least 6 characters"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "email already in use"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "username already in use"}), 400

    user = User(email=email, username=username, display_name=username)
    user.set_password(password)

    try:
        db.session.add(user)
        db.session.flush()  # get user.id before commit

        # If selfie provided, save it; if it fails, rollback (you can change this behaviour)
        if image_base64:
            ok, msg = save_reference_face(int(user.id), image_base64)
            if not ok:
                db.session.rollback()
                return jsonify({"message": msg}), 400

        db.session.commit()

        access_token = create_access_token(identity=str(user.id))
        return jsonify({"token": access_token, "user": user.to_dict()}), 201

    except Exception as e:
        db.session.rollback()
        current_app.logger.exception(f"Registration Error: {e}")
        return jsonify({"message": "Internal server error"}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Accepts:
      - { "email": "...", "password": "..." }
      - { "username": "...", "password": "..." }
      - { "identifier": "...", "password": "..." }  # email or username
    """
    data = request.get_json(silent=True) or {}

    identifier = (data.get("identifier") or data.get("email") or data.get("username") or "").strip()
    password = data.get("password") or ""  # do NOT strip passwords

    # Debug the incoming payload (without printing the password)
    current_app.logger.info(f"[auth/login] identifier='{identifier}' keys={list(data.keys())}")

    if not identifier or not password:
        return jsonify({"message": "identifier and password are required"}), 400

    # Try match email (lowercased) OR username
    user = User.query.filter(
        or_(
            User.email == identifier.lower(),
            User.username == identifier,
        )
    ).first()

    if not user:
        current_app.logger.info(f"[auth/login] user NOT found for '{identifier}'")
        return jsonify({"message": "invalid credentials"}), 401

    if not user.check_password(password):
        current_app.logger.info(f"[auth/login] bad password for user_id={user.id} identifier='{identifier}'")
        return jsonify({"message": "invalid credentials"}), 401

    access_token = create_access_token(identity=str(user.id))
    return jsonify({"token": access_token, "user": user.to_dict()}), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"message": "user not found"}), 404
    return jsonify({"user": user.to_dict()}), 200


@auth_bp.route("/face/verify", methods=["POST"])
@jwt_required()
def verify_face():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"match": False, "distance": None, "message": "user not found"}), 404

    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image_base64")
    if not image_base64:
        return jsonify({"match": False, "distance": None, "message": "image_base64 is required"}), 400

    ref_encoding = _load_reference_encoding(int(user.id))
    if ref_encoding is None:
        return jsonify({"match": False, "distance": None, "message": "no reference face on file"}), 400

    img_bgr = _decode_base64_image(image_base64)
    if img_bgr is None:
        return jsonify({"match": False, "distance": None, "message": "invalid image format"}), 400

    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img, model="hog")
    if not face_locations:
        return jsonify({"match": False, "distance": None, "message": "no face detected"}), 400
    if len(face_locations) > 1:
        return jsonify({"match": False, "distance": None, "message": "multiple faces detected"}), 400

    encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if not encodings:
        return jsonify({"match": False, "distance": None, "message": "face encoding failed"}), 400

    live_encoding = encodings[0]
    distance = float(face_recognition.face_distance([ref_encoding], live_encoding)[0])

    threshold = float(current_app.config.get("FACE_MATCH_THRESHOLD", 0.55))
    match = distance <= threshold

    return jsonify(
        {
            "match": bool(match),
            "distance": distance,
            "threshold": threshold,
            "message": "match" if match else "no match",
        }
    ), 200
