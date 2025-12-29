# backend/app/__init__.py

from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

from config import Config

db = SQLAlchemy()
jwt = JWTManager()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Init extensions
    db.init_app(app)
    jwt.init_app(app)

    # CORS: allow your RN app (and others) to call /api/*
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # -----------------------------
    # JWT error handlers
    # -----------------------------
    @jwt.unauthorized_loader
    def unauthorized_callback(reason):
        return (
            jsonify(
                {
                    "message": "Missing or invalid auth token",
                    "error": reason,
                }
            ),
            401,
        )

    @jwt.invalid_token_loader
    def invalid_token_callback(reason):
        return (
            jsonify(
                {
                    "message": "Invalid auth token",
                    "error": reason,
                }
            ),
            422,
        )

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({"message": "Token has expired"}), 401

    # -----------------------------
    # IMPORT BLUEPRINTS (all routes)
    # -----------------------------
    from .routes.auth_routes import auth_bp
    from .routes.profile_routes import profile_bp
    from .routes.dashboard_routes import dashboard_bp
    from .routes.social_routes import social_bp
    from .routes.rewards_routes import rewards_bp
    from .routes.exercise_routes import exercises_bp

    # ✅ workouts routes live here now
    from .routes.workout_routes import workouts_bp



    # -----------------------------
    # REGISTER BLUEPRINTS
    # -----------------------------
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(profile_bp, url_prefix="/api/profile")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(workouts_bp, url_prefix="/api/workouts")
    app.register_blueprint(social_bp, url_prefix="/api/social")
    app.register_blueprint(rewards_bp, url_prefix="/api/rewards")
    app.register_blueprint(exercises_bp, url_prefix="/api/exercises")



    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    # -----------------------------
    # DB init
    # -----------------------------
    with app.app_context(): 
        db.create_all()

    return app
