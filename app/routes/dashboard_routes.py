#backend/app/routes/dashboard_routes.py
from datetime import date, datetime, timedelta

import base64
import cv2
import numpy as np
import mediapipe as mp
import math

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from .. import db
from ..models.user import User
from ..models.workout import Workout
from ..models.user_daily_stats import UserDailyStats

# Blueprint for dashboard-related endpoints
dashboard_bp = Blueprint("dashboard", __name__)

# Blueprint for workout-related endpoints
workout_bp = Blueprint("workouts", __name__)

# -------------------------
# Mediapipe Pose setup
# -------------------------
mp_pose = mp.solutions.pose
_pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------------
# PUSHUP REP LOGIC CONFIG
# -------------------------

THRESH_UP = 150          # elbow angle ~top position
THRESH_DOWN = 110        # elbow angle ~bottom/depth
SMOOTH_FACTOR = 0.5
INCLINATION_THRESH = 45.0  # degrees; > this => standing / vertical

# In-memory store of rep counters per session:
# key: (user_id, workout_id, exercise_id) -> SmartRepCounter
_rep_sessions = {}


class SmartRepCounter:
    """
    Pushup rep counter + simple coaching, adapted from your training script.
    """

    def __init__(self, up_thresh=150, down_thresh=110, smooth_factor=0.5):
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.alpha = smooth_factor

        self.state = "UP"  # "UP" or "DOWN"
        self.count = 0
        self.feedback = "Get Ready"
        self.feedback_color = (200, 200, 200)
        self.mode = "STANDING"  # "STANDING" or "PUSHUP"

        self.avg_angle = 180.0
        self.in_rep_motion = False

    def update(self, left_elbow, right_elbow, body_inclination):
        """
        left_elbow, right_elbow: raw elbow angles (degrees)
        body_inclination: 0=horizontal plank, 90=standing
        """

        # 1. Check Body Orientation
        if body_inclination > INCLINATION_THRESH:
            # Standing / vertical -> pause counting
            self.mode = "STANDING"
            self.feedback = "PAUSED (Stand)"
            self.feedback_color = (200, 200, 200)
            return (
                self.state,
                self.count,
                self.avg_angle,
                self.feedback,
                self.feedback_color,
                self.mode,
            )
        else:
            self.mode = "PUSHUP"

        # 2. Smooth Signal
        raw_avg = (left_elbow + right_elbow) / 2.0
        self.avg_angle = (self.alpha * raw_avg) + ((1 - self.alpha) * self.avg_angle)
        curr = self.avg_angle

        # 3. State Machine
        if self.state == "UP":
            # detect descent
            if curr < (self.up_thresh - 10):
                self.in_rep_motion = True

            # depth hit
            if curr < self.down_thresh:
                self.state = "DOWN"
                self.feedback = "DEPTH HIT!"
                self.feedback_color = (0, 255, 0)

            # went down a bit then back up without depth
            elif self.in_rep_motion and curr > self.up_thresh:
                self.feedback = "GO DEEPER!"
                self.feedback_color = (0, 0, 255)
                self.in_rep_motion = False

        elif self.state == "DOWN":
            # completion at top
            if curr > self.up_thresh:
                self.state = "UP"
                self.count += 1
                self.feedback = "GOOD REP!"
                self.feedback_color = (0, 255, 0)
                self.in_rep_motion = False

            # stuck in the middle
            elif curr > (self.down_thresh + 20) and curr < self.up_thresh:
                self.feedback = "LOCK ELBOWS!"
                self.feedback_color = (0, 255, 255)

        return (
            self.state,
            self.count,
            curr,
            self.feedback,
            self.feedback_color,
            self.mode,
        )


def _get_pushup_session_counter(user_id: int, workout_id: int, exercise_id: str):
    """
    Returns a SmartRepCounter for this user/workout/exercise.
    Creates a new one if needed.
    """
    key = (user_id, workout_id, exercise_id)
    counter = _rep_sessions.get(key)
    if counter is None:
        counter = SmartRepCounter(
            up_thresh=THRESH_UP, down_thresh=THRESH_DOWN, smooth_factor=SMOOTH_FACTOR
        )
        _rep_sessions[key] = counter
    return counter


def _angle_3pts(a, b, c):
    """
    Compute angle ABC (in degrees) with points a,b,c as (x, y).
    a, b, c are 2D tuples in normalized image coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom < 1e-6:
        return None
    cos_val = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _compute_body_inclination(shoulder_xy, ankle_xy):
    """
    0 deg = horizontal (plank), 90 deg = vertical (standing).
    Uses 2D normalized coordinates (x,y).
    """
    dx = abs(shoulder_xy[0] - ankle_xy[0])
    dy = abs(shoulder_xy[1] - ankle_xy[1])
    if dx < 1e-6:
        return 90.0
    return math.degrees(math.atan(dy / dx))


def _extract_pushup_features_from_landmarks(landmarks):
    """
    Extracts:
      - left_elbow_angle
      - right_elbow_angle
      - avg body inclination (deg)
    from Mediapipe pose landmarks list.
    Returns None if elbows aren't visible enough.
    """

    PL = mp_pose.PoseLandmark

    def pt(idx):
        lm = landmarks[idx.value]
        return (lm.x, lm.y), lm.visibility

    # Key points
    (ls_xy, ls_vis) = pt(PL.LEFT_SHOULDER)
    (le_xy, le_vis) = pt(PL.LEFT_ELBOW)
    (lw_xy, lw_vis) = pt(PL.LEFT_WRIST)

    (rs_xy, rs_vis) = pt(PL.RIGHT_SHOULDER)
    (re_xy, re_vis) = pt(PL.RIGHT_ELBOW)
    (rw_xy, rw_vis) = pt(PL.RIGHT_WRIST)

    (la_xy, la_vis) = pt(PL.LEFT_ANKLE)
    (ra_xy, ra_vis) = pt(PL.RIGHT_ANKLE)

    # Elbow angles (only if joints are reasonably visible)
    left_elbow_angle = (
        _angle_3pts(ls_xy, le_xy, lw_xy)
        if ls_vis > 0.4 and le_vis > 0.4 and lw_vis > 0.4
        else None
    )
    right_elbow_angle = (
        _angle_3pts(rs_xy, re_xy, rw_xy)
        if rs_vis > 0.4 and re_vis > 0.4 and rw_vis > 0.4
        else None
    )

    if left_elbow_angle is None and right_elbow_angle is None:
        return None

    # Fallback if only one arm is visible
    if left_elbow_angle is None:
        left_elbow_angle = right_elbow_angle
    if right_elbow_angle is None:
        right_elbow_angle = left_elbow_angle

    # Body inclination (average L/R)
    inc_l = _compute_body_inclination(ls_xy, la_xy)
    inc_r = _compute_body_inclination(rs_xy, ra_xy)
    avg_inclination = (inc_l + inc_r) / 2.0

    return {
        "left_elbow": float(left_elbow_angle),
        "right_elbow": float(right_elbow_angle),
        "inclination": float(avg_inclination),
    }


def _analyze_pushup_pose(landmarks):
    """
    Existing heuristic push-up analysis (kept for additional coaching text).

    Returns dict:
    {
      "pose_detected": bool,
      "phase": "up" | "down" | "unknown",
      "elbow_angle": float | None,
      "hip_angle": float | None,
      "feedback": str
    }
    """

    def pt(idx):
        lm = landmarks[idx]
        return (lm.x, lm.y), lm.visibility

    # Shoulders, elbows, wrists
    left_shoulder, ls_vis = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    left_elbow, le_vis = pt(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    left_wrist, lw_vis = pt(mp_pose.PoseLandmark.LEFT_WRIST.value)

    right_shoulder, rs_vis = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    right_elbow, re_vis = pt(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    right_wrist, rw_vis = pt(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    # Hips & ankles for body-line check
    left_hip, lh_vis = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    left_ankle, la_vis = pt(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    right_hip, rh_vis = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
    right_ankle, ra_vis = pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    # Elbow angle (pick the best visible arm)
    left_elbow_angle = (
        _angle_3pts(left_shoulder, left_elbow, left_wrist)
        if ls_vis > 0.4 and le_vis > 0.4 and lw_vis > 0.4
        else None
    )
    right_elbow_angle = (
        _angle_3pts(right_shoulder, right_elbow, right_wrist)
        if rs_vis > 0.4 and re_vis > 0.4 and rw_vis > 0.4
        else None
    )

    elbow_angle = None
    if left_elbow_angle is not None and right_elbow_angle is not None:
        elbow_angle = max(left_elbow_angle, right_elbow_angle)
    elif left_elbow_angle is not None:
        elbow_angle = left_elbow_angle
    elif right_elbow_angle is not None:
        elbow_angle = right_elbow_angle

    # Hip / body-line angle (shoulder-hip-ankle)
    left_hip_angle = (
        _angle_3pts(left_shoulder, left_hip, left_ankle)
        if ls_vis > 0.4 and lh_vis > 0.4 and la_vis > 0.4
        else None
    )
    right_hip_angle = (
        _angle_3pts(right_shoulder, right_hip, right_ankle)
        if rs_vis > 0.4 and rh_vis > 0.4 and ra_vis > 0.4
        else None
    )

    hip_angle = None
    if left_hip_angle is not None and right_hip_angle is not None:
        hip_angle = max(left_hip_angle, right_hip_angle)
    elif left_hip_angle is not None:
        hip_angle = left_hip_angle
    elif right_hip_angle is not None:
        hip_angle = right_hip_angle

    if elbow_angle is None:
        return {
            "pose_detected": False,
            "phase": "unknown",
            "elbow_angle": None,
            "hip_angle": hip_angle,
            "feedback": "We can’t clearly see your arms. Turn slightly sideways so your elbows and shoulders are visible.",
        }

    # --- Phase classification (simple thresholds, tune later) ---
    if elbow_angle < 95:
        phase = "down"  # chest near floor
    elif elbow_angle > 150:
        phase = "up"  # top plank position
    else:
        phase = "unknown"

    # --- Rule-based feedback ---
    issues = []

    # Hips line: discourage sagging / piked hips
    if hip_angle is not None and hip_angle < 160:
        issues.append(
            "Keep your body in a straight line – tighten your core so your hips don’t sag or pike up."
        )

    # Range of motion / lockout
    if elbow_angle > 165:
        issues.append("Nice lockout – keep elbows soft, don’t hyperextend.")
    elif elbow_angle > 120 and phase == "up":
        issues.append("At the top, straighten your arms a bit more for a full rep.")
    elif elbow_angle > 110 and phase == "down":
        issues.append(
            "Go a bit deeper – bring your chest closer to the floor if shoulders allow."
        )

    if not issues:
        feedback = "Great push-up form – strong plank and good range of motion."
    else:
        feedback = issues[0]

    return {
        "pose_detected": True,
        "phase": phase,
        "elbow_angle": elbow_angle,
        "hip_angle": hip_angle,
        "feedback": feedback,
    }


# -------------------------
# DASHBOARD OVERVIEW
# -------------------------
@dashboard_bp.route("/overview", methods=["GET"])
@jwt_required()
def dashboard_overview():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({"message": "user not found"}), 404

    today = date.today()
    week_start = today - timedelta(days=6)

    today_stats = UserDailyStats.query.filter_by(
        user_id=user.id, stat_date=today
    ).first()

    today_dict = {
        "points": today_stats.total_points if today_stats else 0,
        "workouts": today_stats.total_workouts if today_stats else 0,
        "duration_seconds": today_stats.total_duration_seconds
        if today_stats
        else 0,
        "reps": today_stats.total_reps if today_stats else 0,
    }

    stats_rows = (
        UserDailyStats.query.filter(
            UserDailyStats.user_id == user.id,
            UserDailyStats.stat_date >= week_start,
            UserDailyStats.stat_date <= today,
        )
        .order_by(UserDailyStats.stat_date.asc())
        .all()
    )

    stats_by_date = {row.stat_date: row for row in stats_rows}
    by_day = []
    total_points = total_workouts = total_duration_seconds = total_reps = 0

    for i in range(7):
        d = week_start + timedelta(days=i)
        row = stats_by_date.get(d)

        p = row.total_points if row else 0
        w = row.total_workouts if row else 0
        dur = row.total_duration_seconds if row else 0
        r = row.total_reps if row else 0

        total_points += p
        total_workouts += w
        total_duration_seconds += dur
        total_reps += r

        by_day.append(
            {
                "date": d.isoformat(),
                "points": p,
                "workouts": w,
                "duration_seconds": dur,
                "reps": r,
            }
        )

    last7days = {
        "total_points": total_points,
        "total_workouts": total_workouts,
        "total_duration_seconds": total_duration_seconds,
        "total_reps": total_reps,
        "by_day": by_day,
    }

    streak = {
        "current_streak_days": user.current_streak_days,
        "longest_streak_days": user.longest_streak_days,
    }

    return (
        jsonify(
            {
                "user": user.to_dict(),
                "today": today_dict,
                "last7days": last7days,
                "streak": streak,
            }
        ),
        200,
    )


# -------------------------
# RECENT WORKOUTS
# -------------------------
@workout_bp.route("/recent", methods=["GET"])
@jwt_required()
def recent_workouts():
    user_id = get_jwt_identity()

    try:
        limit = int(request.args.get("limit", 5))
        if limit <= 0:
            limit = 5
    except ValueError:
        limit = 5

    rows = (
        Workout.query.filter_by(user_id=user_id)
        .order_by(Workout.workout_date.desc(), Workout.started_at.desc())
        .limit(limit)
        .all()
    )

    return jsonify({"workouts": [w.to_summary_dict() for w in rows]}), 200


# -------------------------
# START WORKOUT
# -------------------------
@workout_bp.route("/start", methods=["POST"])
@jwt_required()
def start_workout():
    current_user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    title = (data.get("title") or "Workout").strip()

    exercise_id = (data.get("exercise_id") or "").strip() or None
    difficulty = (data.get("difficulty") or "").strip() or None
    preset_id = (data.get("preset_id") or "").strip() or None

    try:
        target_sets = int(data.get("target_sets")) if data.get("target_sets") else None
    except (TypeError, ValueError):
        target_sets = None

    try:
        target_reps = int(data.get("target_reps")) if data.get("target_reps") else None
    except (TypeError, ValueError):
        target_reps = None

    now = datetime.utcnow()
    today = now.date()

    workout = Workout(
        user_id=current_user_id,
        title=title,
        workout_date=today,
        started_at=now,
        total_duration_seconds=0,
        total_points_earned=0,
        exercise_id=exercise_id,
        difficulty=difficulty,
        preset_id=preset_id,
        target_sets=target_sets,
        target_reps=target_reps,
    )
    db.session.add(workout)
    db.session.commit()

    return jsonify({"workout": workout.to_summary_dict()}), 201


# -------------------------
# COMPLETE WORKOUT
# -------------------------
@workout_bp.route("/complete", methods=["POST"])
@jwt_required()
def complete_workout():
    current_user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    workout_id = data.get("workout_id")
    if not workout_id:
        return jsonify({"message": "workout_id is required"}), 400

    try:
        workout_id = int(workout_id)
    except (TypeError, ValueError):
        return jsonify({"message": "workout_id must be an integer"}), 400

    workout = Workout.query.filter_by(
        id=workout_id, user_id=current_user_id
    ).first()
    if not workout:
        return jsonify({"message": "Workout not found"}), 404

    total_duration_seconds = int(data.get("total_duration_seconds") or 0)
    total_points_earned = int(data.get("total_points_earned") or 0)
    total_reps = int(data.get("total_reps") or 0)

    now = datetime.utcnow()
    workout.ended_at = now
    workout.total_duration_seconds = total_duration_seconds
    workout.total_points_earned = total_points_earned
    workout.total_reps = total_reps  # NEW

    if workout.calories_estimate is None:
        workout.calories_estimate = 0

    # ... rest of your stats / streak logic stays the same ...


# -------------------------
# POSE ANALYSIS FOR EXERCISES
# -------------------------
@workout_bp.route("/<exercise_id>/analyze_frame", methods=["POST"])
@jwt_required()
def analyze_exercise_frame(exercise_id):
    """
    Analyze a single frame for a given exercise.

    JSON body:
    {
      "workout_id": 123,     # optional but recommended
      "image_base64": "..."  # required
    }
    """
    current_user_id = int(get_jwt_identity())

    data = request.get_json() or {}
    image_b64 = data.get("image_base64")
    raw_workout_id = data.get("workout_id") or 0

    try:
        workout_id = int(raw_workout_id)
    except (TypeError, ValueError):
        workout_id = 0

    if not image_b64:
        return jsonify({"message": "image_base64 is required"}), 400

    # Decode base64 -> OpenCV image
    try:
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception:
        return jsonify({"message": "Invalid image_base64"}), 400

    # Resize for speed
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    max_side = max(h, w)
    target = 320
    if max_side > target:
        scale = target / float(max_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h))

    results = _pose_detector.process(rgb)

    if not results.pose_landmarks:
        return (
            jsonify(
                {
                    "pose_detected": False,
                    "phase": "unknown",
                    "elbow_angle": None,
                    "hip_angle": None,
                    "feedback": "We couldn’t detect your full body. Make sure you’re fully in the frame in a push-up position.",
                    "count": 0,
                    "state": "INIT",
                    "mode": "UNKNOWN",
                    "landmarks": [],
                }
            ),
            200,
        )

    landmarks = results.pose_landmarks.landmark
    exercise_id = (exercise_id or "").lower()

    if exercise_id == "pushup":
        # 1) Extract elbow angles + body inclination
        feats = _extract_pushup_features_from_landmarks(landmarks)
        if feats is None:
            result = {
                "pose_detected": False,
                "phase": "unknown",
                "elbow_angle": None,
                "hip_angle": None,
                "feedback": "We can’t clearly see your elbows. Turn slightly sideways so your arms are visible.",
                "count": 0,
                "state": "INIT",
                "mode": "UNKNOWN",
            }
        else:
            # 2) Get / create session counter
            counter = _get_pushup_session_counter(
                user_id=current_user_id,
                workout_id=workout_id,
                exercise_id=exercise_id,
            )

            state, count, current_angle, fb, _fb_color, mode = counter.update(
                feats["left_elbow"], feats["right_elbow"], feats["inclination"]
            )

            # Optional: keep heuristic analysis for extra info (phase, hip angle, etc.)
            heuristic = _analyze_pushup_pose(landmarks)

            result = {
                "pose_detected": heuristic.get("pose_detected", True),
                "phase": heuristic.get("phase", "unknown"),
                "elbow_angle": heuristic.get("elbow_angle"),
                "hip_angle": heuristic.get("hip_angle"),
                "feedback": fb or heuristic.get("feedback"),
                "count": int(count),
                "state": state,  # "UP" / "DOWN"
                "mode": mode,    # "PUSHUP" / "STANDING"
                "avg_elbow_angle": float(current_angle),
                "inclination": float(feats["inclination"]),
            }
    else:
        # Placeholder for other exercises
        result = {
            "pose_detected": True,
            "phase": "unknown",
            "elbow_angle": None,
            "hip_angle": None,
            "feedback": f"Pose analysis for '{exercise_id}' is not implemented yet. Push-ups are fully supported.",
            "count": 0,
            "state": "INIT",
            "mode": "UNKNOWN",
        }

    # Always include landmark coordinates (normalized) for client-side drawing
    result["landmarks"] = [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "visibility": float(lm.visibility),
        }
        for lm in landmarks
    ]

    return jsonify(result), 200
