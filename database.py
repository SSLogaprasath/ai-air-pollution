"""
Database module for Air Quality Forecasting app.

MySQL database with tables for users, saved locations,
alert preferences, health profiles, and forecast history.
"""

import json
import os
from datetime import datetime, timezone
from contextlib import contextmanager

import mysql.connector
from mysql.connector import pooling

# MySQL Configuration — override via environment variables
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "Logan@1804")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "airquality")

# Connection pool for efficient reuse
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        # Ensure the database exists before creating the pool
        _conn = mysql.connector.connect(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
        )
        cursor = _conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        _conn.close()

        _pool = pooling.MySQLConnectionPool(
            pool_name="aq_pool",
            pool_size=5,
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            autocommit=False,
        )
    return _pool


def get_connection():
    return _get_pool().get_connection()


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _dict_row(cursor):
    """Fetch one row as a dict (or None)."""
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def _dict_rows(cursor):
    """Fetch all rows as a list of dicts."""
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(30) NOT NULL UNIQUE,
                email VARCHAR(120) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_locations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                city_name VARCHAR(100) NOT NULL,
                state VARCHAR(100),
                lat DOUBLE NOT NULL,
                lon DOUBLE NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE KEY uq_user_city (user_id, city_name)
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alert_preferences (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                city_name VARCHAR(100) NOT NULL,
                threshold_pm25 DOUBLE NOT NULL DEFAULT 55.4,
                enabled TINYINT(1) NOT NULL DEFAULT 1,
                last_alerted_at DATETIME DEFAULT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE KEY uq_user_alert (user_id, city_name)
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS health_profiles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL UNIQUE,
                age_group VARCHAR(20) DEFAULT 'adult',
                conditions JSON DEFAULT NULL,
                outdoor_hours DOUBLE DEFAULT 2.0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                city_name VARCHAR(100) NOT NULL,
                avg_pm25 DOUBLE NOT NULL,
                aqi_category VARCHAR(50) NOT NULL,
                num_sensors INT DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_fh_user (user_id),
                INDEX idx_fh_created (user_id, created_at)
            ) ENGINE=InnoDB
        """)
        conn.commit()


# ============================================================
# User CRUD
# ============================================================

def create_user(username: str, email: str, password_hash: str) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email.lower(), password_hash),
        )
        return cur.lastrowid


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email.lower(),))
        return _dict_row(cur)


def get_user_by_id(user_id: int) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = %s",
            (user_id,),
        )
        return _dict_row(cur)


def get_user_by_username(username: str) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        return _dict_row(cur)


# ============================================================
# Saved Locations
# ============================================================

def save_location(user_id: int, city_name: str, state: str, lat: float, lon: float) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO saved_locations (user_id, city_name, state, lat, lon)
               VALUES (%s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
                   state=VALUES(state), lat=VALUES(lat), lon=VALUES(lon)""",
            (user_id, city_name, state, lat, lon),
        )
        return cur.lastrowid


def get_saved_locations(user_id: int) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM saved_locations WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        return _dict_rows(cur)


def delete_saved_location(user_id: int, city_name: str) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM saved_locations WHERE user_id = %s AND city_name = %s",
            (user_id, city_name),
        )
        return cur.rowcount > 0


# ============================================================
# Alert Preferences
# ============================================================

def upsert_alert(user_id: int, city_name: str, threshold: float, enabled: bool) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO alert_preferences (user_id, city_name, threshold_pm25, enabled)
               VALUES (%s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
                   threshold_pm25=VALUES(threshold_pm25),
                   enabled=VALUES(enabled)""",
            (user_id, city_name, threshold, int(enabled)),
        )
        return cur.lastrowid


def get_alerts(user_id: int) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM alert_preferences WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        return _dict_rows(cur)


def delete_alert(user_id: int, city_name: str) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM alert_preferences WHERE user_id = %s AND city_name = %s",
            (user_id, city_name),
        )
        return cur.rowcount > 0


# ============================================================
# Health Profile
# ============================================================

def upsert_health_profile(
    user_id: int,
    age_group: str = "adult",
    conditions: list[str] | None = None,
    outdoor_hours: float = 2.0,
) -> int:
    conds_json = json.dumps(conditions or [])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO health_profiles (user_id, age_group, conditions, outdoor_hours)
               VALUES (%s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
                   age_group=VALUES(age_group),
                   conditions=VALUES(conditions),
                   outdoor_hours=VALUES(outdoor_hours)""",
            (user_id, age_group, conds_json, outdoor_hours),
        )
        return cur.lastrowid


def get_health_profile(user_id: int) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM health_profiles WHERE user_id = %s", (user_id,)
        )
        row = _dict_row(cur)
        if row:
            conds = row["conditions"]
            if isinstance(conds, str):
                row["conditions"] = json.loads(conds)
            elif conds is None:
                row["conditions"] = []
            return row
        return None


# ============================================================
# Forecast History
# ============================================================

def add_forecast_history(
    user_id: int, city_name: str, avg_pm25: float, aqi_category: str, num_sensors: int = 0
) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO forecast_history (user_id, city_name, avg_pm25, aqi_category, num_sensors)
               VALUES (%s, %s, %s, %s, %s)""",
            (user_id, city_name, avg_pm25, aqi_category, num_sensors),
        )
        return cur.lastrowid


def get_forecast_history(user_id: int, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM forecast_history
               WHERE user_id = %s
               ORDER BY created_at DESC LIMIT %s""",
            (user_id, limit),
        )
        return _dict_rows(cur)


# ============================================================
# Alert Checking (used by background scheduler)
# ============================================================

def get_enabled_alerts_with_emails(cooldown_hours: int = 6) -> list[dict]:
    """Fetch all enabled alerts that haven't been alerted within the cooldown window,
    joined with user email."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT a.id, a.user_id, a.city_name, a.threshold_pm25,
                      a.last_alerted_at, u.email, u.username
               FROM alert_preferences a
               JOIN users u ON u.id = a.user_id
               WHERE a.enabled = 1
                 AND (a.last_alerted_at IS NULL
                      OR a.last_alerted_at < NOW() - INTERVAL %s HOUR)""",
            (cooldown_hours,),
        )
        return _dict_rows(cur)


def mark_alert_sent(alert_id: int) -> None:
    """Update last_alerted_at to now after sending an alert email."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE alert_preferences SET last_alerted_at = NOW() WHERE id = %s",
            (alert_id,),
        )


# Initialize on import
init_db()
