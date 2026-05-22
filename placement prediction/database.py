"""
database.py - SQLite Database Module for Placement Prediction System
====================================================================
Handles all database operations including creating tables,
inserting student records, querying records, and generating statistics.
"""

import sqlite3
import os
from datetime import datetime


# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement.db")


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def initialize_database():
    """
    Create the students table if it doesn't exist.
    This is called once when the application starts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            cgpa REAL NOT NULL,
            aptitude_score REAL NOT NULL,
            communication_skills INTEGER NOT NULL,
            technical_skills INTEGER NOT NULL,
            internship_count INTEGER NOT NULL,
            project_count INTEGER NOT NULL,
            attendance_percentage REAL NOT NULL,
            certifications_count INTEGER NOT NULL,
            prediction TEXT DEFAULT 'N/A',
            probability REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_student(name, cgpa, aptitude, communication, technical,
                   internships, projects, attendance, certifications,
                   prediction="N/A", probability=0.0):
    """Insert a new student record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students (name, cgpa, aptitude_score, communication_skills,
                              technical_skills, internship_count, project_count,
                              attendance_percentage, certifications_count,
                              prediction, probability, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, cgpa, aptitude, communication, technical, internships,
          projects, attendance, certifications, prediction, probability,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    student_id = cursor.lastrowid
    conn.close()
    return student_id


def get_all_students():
    """Retrieve all student records from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def search_students(query):
    """Search students by name (case-insensitive partial match)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM students WHERE LOWER(name) LIKE ? ORDER BY id DESC",
        (f"%{query.lower()}%",)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_student_by_id(student_id):
    """Retrieve a single student record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_student(student_id):
    """Delete a student record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()


def get_statistics():
    """
    Calculate and return placement statistics from stored records.
    Returns a dictionary with total, placed, not placed counts and percentages.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Total students
    cursor.execute("SELECT COUNT(*) FROM students")
    total = cursor.fetchone()[0]

    # Placed students
    cursor.execute("SELECT COUNT(*) FROM students WHERE prediction = 'Placed'")
    placed = cursor.fetchone()[0]

    # Not placed students
    not_placed = total - placed

    # Average CGPA
    cursor.execute("SELECT AVG(cgpa) FROM students")
    avg_cgpa = cursor.fetchone()[0] or 0.0

    # Average probability
    cursor.execute("SELECT AVG(probability) FROM students")
    avg_probability = cursor.fetchone()[0] or 0.0

    # Average aptitude
    cursor.execute("SELECT AVG(aptitude_score) FROM students")
    avg_aptitude = cursor.fetchone()[0] or 0.0

    conn.close()

    return {
        "total": total,
        "placed": placed,
        "not_placed": not_placed,
        "placement_rate": (placed / total * 100) if total > 0 else 0,
        "avg_cgpa": round(avg_cgpa, 2),
        "avg_probability": round(avg_probability, 1),
        "avg_aptitude": round(avg_aptitude, 1)
    }


def get_cgpa_distribution():
    """Get the distribution of CGPA ranges for chart visualization."""
    conn = get_connection()
    cursor = conn.cursor()
    ranges = {
        "< 6.0": 0, "6.0 - 7.0": 0, "7.0 - 8.0": 0,
        "8.0 - 9.0": 0, "9.0+": 0
    }
    cursor.execute("SELECT cgpa FROM students")
    for row in cursor.fetchall():
        cgpa = row[0]
        if cgpa < 6.0:
            ranges["< 6.0"] += 1
        elif cgpa < 7.0:
            ranges["6.0 - 7.0"] += 1
        elif cgpa < 8.0:
            ranges["7.0 - 8.0"] += 1
        elif cgpa < 9.0:
            ranges["8.0 - 9.0"] += 1
        else:
            ranges["9.0+"] += 1
    conn.close()
    return ranges
