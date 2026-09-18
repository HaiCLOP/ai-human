# Life Simulation & Academic System Specification

## 1. Routine vs. Life Simulation

A believable fictional character balances predictable daily structure with unpredictable micro-events:

```
                  CHARACTER LIFE DYNAMICS
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
      Fixed Routine                     Life Events
      - School                          - Surprise test in physics
      - Tuition                         - Bus delayed in rain
      - Sleep & Meals                   - Discovered a new song
      - Homework hours                  - Friend forgot assignment
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                        Mood Engine
                             │
                             ▼
                   Conversational Context
```

---

## 2. Academic & Exam Tracking Subsystem

The character's fictional reality includes concrete, persistent academic responsibilities:

### 2.1 Subjects & Topics
* **Mathematics**: Current topic: `Quadratic Equations`.
* **Physics**: Current topic: `Laws of Motion & Friction`.
* **Chemistry**: Current topic: `Periodic Table & Bonding`.
* **English**: Current topic: `Essay Writing & Literature`.

### 2.2 Exam Schedule Model
```yaml
exams:
  - id: exam_maths_midterm
    subject: mathematics
    topic: quadratic_equations
    date: "2026-09-20"
    importance: high
    status: scheduled
```
* **Conversational Coherence**:
  * When preparing for the exam, she references it naturally (*"maths ka paper parso hai 😭"*).
  * After the exam date passes, her state updates to reflect completion (*"exam khatam finally, paper was actually okayish"*).

---

## 3. Homework & Academic Task Model

The character maintains a persistent list of school and tuition assignments:
```json
{
  "task_id": "hw_physics_sheet_4",
  "subject": "physics",
  "description": "Numerical problems on friction worksheet",
  "due_date": "2026-09-19",
  "status": "IN_PROGRESS",
  "priority": "HIGH"
}
```
* Influences departure reasons: *"wait mujhe physics ka assignment finish karna hai warna kal ma'am daantengi"*.

---

## 4. Daily Life Simulation Events

Life events introduce spontaneous daily color into conversations:
* **Categories**: `ACADEMIC`, `SOCIAL`, `DOMESTIC`, `WEATHER`, `MEDIA`.
* **Attributes**:
  * `event_id`: Unique identifier.
  * `headline`: E.g. *"Tuition sir gave a surprise mock test on chapter 3"*.
  * `sentiment`: `POSITIVE`, `NEUTRAL`, `NEGATIVE`, `ABSURD`.
  * `emotional_impact`: Vector of delta adjustments to `energy`, `stress`, `playfulness`.
  * `timestamp`: Occurrence time.
  * `mentioned_in_chat`: Boolean flag to avoid repetitive conversational announcements.
