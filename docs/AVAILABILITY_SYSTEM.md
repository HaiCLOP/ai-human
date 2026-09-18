# Availability System & Schedule Resolver Specification

## 1. Architectural Philosophy

The character is an authentic fictional entity with a structured everyday life, recurring commitments, and finite social availability. She must never appear artificially available 24/7.

The system deterministically computes her real-time availability using a layered evaluation:

$$\text{CurrentAvailability}(t) = \text{Resolve}\Big(\text{SpecialEvents}(t), \, \text{AwayOverride}(t), \, \text{WeeklySchedule}(\text{Day}(t), \text{Time}(t))\Big)$$

```mermaid
flowchart TD
    TIME[Current Real / Fictional Time: t] --> SPECIAL{Active Special Event?}
    SPECIAL -->|Yes (Exam / Holiday / Function)| APPLY_SPECIAL[Apply Event Override State & Location]
    SPECIAL -->|No| AWAY_CHECK{Active Away Override / away_until?}
    AWAY_CHECK -->|away_until > t| APPLY_AWAY[Enforce Away State & Activity]
    AWAY_CHECK -->|No| WEEKLY[Query Canonical Weekly Routine]
    WEEKLY --> MATCH_SLOT[Match Day & Time Window [Start, End]]
    MATCH_SLOT --> RESOLVED_STATE[Deterministically Resolved Availability]
```

---

## 2. Availability States Taxonomy

| Availability State | Description | Inbound Message Handling |
| :--- | :--- | :--- |
| `AVAILABLE` | Free to chat, browse, or text casually. | Process immediately via normal conversation pipeline. |
| `BUSY` | Short domestic chores, getting ready, commute. | Suppress immediate response; queue message. |
| `AWAY` | Temporarily stepped away from phone or taking a break. | Suppress immediate response; queue message. |
| `SLEEPING` | Hard sleep period (e.g. 23:00 - 07:00). Zero responses. | Strictly silent queueing. No LLM calls. |
| `AT_SCHOOL` | School classes & academic schedule (e.g. 07:45 - 14:45). | Strictly silent queueing. Zero immediate replies. |
| `AT_TUITION` | Coaching / tuition sessions (e.g. 17:00 - 19:00). | Strictly silent queueing. Zero immediate replies. |
| `STUDYING` | Focused self-study / homework / exam prep. | Either departure note or silent queueing. |
| `EATING` | Family lunch or dinner (30 - 45 mins). | Silent queueing. |
| `FAMILY_TIME` | Family outings, gatherings, or movie night. | Silent queueing. |

---

## 3. Pre-LLM Gating Protocol (Zero API Waste)

When an inbound Instagram DM arrives:
1. Determine `AvailabilityState` at current time.
2. If `State != AVAILABLE`:
   * Insert message into `messages` table in `RECEIVED` status.
   * **Abort immediately**. Do not invoke LLM. Do not emit typing keystrokes.
   * Return `None` to browser controller.

---

## 4. Departure Protocol & Warning Windows

### 4.1 Upcoming Commitment Warning
* **Warning Window**: Configurable in YAML (`availability.departure_warning_minutes: 15`).
* If $t \ge T_{\text{start}} - 15\text{min}$:
  * If conversation is currently active, character may naturally mention her upcoming commitment:
    * E.g.: *"btw mujhe 15 min mein tuition ke liye nikalna hai 😭"*
  * Track `warning_sent_for_activity` in SQLite to prevent repeating the same warning twice.

### 4.2 Departure Trigger
* When $t \ge T_{\text{start}}$:
  * If an inbound message arrives or conversation turn finishes, the character generates a natural departure goodbye:
    * E.g.: *"chalo mujhe jaana padega 😭 2 ghante baad aati hoon, maths ka paper parso hai 💀"*
  * Sets `away_until = T_end`.
  * Sets availability state to the scheduled state (e.g. `AT_TUITION`).

---

## 5. Batch Return Protocol (No Fake Message Queues)

When the character's activity finishes ($t \ge \text{away\_until}$) and she transitions back to `AVAILABLE`:
1. The system checks for unhandled messages accumulated in `messages` where `status = 'RECEIVED'`.
2. **Anti-Spam Invariant**: The character **never** sends $N$ individual replies to $N$ queued messages.
3. All accumulated messages are batched into a single unified context:
   ```markdown
   [ACCUMULATED MESSAGES WHILE AWAY]
   - 17:15 @user: "where are you?"
   - 17:35 @user: "hello?"
   - 18:50 @user: "are you back?"
   ```
4. LLM generates **one single natural return message**:
   * E.g.: *"backkk 😭 tuition ne jaan le li meri. abhi dekha sab, kya chal raha tha?"*
5. All accumulated messages are marked `PROCESSED`.
