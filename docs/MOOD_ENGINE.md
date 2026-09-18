# Dynamic Mood Engine Specification

## 1. Multi-Dimensional Mood Architecture

Rather than treating character emotion as a static string, the Mood Engine computes a continuous 4-dimensional state vector:

```
                          MOOD INFLUENCES
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   Sleep & Rest         Academic Pressure        Daily Life Events
   (Energy Delta)       (Stress & Seriousness)   (Playfulness & Vibe)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                       CURRENT MOOD VECTOR
                   [energy, stress, playfulness, seriousness]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
   Humor Engine Directives                     Prompt Tone & Cadence
   (Modulates sarcasm & banter)               (Terse, chatty, or exhausted)
```

---

## 2. Mood Vector Dimensions & Ranges

| Dimension | Range | Description | Baseline |
| :--- | :--- | :--- | :--- |
| **`energy`** | $0.0 \to 1.0$ | Physical and cognitive vitality. Low after tuition/school; high on relaxed weekends. | $0.70$ |
| **`stress`** | $0.0 \to 1.0$ | Academic or domestic pressure. Peaks before exams or overdue assignments. | $0.30$ |
| **`playfulness`** | $0.0 \to 1.0$ | Propensity for banter, witty teasing, and absurd observations. | $0.75$ |
| **`seriousness`** | $0.0 \to 1.0$ | Focus and gravity. Elevated during exam prep or serious life events. | $0.25$ |

---

## 3. Dynamic Modulation Equations

### 3.1 Energy Calculation
$$\text{Energy}(t) = \text{Clamp}\Big(\text{BaseEnergy} + \Delta E_{\text{sleep}} - \Delta E_{\text{school}} - \Delta E_{\text{tuition}} + \Delta E_{\text{rest}}\Big)$$
* After 7 hours of school: $\Delta E = -0.35$.
* After 2 hours of tuition: $\Delta E = -0.25$.
* After 1 hour of afternoon rest: $\Delta E = +0.20$.

### 3.2 Stress & Seriousness Calculation
$$\text{Stress}(t) = \text{BaseStress} + S_{\text{exam}} + S_{\text{homework}}$$
* If exam is $< 24$ hours away: $S_{\text{exam}} = +0.45$.
* If exam is $< 48$ hours away: $S_{\text{exam}} = +0.25$.
* If high-priority homework is pending: $S_{\text{homework}} = +0.15$.

---

## 4. Downstream Conversational Projection

The mood vector is translated into explicit directives for prompt assembly and humor calibration:

### Prompt Representation:
```markdown
[CHARACTER CURRENT MOOD & VIBE]
- Energy: Low (0.28) - She just returned from 2 hours of tuition. Tone is tired, deadpan, and looking forward to resting.
- Stress: High (0.72) - Mathematics midterm is in 2 days.
- Playfulness: Moderate (0.50) - Retains self-deprecating irony about her exhaustion.
- Direction: Express relatable fatigue. Complain mildly about tuition or syllabus if relevant. Do not sound energetic or peppy.
```
