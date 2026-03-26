# Helper Code

These files need to be copied into your project for certain experiment variants.

## prepare_thread_aware.py
Copy to: `src/preprocessing/prepare_thread_aware.py`

This is a thread-aware preprocessing variant that prepends the parent
question to reply-sourced examples, giving the student model the same
question→answer context the teacher had.

**Example:**
- Without thread-aware: Student sees `"כן בטח, אותה כמות"` (the reply alone)
- With thread-aware: Student sees `"[שאלה] אפשר במקום חמאה שמן קוקוס? [תשובה] כן בטח, אותה כמות"`

The span alignment handles the offset shift automatically.
