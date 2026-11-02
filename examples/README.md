# Examples

Small scripts that use the library the way I'd use it in a real project. They need `OPENAI_API_KEY` (see the main README). Each one has a `main(model=...)` so the tests can run them with a fake model.

| script | what it shows |
|---|---|
| `code_review.py path/to/file.py` | the `reviewer` persona + file tools locked to the file's folder |
| `meeting_actions.py notes.txt` | structured output: pull a to-do list out of messy meeting notes |
| `support_desk.py` | one assistant, separate memory per customer (sessions) |
| `study_buddy.py` | the `tutor` persona with memory, saved to JSON so you can continue tomorrow |
