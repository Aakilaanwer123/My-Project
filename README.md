# Smart Human Emotion Detection — Streamlit Cloud

## Files
- `app.py`: complete Streamlit application
- `requirements.txt`: Python dependencies
- `packages.txt`: Linux system package for Streamlit Cloud

## Deploy
1. Put these files in the root of your GitHub repository.
2. In Streamlit Community Cloud, select this repository, branch `main`, and main file `app.py`.
3. Deploy/reboot and wait for dependency installation to finish.
4. Open the app on a phone. Allow camera permission when asked.

## Notes
- The first face/voice model load may take several minutes and requires internet access.
- Camera capture is a snapshot, not continuous live video.
- Emotion estimates are not a diagnosis and may be inaccurate.
- Streamlit Cloud's local SQLite history may reset when the app environment restarts.
