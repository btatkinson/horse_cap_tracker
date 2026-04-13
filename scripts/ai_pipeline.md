### AI PipeLine overview

1. User downloads a form
2. download_upc.py gets the ml odds and other info in the same format as before
3. chop_form.py separates into race by race txt files
4. track_condition.py gets weather/track info
5. send_claude.py, send_gemini.py sends every race with dynamic weaether prompt to AI predictors
6. (insert horse capper scrapers here)
7. 