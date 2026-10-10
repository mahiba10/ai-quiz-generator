# Beginner Guide: Machine Learning Behind This Project

## 1. What is machine learning?
Instead of writing rules by hand, you show a program many examples and it learns patterns. Example: show it sentences labelled "Python" or "Networks"; it learns which words go with which label.

## 2. Why TF-IDF?
Computers need numbers, not words. TF-IDF turns a sentence into numbers. **TF** (term frequency) = how often a word appears in the sentence. **IDF** (inverse document frequency) = how rare the word is across all sentences. Common words like "the" get low weight; distinctive words like "TCP" or "loops" get high weight.

## 3. What does a text classifier do?
It reads a sentence and chooses one label (here, one of five topics). "Explain overfitting" → Machine Learning Basics.

## 4. Why Naive Bayes or Logistic Regression?
Both are simple, fast and work well on small text datasets. Naive Bayes uses word probabilities; Logistic Regression learns a weight for each word. The training script compares them by cross-validation and keeps the better one (it picked the one printed when you ran training).

## 5. Training and testing
Training = the model learns from labelled examples (80% of the data). Testing = we check it on the other 20% it has never seen. Testing on training data would be like grading a student on questions they memorised. The fixed seed (42) makes the split identical every run.

## 6. Metrics
Example: 10 sentences are truly "Python"; the model labels 8 sentences "Python", 6 of them correctly.
- **Accuracy** = correct predictions / all predictions.
- **Precision** (for Python) = 6/8: when it says Python, how often is it right?
- **Recall** = 6/10: of the real Python sentences, how many did it find?
- **F1-score** = the balance (harmonic mean) of precision and recall.
- **Macro** averages = compute per topic, then average, so every topic counts equally.

## 7. How quiz scoring works
For each question, compare the student's chosen option index with the stored `answer_index`. Count matches. Percentage = correct ÷ total × 100. Empty quiz → 0%. No ML involved.

## 8. How recommendations are generated
After scoring, we group missed questions by their `concept` tag (e.g. "Loops"). Concepts with misses are listed as "revise this", and unseen questions with the same concept are suggested. This is rule-based logic, not ML.

## 9. How the Streamlit app works
Streamlit re-runs `app.py` from top to bottom on every click. To keep your answers, we store them in `st.session_state`, which survives reruns. The `page` value in session state decides which page function to draw.

## 10. How files connect
- `app.py` – the user interface; calls everything else.
- `data/questions.json` → `utils/quiz_logic.py` loads, picks and scores questions.
- `utils/recommendations.py` – turns results into advice.
- `data/topic_training_data.csv` → `ml/train_model.py` → `models/` → `ml/topic_classifier.py` → Topic Finder page.
- `tests/` – automatic checks.

## 11. Quiz app vs ML model vs recommendation logic
| Part | What it does | Learns from data? |
|---|---|---|
| Quiz application | Shows questions, collects answers, scores | No |
| ML model | Predicts a topic from text | Yes |
| Recommendation logic | Suggests concepts to revise from your answers | No (rules) |

## 15 Likely Viva Questions
1. **What is your project about?** A quiz app that scores practice, explains answers, recommends revision, and uses a small ML model to suggest a topic from a query.
2. **How does it relate to SDG 4?** It gives free, accessible self-assessment that supports quality education.
3. **Where is the machine learning?** In the topic classifier (TF-IDF + a scikit-learn classifier). Scoring and recommendations are not ML.
4. **Why not generate questions with AI?** Generated questions can be factually wrong; a curated bank is reliable and works offline.
5. **What is TF-IDF?** A way to convert text into numbers that highlights distinctive words.
6. **Why Logistic Regression/Naive Bayes?** Simple, fast, and suited to small text datasets; chosen by cross-validation.
7. **What is the train/test split?** 80% to learn, 20% to check on unseen data; stratified, seed 42.
8. **How did you avoid data leakage?** Removed duplicates, split before fitting, and used a Pipeline so TF-IDF is fitted only on training text; model choice used cross-validation on training data only.
9. **What accuracy did you get?** State the exact numbers from `models/metrics.json` and say the test set is small (28 sentences).
10. **What are precision and recall?** Precision: of predicted positives, how many are right. Recall: of actual positives, how many were found.
11. **What are the limitations?** Small hand-made dataset, overlapping topics, session-only history, 75 questions.
12. **How do you handle unknown input?** Empty text, no known words, or confidence below 0.40 is reported as unknown/uncertain.
13. **How is the score calculated?** Count answers matching the stored correct index ÷ total × 100.
14. **How do you keep answers when the page reruns?** Store them in `st.session_state`.
15. **What would you improve?** More data and questions, persistent accounts, spaced repetition, and human-reviewed AI-assisted question drafting.
