# Concept Note

**Project title:** AI-Powered Quiz Generator and Personalized Learning Assistant
**Student/author:** _[Your name, roll number, college]_
**Internship:** AICTE–IBM SkillsBuild Machine Learning and Applied AI Internship
**SDG alignment:** SDG 4 – Quality Education

## 1. Abstract
This project is a Streamlit web application for self-assessment. Students select a topic, difficulty and number of questions,
answer multiple-choice questions from a curated bank, and receive scores, explanations and revision recommendations
derived from their actual answers. A lightweight scikit-learn text classifier (TF-IDF with a linear classifier) predicts the likely
topic of a learning query and suggests a quiz. Evaluation results: **pending — to be copied from `models/metrics.json` after you run the training script.**

## 2. Introduction
Frequent low-stakes practice with feedback helps learners find gaps. Many students lack a simple, free tool that shows what to revise.

## 3. Problem statement
Students struggle to assess their understanding of a topic and to identify what to practise.

## 4. Proposed solution
A web app that combines (a) a verified question bank, (b) automatic scoring, (c) rule-based recommendations, and (d) an ML topic classifier.

## 5. Objectives
1. Provide topic-wise MCQ practice. 2. Give immediate feedback with explanations. 3. Recommend concepts to revise. 4. Apply and honestly evaluate one ML technique.

## 6. Scope
Five computer-science topics, three difficulty levels, single-user, session-based history. Out of scope: accounts, databases, AI-generated questions, adaptive difficulty.

## 7. Target users
Undergraduate engineering students and self-learners studying CS fundamentals.

## 8. Existing approaches and limitations
Textbook exercises (no instant feedback), online quiz platforms (often account-based, generic feedback, may need internet or payment), classroom tests (infrequent).

## 9. Proposed methodology
Requirements → question-bank curation → dataset creation → model training and evaluation → application development → testing → deployment.

## 10. System architecture
Streamlit interface (`app.py`); quiz logic (`utils/quiz_logic.py`); recommendation logic (`utils/recommendations.py`); ML training and prediction (`ml/`); data files (`data/`); saved model (`models/`).

## 11. Dataset description
`data/topic_training_data.csv`: 140 hand-written learning queries, 28 per topic, labelled with five topics. Duplicates are removed during training. Question bank: 75 MCQs (5 per topic per difficulty) with concept tags and explanations.

## 12. Machine learning algorithm
TF-IDF vectorisation (unigrams and bigrams) followed by a classifier. Multinomial Naive Bayes and Logistic Regression variants are compared by 5-fold cross-validation on the training set; the best is trained and tested once on a held-out stratified 20% test set (random seed 42).

## 13. Training and evaluation methodology
Metrics: accuracy, macro precision, recall and F1, per-topic report and confusion matrix. Leakage is avoided through a Pipeline and de-duplication. Low-confidence or unseen-word inputs are reported as uncertain/unknown.

## 14. Software and hardware requirements
Python 3.10+, Streamlit, Pandas, scikit-learn, joblib, pytest; any laptop with 4 GB RAM; a web browser. No GPU or paid services.

## 15. Expected outcomes
A working, tested app; a documented classifier with honestly reported metrics; deployed demo (after deployment is verified).

## 16. Actual results (to be completed after testing)
| Item | Result |
|---|---|
| Automated tests passed | _pending – run `pytest`_ |
| Classifier test accuracy | _pending – from `models/metrics.json`_ |
| Macro F1 | _pending_ |
| Deployed URL | _pending – add only after verification_ |

## 17. Limitations
Small hand-made dataset and question bank; the classifier may confuse overlapping topics; recommendations are rule-based; history is not persistent.

## 18. Ethical considerations
Questions are verified rather than machine-generated to avoid teaching wrong answers; feedback describes only the quiz result and makes no claims about ability; no personal data is collected or stored; classifier limits are disclosed.

## 19. Future scope
Larger bank, spaced repetition, persistent progress, more topics, larger independently collected dataset, human-reviewed AI-assisted question drafting.

## 20. Conclusion
The project demonstrates a simple, honest application of ML in education, supporting SDG 4 through accessible self-assessment.

## 21. References
_Add only sources you actually read and cite, for example: scikit-learn documentation (scikit-learn.org), Streamlit documentation (docs.streamlit.io), and the United Nations SDG 4 page (sdgs.un.org/goals/goal4). Do not add references you have not consulted._
