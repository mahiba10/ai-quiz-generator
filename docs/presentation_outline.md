# Presentation Outline (12 slides)
Placeholders marked **[PLACEHOLDER]** must be replaced with your real screenshots/results.

## Slide 1 – Title
- AI-Powered Quiz Generator and Personalized Learning Assistant
- Your name, college, AICTE–IBM SkillsBuild ML & Applied AI Internship
- Visual: SDG 4 logo + app logo/home screenshot **[PLACEHOLDER]**
- Notes: "Hello, I'm ___. My project helps students practise and find what to revise."

## Slide 2 – Problem Statement
- Students can't easily judge their understanding; feedback is delayed; revision is unfocused
- Visual: simple "confused student" icon or 3-step problem flow
- Notes: Describe a typical exam-preparation situation.

## Slide 3 – Motivation and SDG 4
- Quality education needs accessible practice tools
- Free, no-account tool, works with local files
- Visual: SDG 4 icon
- Notes: Link self-assessment to inclusive, equitable quality education.

## Slide 4 – Objectives
- Topic-wise MCQ practice; instant feedback; recommendations; one real ML component
- Visual: 4 icons
- Notes: Read each objective, mention which feature meets it.

## Slide 5 – Proposed Solution
- Quiz → score → explanations → revision suggestions; Topic Finder suggests a quiz
- Visual: user journey diagram (Home → Setup → Quiz → Results)
- Notes: Walk through the journey.

## Slide 6 – System Architecture
- Streamlit UI; quiz logic; recommendation logic; ML model; JSON/CSV data
- Visual: the architecture diagram from README
- Notes: Stress that ML is separate from scoring.

## Slide 7 – Dataset and Preprocessing
- 75 curated MCQs; 140 labelled learning queries (28/topic)
- Cleaning: strip text, remove empty/duplicate rows
- Visual: small table of sample rows
- Notes: Admit the dataset is small and hand-written.

## Slide 8 – Machine Learning Methodology
- TF-IDF + Naive Bayes / Logistic Regression; 5-fold CV on training data; 80/20 stratified split; seed 42
- Unknown/low-confidence handling
- Visual: pipeline diagram (text → TF-IDF → classifier → topic)
- Notes: Explain each box in simple words.

## Slide 9 – Application Implementation and Features
- Topic cards, quiz navigation, preserved answers, results, history, Topic Finder
- Visual: feature screenshots **[PLACEHOLDER]**
- Notes: Quick live demo or screenshots.

## Slide 10 – Testing and Evaluation
- pytest: scoring, percentage, empty quiz, answer checking, classification, invalid input
- Classifier metrics from `models/metrics.json` **[PLACEHOLDER – copy real numbers]**
- Visual: confusion matrix screenshot **[PLACEHOLDER]**
- Notes: Report numbers honestly; mention small test set.

## Slide 11 – Screenshots and Actual Results
- Home, topic selection, quiz, results, classifier, evaluation **[PLACEHOLDERS]**
- Notes: Only show real screenshots.

## Slide 12 – Conclusion, Limitations, Future Scope
- Working app supporting SDG 4; limits: small data, session-only history
- Future: bigger bank, accounts, spaced repetition
- Notes: Thank the audience, invite questions.
