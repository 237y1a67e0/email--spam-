# Email Spam Classifier

This project is an end-to-end Machine Learning and Natural Language Processing (NLP) application designed to automatically detect and filter unsolicited, malicious, or deceptive emails (Spam) from legitimate user correspondence (Ham). By leveraging statistical text processing and pattern recognition algorithms, the system evaluates incoming text streams in real-time.

## 🛠️ Core Functional Architecture
* **Text Refinement:** Normalizes raw email strings by converting characters to lowercase, removing HTML tags/punctuation, and filtering out non-informative filler words.
* **Feature Extraction:** Implements TF-IDF (Term Frequency-Inverse Document Frequency) tokenization to mathematically weigh words, prioritizing unique high-risk triggers over standard language.
* **Predictive Engine:** Utilizes a Multinomial Naive Bayes classifier to calculate the joint probability of an email belonging to a specific class based on its vocabulary composition.
* **UI Deployment:** Features an interactive web-based interface built with Streamlit, enabling users to instantly test custom email bodies.

## 📊 Technical Metrics
* **Data Source:** Trained on the benchmark UCI SMS/Email Spam Collection dataset.
* **High Precision Tuning:** Optimized for maximum precision to guarantee that critical, legitimate consumer emails are almost never accidentally misclassified into the junk folder.


