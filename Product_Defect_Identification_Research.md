# Products Defective Identification Using Machine Learning and Deep Learning Analytics

**Author:** Yawar Abbas  
**Institution:** COMSATS University Islamabad  
**Published in:** Electronic Commerce Research, Springer Nature, Vol. 23, pp. 899–920, 2023  
**Google Scholar:** [View Profile](https://scholar.google.com/citations?user=sWzIG38AAAAJ&hl=en)

---

## Abstract

This research presents a resilient framework for discovering defective products by analysing consumer reviews on Amazon, incorporating Machine Learning (ML) and Deep Learning (DL) approaches. Through the analysis of more than **50,000 reviews**, Natural Language Processing (NLP) was utilised to suggest innovative characteristics across four feature categories, with **"Derived Attributes"** demonstrating the highest level of importance.

Key findings:
- **LSTM model** achieved the highest accuracy of **84.61%** (F1-Score: 0.8427)
- **Random Forest** achieved **74.72%** accuracy among conventional ML models
- Features `negate`, `tone`, and `differ` were the most influential predictors

> *This study highlights the capacity of advanced analytical frameworks to enhance evaluations of product quality and influence future advancements in consumer satisfaction analytics.*

**Keywords:** Amazon · Consumer Reviews · Defective Products · Feature Engineering · NLP · Variable Importance

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background Studies](#2-background-studies)
3. [Research Methodology](#3-research-methodology)
4. [Experiments and Results](#4-experiments-and-results)
5. [Conclusion and Future Work](#5-conclusion-and-future-work)
6. [References](#6-references)

---

## 1. Introduction

In the current era of digital technology, online consumer reviews serve as more than just markers of consumer comfort — they provide a valuable reservoir of data for analysing product quality. The abundance of feedback accessible on platforms such as Amazon offers a distinct opportunity to leverage consumer data for enhancing product standards and detecting problems.

The application of big data in monitoring product quality and detecting problems through consumer feedback is becoming an essential aspect of quality assurance in e-commerce. Consumers often utilise evaluations to convey their levels of contentment and report any experienced problems with their acquisitions. This unsolicited, authentic consumer feedback can offer obvious indicators of quality defects that are frequently overlooked during conventional quality inspections.

### 1.1 Research Problem

Keeping product quality high to guarantee customer satisfaction and brand loyalty has become increasingly critical due to the fast expansion of e-commerce. This research addresses the following core challenges:

- Developing the best ML model for defect detection using online consumer feedback
- Investigating the potential for improving defect identification by extracting new features from existing data
- Evaluating the efficiency of DL models in handling customer feedback compared to conventional ML methods
- Classifying different feature categories and determining which substantially impact defective product identification

### 1.2 Research Objectives

The primary aim of this study is to create an all-encompassing ML and DL system to improve the efficacy of defective product identification using online user evaluations. Specifically, this research aims to:

1. Identify the most efficient ML model for detecting product defects using online consumer reviews
2. Propose innovative feature engineering to discover and create new features from existing data
3. Compare DL and conventional ML approaches in handling textual data from customer evaluations
4. Sort features into groups and identify which categories contribute most to defect identification
5. Identify the most important features for defect identification based on variable importance scoring

### 1.3 Research Questions

- Which is the most robust ML model for identifying defective products using online consumer reviews?
- Which novel features can be derived from existing features to help identify defective products?
- What is the role of DL over conventional ML models in identifying defective products?
- After categorising attributes into different groups, which category has the highest influence?
- Which attributes have the highest feature importance scores for identifying defective products?

---

## 2. Background Studies

### 2.1 Product Defect Identification Overview

Identifying product defects is a vital component of Quality Control (QC) in the manufacturing and retail sectors, directly influencing client satisfaction, brand reputation, and financial performance. Conventional defect detection relies on physical examinations and QC procedures during and after the production process — methods that are labour-intensive, expensive, and time-consuming.

The shift to digital platforms has transformed defect detection. Consumers frequently highlight product defects in their reviews, offering a constant and immediate stream of QC data. ML methods automate and enhance the precision of defect identification from online evaluations by rapidly analysing large quantities of unorganised textual data, detecting prevalent patterns and irregularities that may suggest possible defects.

### 2.2 Integration of ML and Quality Control

ML approaches have significantly transformed QC systems across sectors by improving the efficiency and precision of fault identification:

| Approach | Application |
|---|---|
| Supervised Learning (SVM, NN) | Predicting product failures from historical data |
| Unsupervised Learning (Clustering, PCA) | Detecting anomalous patterns in production data |
| Convolutional Neural Networks (CNNs) | Visual inspection for surface defect detection |
| Recurrent Neural Networks (RNNs) | Sequential text analysis of consumer reviews |

### 2.3 Customer Reviews Impact in Market Analysis

Consumer feedback has become an integral component of market intelligence. Key insights from the literature:

- Text mining and sentiment evaluation uncover significant patterns and customer attitudes
- Reviews are categorised into positive, negative, and neutral groups to identify specific product concerns
- A one-star increase in average product rating can have a substantial effect on sales
- Negative reviews can significantly deter prospective customers and damage brand reputation

### 2.4 ML vs DL Comparative Analysis

| Dimension | Traditional ML | Deep Learning |
|---|---|---|
| Feature Engineering | Manual extraction required | Automatic representation learning |
| Data Type | Best for structured data | Excels with unstructured data (text, images) |
| Interpretability | Higher | Lower |
| Performance on Text | Moderate | Superior |
| Computational Cost | Lower | Higher |

---

## 3. Research Methodology

### 3.1 Dataset Description

| Parameter | Details |
|---|---|
| Source | Amazon product reviews (publicly available) |
| Category | Electronics |
| Total Records | 50,000 (balanced subset) |
| Positive Reviews (4–5 stars) | 25,000 |
| Critical Reviews (1–3 stars) | 25,000 |

The electronics category was selected for its substantial volume of user feedback and strong influence on consumer market insights.

### 3.2 Dataset Preprocessing

**Data Cleaning:**
- 804 duplicate entries identified and removed
- No null values found in any fields
- All review texts standardised to lowercase for uniform text processing

**For ML Models:**
- LIWC (Linguistic Inquiry and Word Count) lexicon applied, yielding 119 diverse attributes
- Feature selection strategies used in Weka: CFS Attribute, ClassifierAttEval, Correlation Coefficient, Gain Ratio, and Info Gain
- **Info Gain** identified as most efficient, selecting **16 important features**
- Missing values removed: `Clout` (5,548), `Authentic` (3,806), `Tone` (8,515)

**For DL Models:**
- Punctuation and special characters removed
- Tokenisation applied (word-level)
- Stopwords eliminated
- Sanitised text reintegrated with review labels

### 3.3 Feature Engineering

Five novel features were engineered to capture subtle psychological and emotional aspects of consumer reviews:

| Engineered Feature | Description |
|---|---|
| **Emotional Intensity** | Average of positive and negative feelings to measure emotional involvement |
| **Review Clarity** | Evaluates transparency using authenticity and cognition measures |
| **Past Focus Intensity** | Combines past tense usage with review tone to show depth of reflection |
| **Negative Sentiment Ratio** | Proportion of negative tones compared to positive tones |
| **Cognitive Complexity** | Considers cognitive processes and linguistic modifiers |

### 3.4 Feature Categorical Distribution

Features were systematically classified into four groups:

| Category | Features | Purpose |
|---|---|---|
| **Linguistic Attributes** | Clout, Authentic, auxverb, negate, focuspast | Linguistic structure and assertiveness |
| **Emotional Tone** | Tone, Affect, tone_pos, tone_neg, emotion, emo_pos, emo_neg | Emotional state and sentiment |
| **Cognitive Processes** | Cognition, cogproc, differ | Cognitive exertion and mental operations |
| **Derived Attributes** | EmotionalIntensity, ReviewClarity, PastFocusIntensity, NegativeSentimentRatio, CognitiveComplexity | Complex engineered relationships |

#### Detailed Feature Descriptions

**Linguistic Attributes:**
- `Clout` — Evaluates the degree of influence or assurance expressed in the review
- `Authentic` — Evaluates the authenticity and integrity conveyed in the content
- `auxverb` — Calculates the number of auxiliary verbs, indicating complexity and mood
- `negate` — Monitors the utilisation of negations, capturing denial or disagreement
- `focuspast` — Examines use of past tense, reflecting on personal encounters

**Emotional Tone:**
- `Tone` — Assesses the overall emotional sentiment of the content
- `Affect` — Quantifies the existence and strength of emotions
- `tone_pos` — Detects instances of favourable emotional reactions
- `tone_neg` — Detects instances of negative emotional expressions
- `emo_pos` — Quantifies pleasant feelings, denoting enjoyment or contentment
- `emo_neg` — Measures negative emotions, expressing unhappiness or discontent

**Cognitive Processes:**
- `Cognition` — Indicates the extent of cognitive processing in writing a review
- `cogproc` — Quantifies the use of cognitive mechanisms such as logical thinking
- `differ` — Monitors the process of distinguishing between thoughts

### 3.5 Models and Evaluation Metrics

**Machine Learning Models:**
- Logistic Regression (baseline)
- Gaussian Naive Bayes
- AdaBoost Classifier
- XGBoost Classifier
- Decision Tree
- Random Forest

**Deep Learning Models:**
- LSTM (Long Short-Term Memory)
- Word2Vec with Dense Layers

**Evaluation Metrics:**
- Accuracy — Overall correctness of the model
- Precision — Capacity to correctly identify positive instances
- Recall — Ability to find all positive instances
- F1-Score — Balanced measure between precision and recall

All models evaluated using **10-fold cross-validation**.

---

## 4. Experiments and Results

### 4.1 Attributes Correlation

Correlation heatmap analysis revealed key attribute relationships with the target class:

- `ReviewClarity` and `CognitiveComplexity` exhibit significant **positive** correlation with defective product labels
- `Tone` and `emo_pos` display **negative** correlation, suggesting higher values are linked to non-defective product reviews

### 4.2 ML Models Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **0.7472** | **0.7477** | — | — |
| XGBoost | 0.7451 | 0.7451 | — | — |
| Logistic Regression | 0.7433 | — | — | — |
| AdaBoost | 0.7432 | — | — | — |
| Naive Bayes | 0.7153 | — | — | — |
| Decision Tree | 0.6642 | — | — | — |

> **Random Forest** emerged as the most outstanding model, consistently demonstrating superior accuracy, precision, recall, and F1-Score.

### 4.3 Feature Category Analysis (Random Forest)

| Feature Category | Accuracy |
|---|---|
| **Derived Attributes** | **0.6833** |
| Emotional Tone | 0.6607 |
| Linguistic Attributes | 0.6549 |
| Cognitive Processes | 0.5800 |

**Finding:** Derived Attributes offered the most comprehensive and powerful combination of signals for identifying defective products.

### 4.4 Deep Learning Models Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| **LSTM** | **0.8461** | **0.8427** | — | — |
| Word2Vec + Dense Layers | 0.8072 | 0.7954 | — | — |
| Random Forest (ML baseline) | 0.7472 | — | — | — |

> **LSTM significantly outperformed all models**, demonstrating its expertise in processing sequential text data and capturing context within user reviews.

### 4.5 Feature Importance Analysis

Top 10 most influential features for defective product identification:

| Rank | Feature | Category | Importance |
|---|---|---|---|
| 1 | `negate` | Linguistic Attributes | Highest |
| 2 | `tone` | Emotional Tone | Very High |
| 3 | `differ` | Cognitive Processes | High |
| 4 | `NegativeSentimentRatio` | Derived Attributes | High |
| 5 | `ReviewClarity` | Derived Attributes | High |
| 6–10 | Additional features | Mixed | Moderate |

**Key Insight:** Emotional and cognitive expressions in language are strong markers of consumer dissatisfaction and product defect signals. The dominance of `negate` confirms that how customers deny or contradict expectations is the single most powerful predictor of product defects.

---

## 5. Conclusion and Future Work

### 5.1 Conclusion

This study formulated an all-encompassing framework for forecasting product defects using both ML and DL techniques applied to Amazon consumer reviews. Key conclusions:

- **Random Forest** was the most efficient conventional ML model, delivering strong and consistent results across all metrics
- **LSTM** excelled in its ability to handle large datasets with minimal preprocessing, achieving 84.61% accuracy
- **Derived Attributes** — the novel engineered features introduced in this study — demonstrated higher predictive importance than standard LIWC lexicon variables
- The features `negate`, `tone`, and `differ` were the most influential predictors of product defects

### 5.2 Future Work

Future research directions include:

- Integration of more detailed sentiment analysis algorithms
- Incorporation of multimodal data sources (photographs, videos from reviews)
- Application of transfer learning and advanced architectures such as **Transformers**
- Development of real-time analytics tools that adapt to fresh user feedback dynamically
- Expanding to cross-category and multilingual product review datasets

---

## 6. References

1. M. Z. Younas, "Defect Identification for Cell Phones Using Product Reviews," CAPITAL UNIVERSITY, 2021.
2. J. Wang et al., "Deep Learning for Smart Manufacturing: Methods and Applications," *Journal of Manufacturing Systems*, vol. 48, pp. 144–156, 2018.
3. C. Wang, X. Qin, and A. Gupta, "Developing App from User Feedback using Deep Learning," 2022.
4. T. Kuo and H.-C. Zhang, "Design for Manufacturability," *IEEE/CPMT International Electronics Manufacturing Technology Symposium*, 1995.
5. Q. H. Duong et al., "Understanding Product Returns: A Systematic Literature Review," *International Journal of Production Economics*, vol. 243, p. 108340, 2022.
6. M. Alzate et al., "Mining the Text of Online Consumer Reviews to Analyze Brand Image," *Journal of Retailing and Consumer Services*, vol. 67, p. 102989, 2022.
7. A. K. Choudhary et al., "Data Mining in Manufacturing: A Review," *Journal of Intelligent Manufacturing*, vol. 20, pp. 501–521, 2009.
8. H. Pallathadka et al., "Applications of AI in Business Management, E-Commerce and Finance," *Materials Today Proceedings*, vol. 80, pp. 2610–2613, 2023.
9. E. Ngai et al., "Decision Support in the Textile and Apparel Supply Chain," *Expert Systems with Applications*, vol. 41, no. 1, pp. 81–91, 2014.
10. B. Tang et al., "Review of Surface Defect Detection of Steel Products Based on Machine Vision," *International Journal of Precision Engineering*, vol. 17, no. 2, pp. 303–322, 2023.
11. S. M. Mudambi and D. Schuff, "What Makes a Helpful Online Review? A Study of Customer Reviews on Amazon," *MIS Quarterly*, pp. 185–200, 2010.
12. B. Liu, *Sentiment Analysis and Opinion Mining*. Springer Nature, 2022.
13. J. A. Chevalier and D. Mayzlin, "The Effect of Word of Mouth on Sales: Online Book Reviews," *Journal of Marketing Research*, vol. 43, no. 3, pp. 345–354, 2006.
14. G. James et al., *An Introduction to Statistical Learning*. Springer, 2013.
15. Y. LeCun, Y. Bengio, and G. Hinton, "Deep Learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.
16. R. L. Boyd et al., "The Development and Psychometric Properties of LIWC-22," University of Texas at Austin, pp. 1–47, 2022.
17. Y. R. Tausczik and J. W. Pennebaker, "The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods," *Journal of Language and Social Psychology*, vol. 29, no. 1, pp. 24–54, 2010.
18. M. L. Newman et al., "Gender Differences in Language Use: An Analysis of 14,000 Text Samples," *Discourse Processes*, vol. 45, no. 3, pp. 211–236, 2008.
19. C. Zhou et al., "How Does Topic Consistency Affect Online Review Helpfulness?" *Electronic Commerce Research*, vol. 23, no. 4, pp. 2943–2978, 2023.
20. M. Thelwall et al., "Sentiment in Twitter Events," *Journal of the American Society for Information Science and Technology*, vol. 62, no. 2, pp. 406–418, 2011.
21. J. W. Pennebaker et al., "The Development and Psychometric Properties of LIWC2015," 2015.
22. M. L. Kern et al., "The Online Social Self: An Open Vocabulary Approach to Personality," vol. 21, no. 2, pp. 158–169, 2014.

---

## Citation

If you use this work, please cite:

```bibtex
@article{abbas2023defective,
  title     = {Defective Products Identification Framework Using Online Reviews},
  author    = {Abbas, Yawar and Malik, M. S. I.},
  journal   = {Electronic Commerce Research},
  volume    = {23},
  pages     = {899--920},
  year      = {2023},
  publisher = {Springer Nature}
}
```

---

## Author

**Yawar Abbas**  
Data Scientist and AI Specialist  
MS Computer Science, COMSATS University Islamabad  
[LinkedIn](https://www.linkedin.com/in/yawarabbasmalik/) · [Google Scholar](https://scholar.google.com/citations?user=sWzIG38AAAAJ&hl=en) · yawar.abbas.malik@gmail.com
