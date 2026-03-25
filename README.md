# Defective Products Identification Framework Using Online Reviews

**Authors:** Yawar Abbas & M. S. I. Malik  
**Published in:** *Electronic Commerce Research*, Springer Nature, Vol. 23, No. 2, pp. 899–920, 2023  
**DOI / Publisher Link:** [Springer Nature](https://link.springer.com/journal/10660)  
**Google Scholar:** [Yawar Abbas](https://scholar.google.com/citations?user=sWzIG38AAAAJ&hl=en)

---

## Abstract

The research presents a resilient framework for discovering defective products by analysing consumer reviews on Amazon, incorporating Machine Learning (ML) and Deep Learning (DL) approaches. Through the analysis of more than **50,000 reviews**, Natural Language Processing (NLP) was utilised to suggest innovative characteristics across four feature categories, with **"Derived Attributes"** demonstrating the highest level of importance.

Key results:

- **LSTM** achieved the highest accuracy of **84.61%** with an F1-Score of **0.8427**
- **Random Forest** was the best conventional ML model at **74.72%** accuracy
- Top predictive features: `negate`, `tone`, and `differ`

> *This study highlights the capacity of advanced analytical frameworks to enhance evaluations of product quality and influence future advancements in consumer satisfaction analytics.*

**Keywords:** Amazon · Consumer Reviews · Defective Products · Feature Engineering · NLP · Variable Importance

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background Studies](#2-background-studies)
3. [Research Methodology](#3-research-methodology)
4. [Analytics and Results](#4-analytics-and-results)
5. [Conclusion and Future Work](#5-conclusion-and-future-work)
6. [References](#6-references)
7. [Citation](#citation)

---

## 1. Introduction

In the current era of digital technology, online consumer reviews serve as more than just markers of consumer comfort — they provide a valuable reservoir of data for analysing product quality. The abundance of feedback on platforms such as Amazon offers a distinct opportunity to leverage consumer data for enhancing product standards and detecting problems. Businesses can utilise text mining and NLP tools to methodically analyse these evaluations and derive significant patterns and insights regarding product problems.

The advancement of ML and data analytics techniques has facilitated more intricate examination of unorganised data such as consumer reviews. These tools detect particular phrases and keywords that commonly occur in the context of problems and establish connections to specific product characteristics. This study investigates the capacity of Amazon consumer reviews as a data resource for detecting product defects and examines the efficacy of different text analytics methodologies in identifying and classifying them.

### 1.1 Research Problem

With the rapid expansion of e-commerce, maintaining product quality to guarantee customer satisfaction and brand loyalty has become increasingly critical. This research addresses the following core challenges:

- Developing the most efficient ML model for defect detection using online consumer feedback
- Improving defect identification by extracting new features from existing data
- Evaluating DL models against conventional ML models for handling unstructured customer feedback
- Determining which feature categories most substantially impact defective product identification
- Discovering which characteristics are most indicative of product problems

### 1.2 Research Objectives

The primary aim of this study is to create an all-encompassing ML and DL system to improve defective product identification using online user evaluations. Specifically, this research aims to:

1. Identify the most efficient and robust ML model for detecting product defects
2. Propose innovative feature engineering to discover and create new predictive features
3. Compare DL and conventional ML approaches in handling consumer review text
4. Categorise extracted features and determine which group has the highest predictive influence
5. Identify the most important individual features for defect identification

### 1.3 Research Questions

- Which is the most robust ML model for identifying defective products using online consumer reviews?
- Which novel features can be derived from existing attributes to improve defect identification?
- What is the role of DL over conventional ML models in this task?
- After categorising attributes into groups, which category carries the highest influence?
- Which attributes have the highest feature importance scores for predicting defective products?

---

## 2. Background Studies

### 2.1 Product Defect Identification Overview

Identifying product defects is a vital component of Quality Control (QC) in the manufacturing and retail sectors, directly influencing client satisfaction, brand reputation, and financial performance. Conventional defect detection relies on physical examinations during and after production, methods that are labour-intensive, expensive, and time-consuming.

The digital transformation of commerce has shifted this paradigm. Consumers now highlight product defects in their evaluations, providing a constant and immediate data stream for QC purposes. Historical incidents — such as large-scale recalls by automobile and electronics corporations — emphasise the significant consequences of insufficient defect detection, including financial losses and harm to customer wellbeing.

Recent research has highlighted ML methods that automate defect identification using online evaluations. These models rapidly analyse large volumes of unorganised textual data, detecting prevalent patterns and irregularities that may suggest possible defects. Sophisticated data analytics also enhances understanding of the underlying reasons behind faults, assisting in the improvement of production processes and product designs.

Key challenges remain: the fluctuation of consumer reporting criteria, the subjective character of evaluations, and the difficulty of distinguishing genuine product issues from user errors or rare incidents.

### 2.2 Integration of ML and Quality Control

ML approaches have significantly transformed QC systems across sectors by improving the efficiency and precision of fault identification. Key applications include:

| Approach | Application in QC |
|---|---|
| Supervised Learning (SVM, NN) | Predicting product failures from historical labelled data |
| Unsupervised Learning (Clustering, PCA) | Detecting anomalous patterns in unlabelled production data |
| Convolutional Neural Networks (CNNs) | Visual inspection for surface defect detection in images |
| Recurrent Neural Networks (RNNs) | Sequential text analysis of consumer review data |
| Predictive Maintenance Models | Forecasting equipment failures to prevent defective output |

The integration of ML into QC results in a substantial decrease in the costs associated with quality failures and an enhancement in customer satisfaction. However, challenges remain: model efficacy depends heavily on data quality and quantity, and integrating new technologies into existing production processes often requires substantial investment in infrastructure and training.

### 2.3 Customer Reviews Impact in Market Analysis

Consumer feedback has become an integral component of market intelligence. Key insights from the literature:

- Reviews provide direct qualitative data for improving corporate goals, product design, and customer service
- Text mining and sentiment evaluation uncover patterns enabling categorisation into positive, negative, and neutral signals
- A one-star increase in average product rating can produce a substantial rise in sales
- Negative reviews significantly deter prospective customers and damage brand reputation
- Competitive analysis through sentiment monitoring reveals relative market positioning versus rivals
- Fraudulent reviews undermine data reliability, driving development of detection algorithms
- Proactive engagement with customer reviews improves brand retention and loyalty

### 2.4 ML versus DL: Comparative Analysis

| Dimension | Traditional ML | Deep Learning |
|---|---|---|
| Feature Engineering | Manual extraction required | Automatic representation learning |
| Data Type Strength | Structured, labelled data | Unstructured data (text, images, audio) |
| Scalability | Plateaus with growing data | Improves as data volume grows |
| Interpretability | Higher | Lower (black-box concern) |
| Computational Cost | Lower | Significantly higher |
| Text Classification | Moderate performance | Superior performance |

The primary benefit of DL over regular ML is its capacity to expand in tandem with data volume and complexity. CNNs have transformed visual inspection systems in industrial settings. However, DL's opacity can raise difficulties in high-standards sectors such as pharmaceuticals and aerospace.

Hybrid models are increasingly prevalent — combining DL for feature extraction with interpretable ML models for classification or prediction — yielding synergistic outcomes in predictive maintenance and anomaly detection.

---

## 3. Research Methodology

### 3.1 Dataset Description

| Parameter | Details |
|---|---|
| Source | Publicly available Amazon product reviews repository |
| Category | Electronics |
| Total Records | 50,000 (balanced subset) |
| Positive Reviews (4–5 stars) | 25,000 |
| Critical Reviews (1–3 stars) | 25,000 |

The electronics category was selected for its substantial volume of user feedback and strong influence on consumer market insights. A balanced subset was extracted to enable a fair comparative study of positive and critical consumer feedback.

### 3.2 Dataset Preprocessing

**General Cleaning:**

| Step | Detail |
|---|---|
| Duplicate removal | 804 duplicate entries identified and removed |
| Null value check | No missing data found in any field |
| Text standardisation | All review text converted to lowercase |

**Conventional ML Preprocessing:**

The LIWC (Linguistic Inquiry and Word Count) lexicon was applied to enrich the dataset with **119 diverse attributes** encompassing linguistic, emotional, and psychological dimensions of consumer feedback.

Feature selection techniques applied in Weka:

| Technique | Outcome |
|---|---|
| CFS Attribute | Applied |
| ClassifierAttEval Attributes | Applied |
| ClassifierSubsetEval | Applied |
| Correlation Coefficient | Applied |
| Gain Ratio | Applied |
| **Info Gain** | **Most efficient — selected 16 key features** |

Missing values removed after LIWC extraction:

| Attribute | Missing Entries |
|---|---|
| Clout | 5,548 |
| Authentic | 3,806 |
| Tone | 8,515 |
| Remaining 12 attributes | 35 (combined) |

**DL Preprocessing:**

1. All punctuation and special characters removed (alphanumeric and whitespace retained)
2. Tokenisation applied at word level
3. Stopwords eliminated to emphasise meaningful keywords
4. Sanitised text reintegrated with corresponding review labels

### 3.3 Feature Engineering

Five novel features were engineered using a `safe_convert_to_float` function to convert non-numeric data to floating-point values, capturing subtle psychological and emotional dimensions:

| Engineered Feature | Formula / Logic | Purpose |
|---|---|---|
| **Emotional Intensity** | Average of positive and negative sentiment scores | Measures level of emotional involvement |
| **Review Clarity** | Combined authenticity and cognition measures | Evaluates transparency and straightforwardness |
| **Past Focus Intensity** | Past tense usage combined with review tone | Captures depth of reflection in consumer narratives |
| **Negative Sentiment Ratio** | Negative tone / Positive tone | Identifies dominant negative feelings |
| **Cognitive Complexity** | Cognitive processes combined with linguistic modifiers | Quantifies complexity of thought in reviews |

### 3.4 Feature Categorical Distribution

Features were systematically classified into four groups:

| Category | Features Included |
|---|---|
| **Linguistic Attributes** | Clout, Authentic, auxverb, negate, focuspast |
| **Emotional Tone** | Tone, Affect, tone_pos, tone_neg, emotion, emo_pos, emo_neg |
| **Cognitive Processes** | Cognition, cogproc, differ |
| **Derived Attributes** | EmotionalIntensity, ReviewClarity, PastFocusIntensity, NegativeSentimentRatio, CognitiveComplexity |

#### Table 1 — Complete Feature Descriptions

| Category | Feature | Description |
|---|---|---|
| Linguistic Attributes | Clout | Evaluates degree of influence or assurance expressed in the review |
| Linguistic Attributes | Authentic | Evaluates authenticity and integrity conveyed in the content |
| Linguistic Attributes | auxverb | Calculates auxiliary verbs indicating complexity and mood of the statement |
| Linguistic Attributes | negate | Monitors use of negations — capturing denial or disagreement |
| Linguistic Attributes | focuspast | Examines past tense usage, reflecting on personal encounters |
| Emotional Tone | Tone | Assesses overall emotional sentiment of the content |
| Emotional Tone | Affect | Quantifies existence and strength of emotions |
| Emotional Tone | tone_pos | Detects and tallies instances of favourable emotional reactions |
| Emotional Tone | tone_neg | Detects and tallies instances of negative emotional expressions |
| Emotional Tone | emotion | Conveys overall emotional tone of the material |
| Emotional Tone | emo_pos | Quantifies pleasant feelings — enjoyment or contentment |
| Emotional Tone | emo_neg | Measures negative emotions — unhappiness or discontent |
| Cognitive Processes | Cognition | Indicates extent of cognitive processing required in writing the review |
| Cognitive Processes | cogproc | Quantifies cognitive mechanisms such as logical thinking and categorisation |
| Cognitive Processes | differ | Monitors distinguishing between thoughts — variety of perspectives |
| Derived Attributes | EmotionalIntensity | Mean of positive and negative feelings to achieve emotional equilibrium |
| Derived Attributes | ReviewClarity | Combined authenticity and cognitive measures for clarity assessment |
| Derived Attributes | PastFocusIntensity | Historical perspectives combined with narrative tone |
| Derived Attributes | NegativeSentimentRatio | Ratio of negative to positive tones — key sentiment analysis metric |
| Derived Attributes | CognitiveComplexity | Intricacy of thinking via cognitive processes and linguistic modifiers |

### 3.5 Models and Evaluation Metrics

**Machine Learning Models:**

| Model | Type |
|---|---|
| Logistic Regression | Baseline binary classification |
| Gaussian Naive Bayes | Probabilistic classification |
| AdaBoost | Boosting ensemble |
| XGBoost | Gradient boosting ensemble |
| Decision Tree | Tree-based, non-linear |
| Random Forest | Tree-based ensemble |

**Deep Learning Models:**

| Model | Strength |
|---|---|
| LSTM | Sequential dependency modelling for text |
| Word2Vec + Dense Layers | Semantic word embedding-based classification |

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  
**Validation:** 10-fold cross-validation applied to all models

---

## 4. Analytics and Results

### 4.1 Attributes Correlation

Correlation heatmap analysis (Fig. 4) revealed key attribute relationships with the target class:

| Attribute | Correlation Direction | Interpretation |
|---|---|---|
| ReviewClarity | Positive | Higher clarity → higher probability of defective review |
| CognitiveComplexity | Positive | More complex thought → defect more likely flagged |
| Tone | Negative | Higher tone → associated with non-defective product reviews |
| emo_pos | Negative | Higher positive emotion → associated with satisfied reviews |

### 4.2 Conventional ML Models — Performance Results

All models evaluated using 10-fold cross-validation:

| Model | Accuracy | Precision |
|---|---|---|
| **Random Forest** | **0.7472** | **0.7477** |
| XGBoost | 0.7451 | 0.7451 |
| Logistic Regression | 0.7433 | — |
| AdaBoost | 0.7432 | — |
| Naive Bayes | 0.7153 | — |
| Decision Tree | 0.6642 | — |

> **Random Forest** emerged as the most outstanding ML model, demonstrating superior and consistent performance across accuracy, precision, recall, and F1-Score. (Fig. 5)

### 4.3 Feature Category Analytics — Random Forest (10-fold CV)

| Feature Category | Accuracy |
|---|---|
| **Derived Attributes** | **0.6833** |
| Emotional Tone | 0.6607 |
| Linguistic Attributes | 0.6549 |
| Cognitive Processes | 0.5800 |

> **Finding:** Derived Attributes — the novel engineered features introduced in this study — offered the most comprehensive and powerful combination of signals for identifying defective products. (Fig. 6)

### 4.4 Deep Learning Models — Performance Results

| Model | Accuracy | F1-Score |
|---|---|---|
| **LSTM** | **0.8461** | **0.8427** |
| Word2Vec + Dense Layers | 0.8072 | 0.7954 |
| Random Forest (ML baseline) | 0.7472 | — |

> **LSTM significantly outperformed all models**, demonstrating its expertise in processing sequential data and capturing contextual dependencies within user reviews. Its ability to dynamically model sequences makes it particularly effective for consumer review text. (Fig. 7)

### 4.5 Feature Importance — Top 10 Variables

| Rank | Feature | Category | Role |
|---|---|---|---|
| 1 | `negate` | Linguistic Attributes | Captures subtleties in consumer dissatisfaction — denial and contradiction |
| 2 | `tone` | Emotional Tone | Overall attitude expressed strongly influences perceived product quality |
| 3 | `differ` | Cognitive Processes | Variability in customer opinions carries significant predictive weight |
| 4 | `NegativeSentimentRatio` | Derived Attributes | Proportion of negative to positive tone |
| 5 | `ReviewClarity` | Derived Attributes | Clarity and authenticity of review expression |
| 6–10 | Additional features | Mixed | Moderate importance |

> **Key Insight:** Emotional and cognitive expressions in language are the strongest markers of consumer dissatisfaction. The dominance of `negate` confirms that how customers deny or contradict expectations is the single most powerful predictor of product defects. (Fig. 8)

---

## 5. Conclusion and Future Work

### 5.1 Conclusion

This study formulated an all-encompassing framework for forecasting product defects using both ML and DL techniques applied to Amazon consumer reviews. Key conclusions:

- **Random Forest** was the most efficient conventional ML model, delivering strong and consistent results across all evaluation metrics
- **LSTM** excelled over all models, achieving 84.61% accuracy with minimal preprocessing and feature engineering requirements
- **Derived Attributes** — the novel engineered features introduced in this study — demonstrated higher predictive importance than standard LIWC lexicon variables
- Features `negate`, `tone`, and `differ` were the most influential predictors of product defects
- The feature importance analysis confirmed the efficacy of the engineering approach and its potential for future product quality evaluations

### 5.2 Future Work

Future research directions include:

- Integration of more detailed sentiment analysis algorithms
- Incorporation of multimodal data sources such as review photographs and videos
- Application of transfer learning and advanced architectures such as **Transformers** for improved understanding of complex textual dependencies
- Development of real-time analytics tools that adapt dynamically to fresh user feedback
- Expansion to cross-category and multilingual product review datasets

---

## 6. References

[1] M. Z. Younas, "Defect Identification for Cell Phones Using Product Reviews," CAPITAL UNIVERSITY, 2021.

[2] J. Wang, Y. Ma, L. Zhang, R. X. Gao, and D. Wu, "Deep Learning for Smart Manufacturing: Methods and Applications," *Journal of Manufacturing Systems*, vol. 48, pp. 144–156, 2018.

[3] C. Wang, X. Qin, and A. Gupta, "Developing App from User Feedback using Deep Learning," 2022.

[4] T. Kuo and H.-C. Zhang, "Design for Manufacturability," *IEEE/CPMT International Electronics Manufacturing Technology Symposium*, 1995, pp. 446–459.

[5] Q. H. Duong et al., "Understanding Product Returns: A Systematic Literature Review," *International Journal of Production Economics*, vol. 243, p. 108340, 2022.

[6] E. N. Torres, D. Singh, and A. Robertson-Ring, "Consumer Reviews and the Creation of Booking Transaction Value," *International Journal of Hospitality Management*, vol. 50, pp. 77–83, 2015.

[7] M. Alzate, M. Arce-Urriza, and J. Cebollada, "Mining the Text of Online Consumer Reviews to Analyze Brand Image," *Journal of Retailing and Consumer Services*, vol. 67, p. 102989, 2022.

[8] S. Kumar et al., "Harnessing Machine Learning to Optimize Customer Relations," *International Conference on Micro-Electronics and Telecommunication Engineering*, 2023, pp. 437–446.

[9] C.-P. Tsai et al., "Bridging the Gap Between Big Data System Software Stack and Applications," *IEEE International Conference on Big Data*, 2018, pp. 1865–1874.

[10] A. K. Choudhary, J. A. Harding, and M. K. Tiwari, "Data Mining in Manufacturing: A Review," *Journal of Intelligent Manufacturing*, vol. 20, pp. 501–521, 2009.

[11] H. Pallathadka et al., "Applications of Artificial Intelligence in Business Management, E-Commerce and Finance," *Materials Today Proceedings*, vol. 80, pp. 2610–2613, 2023.

[12] E. Ngai et al., "Decision Support in the Textile and Apparel Supply Chain," *Expert Systems with Applications*, vol. 41, no. 1, pp. 81–91, 2014.

[13] B. Tang, L. Chen, W. Sun, and Z. Lin, "Review of Surface Defect Detection of Steel Products Based on Machine Vision," *International Journal of Precision Engineering*, vol. 17, no. 2, pp. 303–322, 2023.

[14] F. Hodavand, I. J. Ramaji, and N. Sadeghi, "Digital Twin for Fault Detection and Diagnosis of Building Operations," *Buildings*, vol. 13, no. 6, p. 1426, 2023.

[15] J. Lee, H.-A. Kao, and S. Yang, "Service Innovation and Smart Analytics for Industry 4.0 and Big Data Environment," *Procedia CIRP*, vol. 16, pp. 3–8, 2014.

[16] S. M. Mudambi and D. Schuff, "What Makes a Helpful Online Review? A Study of Customer Reviews on Amazon," *MIS Quarterly*, pp. 185–200, 2010.

[17] T. Y. Lee and E. T. Bradlow, "Automated Marketing Research Using Online Customer Reviews," *Journal of Marketing Research*, vol. 48, no. 5, pp. 881–894, 2011.

[18] B. Liu, *Sentiment Analysis and Opinion Mining*. Springer Nature, 2022.

[19] J. A. Chevalier and D. Mayzlin, "The Effect of Word of Mouth on Sales: Online Book Reviews," *Journal of Marketing Research*, vol. 43, no. 3, pp. 345–354, 2006.

[20] A. Ghose and P. G. Ipeirotis, "Estimating the Helpfulness and Economic Impact of Product Reviews," *IEEE Transactions on Knowledge and Data Engineering*, vol. 23, no. 10, pp. 1498–1512, 2010.

[21] N. Jindal and B. Liu, "Opinion Spam and Analysis," *Proceedings of the 2008 International Conference on Web Search and Data Mining*, 2008, pp. 219–230.

[22] G. Cui, H.-K. Lui, and X. Guo, "The Effect of Online Consumer Reviews on New Product Sales," *International Journal of Electronic Commerce*, vol. 17, no. 1, pp. 39–58, 2012.

[23] T. Hennig-Thurau et al., "Electronic Word-of-Mouth via Consumer-Opinion Platforms," *Journal of Interactive Marketing*, vol. 18, no. 1, pp. 38–52, 2004.

[24] G. James, D. Witten, T. Hastie, and R. Tibshirani, *An Introduction to Statistical Learning*. Springer, 2013.

[25] Y. LeCun, Y. Bengio, and G. Hinton, "Deep Learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.

[26] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," *Advances in Neural Information Processing Systems*, vol. 25, 2012.

[27] D. Castelvecchi, "Can We Open the Black Box of AI?" *Nature*, vol. 538, no. 7623, p. 20, 2016.

[28] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.

[29] R. L. Boyd, A. Ashokkumar, S. Seraj, and J. W. Pennebaker, "The Development and Psychometric Properties of LIWC-22," University of Texas at Austin, pp. 1–47, 2022.

[30] Y. R. Tausczik and J. W. Pennebaker, "The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods," *Journal of Language and Social Psychology*, vol. 29, no. 1, pp. 24–54, 2010.

[31] M. L. Newman, C. J. Groom, L. D. Handelman, and J. W. Pennebaker, "Gender Differences in Language Use: An Analysis of 14,000 Text Samples," *Discourse Processes*, vol. 45, no. 3, pp. 211–236, 2008.

[32] C. Zhou et al., "How Does Topic Consistency Affect Online Review Helpfulness?" *Electronic Commerce Research*, vol. 23, no. 4, pp. 2943–2978, 2023.

[33] M. Thelwall, K. Buckley, and G. Paltoglou, "Sentiment in Twitter Events," *Journal of the American Society for Information Science and Technology*, vol. 62, no. 2, pp. 406–418, 2011.

[34] J. W. Pennebaker, R. L. Boyd, K. Jordan, and K. Blackburn, "The Development and Psychometric Properties of LIWC2015," 2015.

[35] M. L. Kern et al., "The Online Social Self: An Open Vocabulary Approach to Personality," vol. 21, no. 2, pp. 158–169, 2014.

[36] **Y. Abbas and M. S. I. Malik, "Defective Products Identification Framework Using Online Reviews," *Electronic Commerce Research*, Springer Nature, vol. 23, no. 2, pp. 899–920, 2023.**

---

## Citation

If you use this work, please cite:

```bibtex
@article{abbas2023defective,
  title     = {Defective Products Identification Framework Using Online Reviews},
  author    = {Abbas, Yawar and Malik, M. S. I.},
  journal   = {Electronic Commerce Research},
  volume    = {23},
  number    = {2},
  pages     = {899--920},
  year      = {2023},
  publisher = {Springer Nature}
}
```

---

## Author

**Yawar Abbas**  
MS Computer Science · COMSATS University Islamabad  
Data Scientist and AI Specialist  
[LinkedIn](https://www.linkedin.com/in/yawarabbasmalik/) · [Google Scholar](https://scholar.google.com/citations?user=sWzIG38AAAAJ&hl=en) · yawar.abbas.malik@gmail.com
