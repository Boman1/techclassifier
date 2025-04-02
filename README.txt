Hybrid CTA Classifier with Word2Vec (Encoding Safe)
====================================================

This tool classifies descriptions using:
✅ Word2Vec-powered keyword expansion
✅ Semantic sublayer detection
✅ Encoding-safe CSV reading (UTF-8 & Windows CSV)

-----------------------------
HOW TO USE
-----------------------------
1. Double-click: run_classifier.bat
2. When prompted:
   - Enter path to your Word2Vec model (e.g., GoogleNews-vectors-negative300.bin)
   - Enter path to your CSV file (must include 'Description' column)

-----------------------------
WHAT'S INCLUDED
-----------------------------
- hybrid_cta_classifier_autoinstall_encoding_safe.py (the script)
- run_classifier.bat (easy launcher)
- README.txt (this file)

No setup required. Python will auto-install dependencies.

Results will be saved to: `classified_with_word2vec.csv`
