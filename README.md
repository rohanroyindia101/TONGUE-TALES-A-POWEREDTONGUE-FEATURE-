Problem 2: TongueTales – AI-Powered Tongue Feature Extraction and Health Scoring
Background:
Tongue analysis has long been used in Eastern medical systems as a window into the body's internal health. With recent advances in computer vision and deep learning, we now have the opportunity to digitize and automate this traditional diagnostic method. This challenge is to design a model that estimates key tongue features from images and translates them into holistic health scores.
❗ Problem Statement:
You are challenged to build a computer vision system that analyzes a tongue image and extracts five features, each on a scale of 0 (least visible/absent) to 10 (strongly present):
1.
Coated Tongue – Amount of white or yellowish coating
2.
Jagged Tongue Shape – Degree of serrated or uneven edges
3.
Cracks on the Tongue – Number and depth of fissures
4.
Size of Filiform Papillae – Fine hair-like projections on tongue surface
5.
Redness of Fungiform Papillae – Color intensity at red, round papillae
Based on these, compute two final scores: • Nutrition Score • Mantle Score
These scores are functions of the extracted features, simulating internal energy or health as interpreted in traditional medicine.
Note: You do not have access to labeled training data. Your task is unsupervised or weakly supervised modeling, using public datasets, synthetic labeling techniques, or expert-informed heuristics.
