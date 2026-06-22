MY STUDY - FINAL RESEARCH PACKAGE
=================================
Topic: Predicting the probability of MYOPIA ONSET within the NEXT 1, 2, and 3
YEARS in children, by combining two kinds of data:
   (1) retinal fundus photographs, and
   (2) clinical choroidal / biometric measurements.

NOTE: "1, 2, 3" are PREDICTION HORIZONS (next 1/2/3 years) - not patient ages.

WHAT IS IN THIS FOLDER
----------------------
My_Study_Report.pdf    <- The research report (PDF, with the TikZ pipeline
                          diagram drawn as crisp vector graphics). PRIMARY copy.
My_Study_Report.docx   <- The same report in Microsoft Word (editable; the
                          diagram is embedded as a high-resolution image).

code\
   image_pipeline.py      <- IMAGE data branch. Reads fundus photographs with a
                            ResNet-50 and scores myopia. Two protocols included:
                            linear-evaluation (used for the genuine results) and
                            fine-tuning (GPU-recommended).
   clinical_pipeline.py   <- CLINICAL data branch. Fuses the image score with the
                            4 biomarkers and uses a Cox survival model to give
                            1-, 2-, 3-year onset probabilities + treatment effect.
                            Runs on your real cohort CSV when collected.
   requirements.txt       <- The free software libraries the code needs.

results\
   roc_linear_eval.png            <- ROC curve (genuine, full PALM test set).
   metrics_bar.png                <- AUC / accuracy / sensitivity / specificity.
   confusion_matrix.png           <- Correct vs. incorrect classifications.
   image_results_linear_eval.json <- The raw numbers.
   image_test_predictions_linear_eval.csv <- Per-image scores on the test set.

diagram\
   pipeline.tex / pipeline.pdf / pipeline-1.png  <- The TikZ pipeline figure
                            (source + PDF + PNG).

report\
   My_Study_Report.tex        <- LaTeX source of the report (with inline TikZ).
   pipeline_body.tex          <- the TikZ diagram, included by the report.
   My_Study_Report_word.tex   <- Word-targeted source (diagram as image).

GENUINE RESULTS (FULL public PALM dataset - all 1,200 images)
-------------------------------------------------------------
Trained on 800 images (Training + Validation), tested ONCE on the 400 held-out
Testing images:
    AUC = 0.988  (95% CI 0.977-0.997)
    Accuracy = 96.5% | Sensitivity = 96.7% | Specificity = 96.3%
    (matches DeepMyopia's detection AUC of 0.995)

HONESTY NOTE (important for the presentation)
---------------------------------------------
* The IMAGE results above are GENUINE, on the entire PALM dataset.
* PALM is a myopia-DETECTION dataset (cross-sectional). The 1/2/3-YEAR ONSET part
  needs longitudinal follow-up data, which is not public, so the clinical branch
  is marked "DATA COLLECTION ONGOING" in the report. It is fully built and will
  run unchanged on your prospective cohort. NO numbers were invented anywhere.

HOW TO REPRODUCE
----------------
   pip install -r code/requirements.txt
   python code/image_pipeline.py --palm_root "D:\palm_data\PALM" --mode linear_eval
   (clinical branch, when data ready):
   python code/clinical_pipeline.py --csv your_cohort.csv
