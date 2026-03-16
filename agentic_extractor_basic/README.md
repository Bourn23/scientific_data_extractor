Step 1: 03.2 (provenance)
    python data_extractor/03.2.extract_per_system_provenance.py --paper "<name>" --force
    → process_paragraphs_v2_llm_grouping.json

  Step 2: 03 (material features)
    python data_extractor/03.extract_material_features.py --paper "<name>" --force
    → T0_structured_features.csv  (inside the paper folder)

  Step 3: T1 (tier analysis)
    python actionable_analysis/scripts/t1_tier_pipeline.py \
      --input "<paper_folder>/T0_structured_features.csv" \
      --output-dir <output_dir>
    → T1_tiered_dataset.csv, T1_tier_summary.json, plots

  For batch processing across all papers, use --all on steps 1 and 2, then point T1 at the
  consolidated actionable_analysis/material_features_output/T0_structured_features.csv.