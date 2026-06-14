# Case Traceability Audit

## Missing Columns
The dataset `real_cfd_dataset.parquet` is missing the following required tracking identifiers:
- `simulation_id`
- `case_path`
- `vtk_path`

## Physical Path Verification
Because `case_path` and `vtk_path` do not exist in the DataFrame, 100% of the dataset rows are **Orphan Dataset Rows**. There is no link between an individual row and a distinct OpenFOAM simulation folder.
