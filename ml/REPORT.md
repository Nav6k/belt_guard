# Prediction model: evaluation report

Time split: train on first 40 days, test on the remainder. Horizon: failure within 48 h. Alert threshold 0.5, debounced over 2 x 10 min.

| metric | value |
|---|---|
| rows | 57003 |
| positive_rows | 6065 |
| average_precision | 0.899 |
| row_precision | 0.847 |
| row_recall | 0.865 |
| mode_accuracy_on_detected | 0.998 |
| ttf_mae_hours | 8.0 |
| test_failures | 20 |
| failures_detected | 20 |
| median_lead_time_h | 56.05 |
| false_alarms_per_station_day | 0.0604 |

## Per-failure results (test period)

| station   | mode              | detected   |   lead_time_h |
|:----------|:------------------|:-----------|--------------:|
| ST-01     | splice_failure    | True       |          95.8 |
| ST-02     | splice_failure    | True       |          38.4 |
| ST-02     | belt_misalignment | True       |          22.6 |
| ST-04     | bearing_failure   | True       |          47.8 |
| ST-05     | belt_misalignment | True       |          47.1 |
| ST-05     | belt_misalignment | True       |          56   |
| ST-06     | splice_failure    | True       |          69.2 |
| ST-06     | belt_misalignment | True       |          52   |
| ST-08     | belt_misalignment | True       |          60.9 |
| ST-09     | bearing_failure   | True       |          50.8 |
| ST-10     | bearing_failure   | True       |          58.1 |
| ST-13     | belt_misalignment | True       |          63.9 |
| ST-14     | splice_failure    | True       |          39.1 |
| ST-15     | bearing_failure   | True       |          37.9 |
| ST-16     | bearing_failure   | True       |          26.9 |
| ST-17     | bearing_failure   | True       |          82   |
| ST-18     | bearing_failure   | True       |          67.1 |
| ST-19     | splice_failure    | True       |          56.1 |
| ST-20     | bearing_failure   | True       |          77.4 |
| ST-20     | bearing_failure   | True       |          67.4 |
