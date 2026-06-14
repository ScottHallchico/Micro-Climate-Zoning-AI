# Phase 4 Execution Report
**Total Runtime**: 2.6 minutes
**Peak RAM**: 2338.7 MB

| Event | Elapsed (s) | RAM (MB) |
|-------|-------------|----------|
| Loading building_climate_features_v3.parquet... | 0.0 | 203.3 |
| Step 1 — Urban Archetype Discovery (MiniBatchKMeans) | 1.3 | 2180.1 |
|   Sweeping K=8..22 with Silhouette / Davies-Bouldin / Calinski-Harabasz... | 1.4 | 2263.5 |
|     K= 8  Sil=0.1950  DBI=1.4826  CHI=174794 | 3.1 | 2306.4 |
|     K= 9  Sil=0.1491  DBI=1.4685  CHI=173178 | 4.4 | 2282.9 |
|     K=10  Sil=0.1254  DBI=1.5344  CHI=172437 | 5.8 | 2308.9 |
|     K=11  Sil=0.1576  DBI=1.3951  CHI=183130 | 7.4 | 2337.6 |
|     K=12  Sil=0.1569  DBI=1.4298  CHI=177029 | 9.0 | 2337.6 |
|     K=13  Sil=0.1504  DBI=1.5305  CHI=164998 | 11.3 | 2337.7 |
|     K=14  Sil=0.1615  DBI=1.4574  CHI=160532 | 13.6 | 2337.8 |
|     K=15  Sil=0.1445  DBI=1.4238  CHI=159579 | 15.7 | 2337.8 |
|     K=16  Sil=0.1386  DBI=1.5030  CHI=152271 | 18.3 | 2337.9 |
|     K=17  Sil=0.1384  DBI=1.4464  CHI=150178 | 19.8 | 2337.9 |
|     K=18  Sil=0.1380  DBI=1.4972  CHI=135731 | 21.3 | 2338.0 |
|     K=19  Sil=0.1457  DBI=1.4707  CHI=137442 | 23.1 | 2338.0 |
|     K=20  Sil=0.1415  DBI=1.4059  CHI=134949 | 25.9 | 2338.1 |
|     K=21  Sil=0.1269  DBI=1.5043  CHI=130344 | 28.5 | 2338.1 |
|     K=22  Sil=0.1372  DBI=1.4637  CHI=137095 | 32.0 | 2338.2 |
|   Best K = 11 (composite rank = 1.67) | 32.1 | 2338.4 |
|   Computing final 100k-sample silhouette score... | 32.8 | 2338.4 |
|   Final Silhouette (100k): 0.1576 | 136.8 | 2338.7 |
| Step 2 — Representative Neighborhood Selection (feature-space centroid) | 137.2 | 1348.1 |
|   Archetype  0: 132,507 members | rep dist=0.167 | complexity=Medium | 142.3 | 2147.8 |
|   Archetype  1: 243,158 members | rep dist=0.153 | complexity=Medium | 143.6 | 2135.9 |
|   Archetype  2: 76,818 members | rep dist=0.249 | complexity=High | 144.8 | 2137.5 |
|   Archetype  3: 20,304 members | rep dist=0.766 | complexity=High | 146.1 | 2137.9 |
|   Archetype  4: 69,946 members | rep dist=0.304 | complexity=High | 147.3 | 2138.5 |
|   Archetype  5: 141,092 members | rep dist=0.186 | complexity=High | 148.6 | 2139.9 |
|   Archetype  6: 130,792 members | rep dist=0.175 | complexity=High | 149.9 | 2141.6 |
|   Archetype  7:  2,970 members | rep dist=1.927 | complexity=High | 151.1 | 2142.9 |
|   Archetype  8: 34,530 members | rep dist=0.548 | complexity=High | 152.3 | 2143.6 |
|   Archetype  9: 164,446 members | rep dist=0.182 | complexity=Medium | 153.6 | 2143.9 |
|   Archetype 10: 66,382 members | rep dist=0.260 | complexity=High | 154.8 | 2145.5 |
|   archetype_report.md written | 155.3 | 2080.9 |
| Generating spatial summary report... | 155.3 | 2080.9 |
| Saving spatial archetype map... | 155.3 | 2081.3 |